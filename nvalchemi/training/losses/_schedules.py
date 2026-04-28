# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight schedules for loss functions.

A :class:`LossWeightSchedule` is any callable object that maps
``(step, epoch) -> float`` and exposes a ``per_epoch`` flag. Four
concrete Pydantic-validated schedules are provided:
:class:`ConstantWeight`, :class:`LinearWeight`, :class:`CosineWeight`,
and :class:`PiecewiseWeight`, each tagged with a ``type`` literal so
they form a discriminated union (:data:`WeightScheduleField`) that
round-trips through :class:`nvalchemi.training.BaseSpec`.

The concrete schedules always receive both the global step and epoch.
When ``per_epoch=False`` (the default), schedule windows and boundaries
advance by global step. When ``per_epoch=True``, they advance by epoch,
which lets loss weights follow optimizers or learning-rate schedulers
that update once per epoch.

Adding a new schedule
---------------------

You can choose to write your own arbitrary function, providing that
the object is callable with the ``step`` and ``epoch`` integer signature.

Alternatively, you can subclass the :class:`_BaseWeightSchedule` through
the following:

1. Subclass :class:`_BaseWeightSchedule` to inherit ``per_epoch`` and
   the frozen Pydantic config.
2. Add ``type: Literal["<your_tag>"] = "<your_tag>"`` as the discriminator.
3. Implement ``__call__(step: int, epoch: int) -> float``; use
   ``self._schedule_index(step, epoch)`` for schedules that advance over
   a training counter.
4. Extend :data:`WeightScheduleField` with the new class in the union so
   that :class:`BaseLossFunction.weight` accepts it.

Arbitrary Python callable objects may satisfy :class:`LossWeightSchedule`
at runtime but are **not** accepted by :class:`BaseLossFunction.weight`,
which is typed as the closed discriminated union for serialization
purposes.
"""

from __future__ import annotations

import bisect
import math
from typing import (
    Annotated,
    Literal,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from pydantic import BaseModel, Field, model_validator


@runtime_checkable
class LossWeightSchedule(Protocol):
    """Runtime-checkable protocol for loss-weight schedules.

    Callable objects with signature ``(step: int, epoch: int) -> float``
    and a ``per_epoch`` attribute satisfy this protocol at runtime (via
    :func:`typing.runtime_checkable`). For serialization-aware storage on
    :class:`~nvalchemi.training.losses.BaseLossFunction.weight`, however,
    only the concrete classes in :data:`WeightScheduleField` are accepted
    ---arbitrary callables cannot round-trip through
    :class:`~nvalchemi.training.BaseSpec`. See the module docstring for
    the extension path to add a new schedule.

    Attributes
    ----------
    per_epoch
        If ``True``, the schedule should advance by ``epoch`` instead of
        by ``step``. This aligns loss-weight updates with training loops
        that update learning-rate schedules once per epoch.

    Parameters
    ----------
    step
        Current global training step (0-indexed).
    epoch
        Current epoch number (0-indexed).

    Returns
    -------
    float
        Scalar weight to apply to the associated loss term.
    """

    per_epoch: Annotated[
        bool,
        "Whether the schedule steps per epoch; if False, schedule will update per step/batch.",
    ]

    def __call__(self, step: int, epoch: int) -> float:
        """Evaluate the schedule at ``(step, epoch)``."""
        ...


# ---------------------------------------------------------------------------
# Concrete schedules (tagged via the ``type`` literal for discriminated unions)
# ---------------------------------------------------------------------------


_PositiveSteps: TypeAlias = Annotated[
    int,
    Field(
        gt=0,
        description="Positive length of the schedule window in steps or epochs.",
    ),
]


class _BaseWeightSchedule(BaseModel):
    """Base Pydantic model for serializable loss-weight schedules.

    Attributes
    ----------
    per_epoch
        If ``False``, schedule windows advance by global step. If
        ``True``, they advance by epoch.
    schedule_type
        Used and defined in the model to help the deserialization
        process, which needs to be set by by field and remain
        immutable.
    """

    model_config = {"frozen": True}

    schedule_type: Annotated[
        str,
        Field(
            description="This field is needed for subclasses to differentiate"
            " between them during (de)serialization.",
            frozen=True,
        ),
    ]
    per_epoch: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether to advance this schedule by epoch instead of by global step."
            ),
        ),
    ] = False

    def _schedule_index(self, step: int, epoch: int) -> int:
        """Return the counter used to advance this schedule."""
        return epoch if self.per_epoch else step


class ConstantWeight(_BaseWeightSchedule):
    """Schedule that returns the same value for every update index."""

    schedule_type: Annotated[
        Literal["constant"], Field(default="constant", frozen=True)
    ]
    value: Annotated[float, Field(description="Constant weight value.")]

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Return :attr:`value`, ignoring ``step`` and ``epoch``."""
        return float(self.value)


class LinearWeight(_BaseWeightSchedule):
    """Linear ramp from ``start`` at index 0 to ``end`` at ``num_steps``.

    The schedule index is the global step when ``per_epoch=False`` and
    the epoch when ``per_epoch=True``. Values are clamped to ``start``
    for index ``<= 0`` and to ``end`` for index ``>= num_steps``.
    """

    schedule_type: Annotated[Literal["linear"], Field(default="linear", frozen=True)]
    start: Annotated[float, Field(description="Weight at schedule index 0.")]
    end: Annotated[float, Field(description="Weight at schedule index `num_steps`.")]
    num_steps: _PositiveSteps

    def __call__(self, step: int, epoch: int) -> float:
        """Linear ramp from ``start`` to ``end``, clamped at both ends."""
        idx = self._schedule_index(step, epoch)
        if idx <= 0:
            return float(self.start)
        if idx >= self.num_steps:
            return float(self.end)
        frac = idx / self.num_steps
        return float(self.start + (self.end - self.start) * frac)


class CosineWeight(_BaseWeightSchedule):
    """Half-cosine interpolation from ``start`` to ``end`` over ``num_steps``.

    The schedule index is the global step when ``per_epoch=False`` and
    the epoch when ``per_epoch=True``. At index ``0`` the value is
    ``start``; at index ``num_steps`` it is ``end``; outside that window
    the value is clamped.
    """

    schedule_type: Annotated[Literal["cosine"], Field(default="cosine", frozen=True)]
    start: Annotated[float, Field(description="Weight at schedule index 0.")]
    end: Annotated[float, Field(description="Weight at schedule index num_steps.")]
    num_steps: _PositiveSteps

    def __call__(self, step: int, epoch: int) -> float:
        """Half-cosine interpolation, clamped at both ends."""
        idx = self._schedule_index(step, epoch)
        if idx <= 0:
            return float(self.start)
        if idx >= self.num_steps:
            return float(self.end)
        # Half-cosine: cos(0)=1 at index=0 -> start; cos(pi)=-1 at num_steps -> end.
        frac = 0.5 * (1.0 - math.cos(math.pi * idx / self.num_steps))
        return float(self.start + (self.end - self.start) * frac)


class PiecewiseWeight(_BaseWeightSchedule):
    """Piecewise-constant schedule over strictly increasing boundaries.

    For ``boundaries = (b_0, ..., b_{k-1})`` and ``values = (v_0, ..., v_k)``,
    returns ``v_0`` for schedule index ``< b_0``, ``v_1`` for
    ``b_0 <= index < b_1``, and so on. The schedule index is the global
    step when ``per_epoch=False`` and the epoch when ``per_epoch=True``.
    Tuples (rather than lists) keep instances hashable under the frozen
    model config.
    """

    schedule_type: Annotated[
        Literal["piecewise"], Field(default="piecewise", frozen=True)
    ]
    boundaries: Annotated[
        tuple[int, ...],
        Field(
            description=(
                "Strictly increasing, non-negative schedule-index boundaries."
            ),
        ),
    ]
    values: Annotated[
        tuple[float, ...],
        Field(description="Values for each interval; length len(boundaries) + 1."),
    ]

    @model_validator(mode="after")
    def _check_boundaries_and_values(self) -> PiecewiseWeight:
        """Enforce strictly-increasing non-negative boundaries and correct length."""
        if len(self.values) != len(self.boundaries) + 1:
            raise ValueError(
                f"values must have length len(boundaries) + 1; got "
                f"len(values)={len(self.values)}, "
                f"len(boundaries)={len(self.boundaries)}"
            )
        prev = -1
        for b in self.boundaries:
            if b < 0:
                raise ValueError(
                    f"boundaries must be non-negative; got {self.boundaries}"
                )
            if b <= prev:
                raise ValueError(
                    f"boundaries must be strictly increasing; got {self.boundaries}"
                )
            prev = b
        return self

    def __call__(self, step: int, epoch: int) -> float:
        """Return the value of the interval containing the schedule index.

        ``bisect_right`` gives the count of boundaries that the index has
        reached or passed, which is the index into :attr:`values`.
        """
        idx = bisect.bisect_right(self.boundaries, self._schedule_index(step, epoch))
        return float(self.values[idx])


# Discriminated union used by BaseLossFunction's ``weight`` field.
# Differentiates based on the `schedule_type` field.
WeightScheduleField: TypeAlias = Annotated[
    ConstantWeight | LinearWeight | CosineWeight | PiecewiseWeight,
    Field(discriminator="schedule_type"),
]
