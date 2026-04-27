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

A :class:`LossWeightSchedule` is any callable that maps
``(step, epoch) -> float``. Four concrete Pydantic-validated schedules
are provided: :class:`ConstantWeight`, :class:`LinearWeight`,
:class:`CosineWeight`, and :class:`PiecewiseWeight`, each tagged with a
``type`` literal so they form a discriminated union
(:data:`WeightScheduleField`) that round-trips through
:class:`nvalchemi.training.BaseSpec`.

Adding a new schedule
---------------------
1. Subclass :class:`pydantic.BaseModel` with ``model_config = _SCHEDULE_CONFIG``
   (i.e. ``frozen=True``).
2. Add ``type: Literal["<your_tag>"] = "<your_tag>"`` as the discriminator.
3. Implement ``__call__(step: int, epoch: int) -> float``.
4. Extend :data:`WeightScheduleField` with the new class in the union so
   that :class:`BaseLossFunction.weight` accepts it.

Arbitrary Python callables satisfy :class:`LossWeightSchedule` at runtime
but are **not** accepted by :class:`BaseLossFunction.weight`, which is
typed as the closed discriminated union for serialization purposes.
"""

from __future__ import annotations

import bisect
import math
from typing import Annotated, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


@runtime_checkable
class LossWeightSchedule(Protocol):
    """Runtime-checkable protocol for loss-weight schedules.

    Any callable with signature ``(step: int, epoch: int) -> float``
    satisfies this protocol at runtime (via
    :func:`typing.runtime_checkable`). For serialization-aware storage on
    :class:`~nvalchemi.training.losses.BaseLossFunction.weight`, however,
    only the concrete classes in :data:`WeightScheduleField` are accepted
    — arbitrary callables cannot round-trip through
    :class:`~nvalchemi.training.BaseSpec`. See the module docstring for
    the extension path to add a new schedule.

    Parameters
    ----------
    step
        Current global training step (0-indexed).
    epoch
        Current epoch number (0-indexed); schedules that depend only on
        ``step`` may ignore this argument.

    Returns
    -------
    float
        Scalar weight to apply to the associated loss term.
    """

    def __call__(self, step: int, epoch: int) -> float:
        """Evaluate the schedule at ``(step, epoch)``."""
        ...


# ---------------------------------------------------------------------------
# Concrete schedules (tagged via the ``type`` literal for discriminated unions)
# ---------------------------------------------------------------------------


_SCHEDULE_CONFIG = ConfigDict(frozen=True)

_PositiveSteps: TypeAlias = Annotated[
    int,
    Field(gt=0, description="Positive length of the schedule window in steps."),
]


class ConstantWeight(BaseModel):
    """Schedule that returns the same value at every step."""

    model_config = _SCHEDULE_CONFIG

    type: Literal["constant"] = "constant"
    value: Annotated[float, Field(description="Constant weight value.")]

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Return :attr:`value`, ignoring ``step`` and ``epoch``."""
        return float(self.value)


class LinearWeight(BaseModel):
    """Linear ramp from ``start`` at step 0 to ``end`` at step ``num_steps``.

    Values are clamped to ``start`` for ``step <= 0`` and to ``end`` for
    ``step >= num_steps``.
    """

    model_config = _SCHEDULE_CONFIG

    type: Literal["linear"] = "linear"
    start: Annotated[float, Field(description="Weight at step 0.")]
    end: Annotated[float, Field(description="Weight at step num_steps.")]
    num_steps: _PositiveSteps

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Linear ramp from ``start`` to ``end``, clamped at both ends."""
        if step <= 0:
            return float(self.start)
        if step >= self.num_steps:
            return float(self.end)
        frac = step / self.num_steps
        return float(self.start + (self.end - self.start) * frac)


class CosineWeight(BaseModel):
    """Half-cosine interpolation from ``start`` to ``end`` over ``num_steps``.

    At ``step == 0`` the value is ``start``; at ``step == num_steps`` it
    is ``end``; outside that window the value is clamped.
    """

    model_config = _SCHEDULE_CONFIG

    type: Literal["cosine"] = "cosine"
    start: Annotated[float, Field(description="Weight at step 0.")]
    end: Annotated[float, Field(description="Weight at step num_steps.")]
    num_steps: _PositiveSteps

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Half-cosine interpolation, clamped at both ends."""
        if step <= 0:
            return float(self.start)
        if step >= self.num_steps:
            return float(self.end)
        # Half-cosine: cos(0)=1 at step=0 -> start; cos(pi)=-1 at step=num_steps -> end.
        frac = 0.5 * (1.0 - math.cos(math.pi * step / self.num_steps))
        return float(self.start + (self.end - self.start) * frac)


class PiecewiseWeight(BaseModel):
    """Piecewise-constant schedule over strictly increasing step boundaries.

    For ``boundaries = (b_0, ..., b_{k-1})`` and ``values = (v_0, ..., v_k)``,
    returns ``v_0`` for ``step < b_0``, ``v_1`` for ``b_0 <= step < b_1``,
    and so on. Tuples (rather than lists) keep instances hashable under
    the frozen model config.
    """

    model_config = _SCHEDULE_CONFIG

    type: Literal["piecewise"] = "piecewise"
    boundaries: Annotated[
        tuple[int, ...],
        Field(description="Strictly increasing, non-negative step boundaries."),
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

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Return the value of the interval containing ``step``.

        ``bisect_right`` gives the count of boundaries that ``step`` has
        reached or passed, which is the index into :attr:`values`.
        """
        idx = bisect.bisect_right(self.boundaries, step)
        return float(self.values[idx])


# Discriminated union used by BaseLossFunction's ``weight`` field.
WeightScheduleField: TypeAlias = Annotated[
    ConstantWeight | LinearWeight | CosineWeight | PiecewiseWeight,
    Field(discriminator="type"),
]
