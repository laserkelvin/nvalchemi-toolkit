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
"""Concrete weight schedules for loss functions.

Four Pydantic-validated schedules are provided:
:class:`ConstantWeight`, :class:`LinearWeight`, :class:`CosineWeight`,
and :class:`PiecewiseWeight`. Each schedule fixes ``schedule_type`` to a
literal value so the closed discriminated union
(:data:`WeightScheduleField`) round-trips through
:class:`nvalchemi.training.BaseSpec`.

The concrete schedules always receive both the global step and epoch.
When ``per_epoch=False`` (the default), schedule windows and boundaries
advance by global step. When ``per_epoch=True``, they advance by epoch,
which lets loss weights follow optimizers or learning-rate schedulers
that update once per epoch.

Adding a new schedule
---------------------

You can choose to write your own arbitrary function, providing that
the object is callable with the ``step`` and ``epoch`` integer signature.

Alternatively, you can subclass :class:`~nvalchemi.training.losses.base._BaseWeightSchedule`:

1. Subclass :class:`~nvalchemi.training.losses.base._BaseWeightSchedule`
   to inherit ``per_epoch`` and the frozen Pydantic config.
2. Add ``schedule_type: Literal["<your_tag>"] = Field(default="<your_tag>", frozen=True)``
   as the discriminator.
3. Implement ``__call__(step: int, epoch: int) -> float``; use
   ``self._schedule_index(step, epoch)`` for schedules that advance over
   a training counter.
4. Extend :data:`WeightScheduleField` with the new class in the union so
   that :class:`BaseLossFunction.weight` accepts it.

Arbitrary Python callable objects may satisfy
:class:`~nvalchemi.training.losses.LossWeightSchedule` at runtime but are
**not** accepted by :class:`BaseLossFunction.weight`, which is typed as
the closed discriminated union for serialization purposes.
"""

from __future__ import annotations

import bisect
import math
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from nvalchemi.training.losses.base import _BaseWeightSchedule

_PositiveSteps: TypeAlias = Annotated[
    int,
    Field(
        gt=0,
        description="Positive length of the schedule window in steps or epochs.",
    ),
]


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
        idx = self._map_schedule_index(step, epoch)
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
        idx = self._map_schedule_index(step, epoch)
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
        idx = bisect.bisect_right(
            self.boundaries, self._map_schedule_index(step, epoch)
        )
        return float(self.values[idx])


WeightScheduleField: TypeAlias = Annotated[
    ConstantWeight | LinearWeight | CosineWeight | PiecewiseWeight,
    Field(discriminator="schedule_type"),
]
"""Closed discriminated union accepted by :class:`BaseLossFunction.weight`."""
