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
"""Abstract base class for Pydantic-serializable loss functions."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.training.losses._schedules import (
    ConstantWeight,
    WeightScheduleField,
)

if TYPE_CHECKING:
    from nvalchemi.hooks._context import HookContext


class BaseLossFunction(BaseModel, abc.ABC):
    """Abstract Pydantic base for ALCHEMI loss functions.

    Subclasses override :meth:`compute`, which produces the raw
    (unweighted) loss tensor from a
    :class:`~nvalchemi.hooks._context.HookContext`. The concrete
    :meth:`__call__` returns
    ``weight(ctx.step_count, ctx.epoch or 0) * compute(ctx)`` so
    subclasses need not handle weight scheduling directly.

    ``weight`` is typed as the discriminated union
    :data:`~nvalchemi.training.losses.WeightScheduleField` (tagged on the
    ``type`` literal of each concrete schedule) so that subclasses
    round-trip cleanly through
    :class:`~nvalchemi.training.BaseSpec`.

    Parameters
    ----------
    weight
        Per-step scalar schedule; defaults to
        ``ConstantWeight(value=1.0)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    weight: Annotated[
        WeightScheduleField,
        Field(
            default_factory=lambda: ConstantWeight(value=1.0),
            description="Per-step scalar schedule multiplied into compute(ctx).",
        ),
    ]

    @abc.abstractmethod
    def compute(self, ctx: HookContext) -> torch.Tensor:
        """Compute the unweighted loss tensor for ``ctx``.

        Concrete subclasses read predictions and targets from
        ``ctx.batch`` and must preserve autograd.

        Parameters
        ----------
        ctx
            Hook context carrying ``batch``, ``step_count``, ``epoch``,
            and other training state.

        Returns
        -------
        torch.Tensor
            Unweighted loss value.
        """

    def __call__(self, ctx: HookContext) -> torch.Tensor:
        """Evaluate :meth:`compute` and multiply by the scheduled weight.

        ``ctx.epoch`` is coerced from ``None`` to ``0``.

        Parameters
        ----------
        ctx
            Hook context with ``step_count`` and optional ``epoch``.

        Returns
        -------
        torch.Tensor
            ``weight(step, epoch) * compute(ctx)``.
        """
        w = self.weight(ctx.step_count, ctx.epoch or 0)
        return w * self.compute(ctx)
