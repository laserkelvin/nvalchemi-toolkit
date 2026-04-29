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
"""Composable Pydantic-serializable loss-function abstractions."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi.training.losses.schedules import (
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
    subclasses need not handle weight scheduling directly. The schedule
    receives both counters; its ``per_epoch`` flag determines whether it
    advances by global step or by epoch.

    ``weight`` is typed as the discriminated union
    :data:`~nvalchemi.training.losses.WeightScheduleField` (tagged on the
    ``schedule_type`` literal of each concrete schedule) so that
    subclasses round-trip cleanly through
    :class:`~nvalchemi.training.BaseSpec`.

    Arithmetic operators (``+``, ``*``) build a
    :class:`ComposedLossFunction` that evaluates a weighted sum of
    component losses. ``+`` flattens a composition operand only when
    its outer ``weight`` is the identity ``ConstantWeight(value=1.0)``;
    scheduled compositions are treated as atomic components to preserve
    their schedules.

    Parameters
    ----------
    weight
        Per-step or per-epoch scalar schedule; defaults to
        ``ConstantWeight(value=1.0)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    weight: Annotated[
        WeightScheduleField,
        Field(
            default_factory=lambda: ConstantWeight(value=1.0),
            description=(
                "Per-step or per-epoch scalar schedule multiplied into compute(ctx)."
            ),
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

        ``ctx.epoch`` is coerced from ``None`` to ``0``. The schedule's
        ``per_epoch`` flag determines which counter advances it.

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

    # -----------------------------------------------------------------
    # Arithmetic: build / flatten ComposedLossFunction
    # -----------------------------------------------------------------

    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other``, flattening only identity-weight compositions.

        Both operands must be :class:`BaseLossFunction` instances; any
        other type yields :data:`NotImplemented` so Python falls through
        to the reflected operator or raises :class:`TypeError`.

        A :class:`ComposedLossFunction` operand is unpacked into its
        components and weights **only if** its outer ``weight`` schedule
        is the identity ``ConstantWeight(value=1.0)``. A composition
        with a non-identity schedule is treated as an atomic component
        (with weight ``1.0``) so its outer schedule is preserved rather
        than silently dropped. The resulting composition always has
        outer weight ``ConstantWeight(value=1.0)``.
        """
        if not isinstance(other, BaseLossFunction):
            return NotImplemented
        left_components, left_weights = _composition_terms(self)
        right_components, right_weights = _composition_terms(other)
        return ComposedLossFunction(
            components=left_components + right_components,
            weights=left_weights + right_weights,
            weight=ConstantWeight(value=1.0),
        )

    def __radd__(self, other: Any) -> BaseLossFunction:
        """Support ``sum([...])`` by treating an integer-zero seed as identity.

        Python's :func:`sum` seeds the accumulator with ``0``; returning
        ``self`` unchanged for that case keeps the final composition
        flat. Any other left operand yields :data:`NotImplemented`.
        """
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar: Any) -> ComposedLossFunction:
        """Return a composition with this loss scaled by ``scalar``.

        If ``self`` is already a :class:`ComposedLossFunction`, the
        scalar is broadcast into each component weight while the outer
        schedule is preserved; otherwise the result is a single-component
        composition with weight ``float(scalar)`` and outer
        ``ConstantWeight(value=1.0)``. Non-numeric ``scalar`` values
        yield :data:`NotImplemented`.
        """
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            return NotImplemented
        factor = float(scalar)
        if isinstance(self, ComposedLossFunction):
            return ComposedLossFunction(
                components=self.components,
                weights=tuple(w * factor for w in self.weights),
                weight=self.weight,
            )
        return ComposedLossFunction(
            components=(self,),
            weights=(factor,),
            weight=ConstantWeight(value=1.0),
        )

    def __rmul__(self, scalar: Any) -> ComposedLossFunction:
        """Scalar multiplication is commutative; delegate to :meth:`__mul__`."""
        return self.__mul__(scalar)


def _is_identity_weight(schedule: WeightScheduleField) -> bool:
    """Return ``True`` if ``schedule`` is ``ConstantWeight(value=1.0)``."""
    return isinstance(schedule, ConstantWeight) and schedule.value == 1.0


def _composition_terms(
    loss: BaseLossFunction,
) -> tuple[tuple[BaseLossFunction, ...], tuple[float, ...]]:
    """Return components/weights, flattening only identity-weight compositions."""
    if isinstance(loss, ComposedLossFunction) and _is_identity_weight(loss.weight):
        return loss.components, loss.weights
    return (loss,), (1.0,)


class ComposedLossFunction(BaseLossFunction):
    """Weighted sum of :class:`BaseLossFunction` components.

    The total loss evaluates to

    .. math::

        L(\\mathrm{ctx}) = w_{\\mathrm{outer}}(\\mathrm{step}, \\mathrm{epoch})
                          \\cdot \\sum_i c_i \\cdot L_i(\\mathrm{ctx})

    where :math:`w_{\\mathrm{outer}}` is the inherited ``weight``
    schedule, :math:`c_i` is ``weights[i]``, and :math:`L_i` is
    ``components[i](ctx)``. Each :math:`L_i` call already incorporates
    the component's own ``weight`` schedule, so the outer schedule is
    applied exactly once.

    Parameters
    ----------
    components
        Loss terms to combine; must contain at least one element.
    weights
        Static scalars broadcast into the components; ``len(weights)``
        must equal ``len(components)``.
    weight
        Outer schedule applied once to the weighted sum (inherited).

    Notes
    -----
    Compositions with an identity outer weight are flattened by
    :meth:`__add__`, so ``(a + b) + c`` stores three components rather
    than a tree. A composition with a non-identity outer schedule is
    treated as an atomic component by ``+`` so its schedule survives.
    Scalar multiplication (``float * composed``) rescales every
    component weight in place into a new instance, preserving the
    outer schedule.
    """

    components: Annotated[
        tuple[BaseLossFunction, ...],
        Field(description="Component losses to combine (at least one)."),
    ]
    weights: Annotated[
        tuple[float, ...],
        Field(description="Static scalar applied to each component."),
    ]

    @model_validator(mode="after")
    def _check_components_and_weights(self) -> ComposedLossFunction:
        """Enforce non-empty components and matching weight-vector length."""
        if len(self.components) == 0:
            raise ValueError("components must contain at least one loss term")
        if len(self.weights) != len(self.components):
            raise ValueError(
                f"weights length ({len(self.weights)}) must equal "
                f"components length ({len(self.components)})"
            )
        return self

    def compute(self, ctx: HookContext) -> torch.Tensor:
        """Return the weighted sum of component outputs.

        Accumulates into a single running tensor rather than building
        an intermediate list, so only one autograd node per term is
        added and no stack/concat allocation is needed.

        Parameters
        ----------
        ctx
            Hook context forwarded to each component.

        Returns
        -------
        torch.Tensor
            Scalar loss ``sum(weights[i] * components[i](ctx))``. Does
            NOT multiply by the outer ``weight`` schedule — that is
            applied once by the inherited :meth:`__call__`.
        """
        total = self.weights[0] * self.components[0](ctx)
        for w, comp in zip(self.weights[1:], self.components[1:], strict=True):
            total = total + w * comp(ctx)
        return total
