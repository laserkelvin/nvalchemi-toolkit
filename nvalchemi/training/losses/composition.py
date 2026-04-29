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
"""Composable Pydantic-serializable loss-function abstractions.

Composition semantics (for both :class:`BaseLossFunction.__add__`,
:class:`BaseLossFunction.__mul__`, and :class:`ComposedLossFunction`):

- ``loss_a + loss_b`` produces a :class:`ComposedLossFunction` with two
  components and outer weight ``ConstantWeight(value=1.0)``.
- ``+`` flattens a composition operand **only when** its outer ``weight``
  is the identity ``ConstantWeight(value=1.0)``. A composition with a
  non-identity schedule is treated as an atomic component (with weight
  ``1.0``) so its outer schedule is preserved rather than silently
  dropped.
- ``float * loss`` and ``loss * float`` rescale component weights. If
  the operand is already a :class:`ComposedLossFunction`, the scalar is
  broadcast into each component weight and the outer schedule is
  preserved; otherwise the result is a single-component composition
  with weight ``float(scalar)`` and identity outer weight.
- ``sum([...])`` works because ``__radd__`` returns ``self`` when the
  left operand is integer ``0``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from nvalchemi.training.losses.base import LossWeightSchedule
from nvalchemi.training.losses.schedules import ConstantWeight

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch


class BaseLossFunction(BaseModel, abc.ABC):
    """Abstract Pydantic base for ALCHEMI loss functions.

    Subclasses override :meth:`compute`, which produces the raw
    (unweighted) loss tensor from a
    :class:`~nvalchemi.data.batch.Batch`. The concrete :meth:`__call__`
    returns ``weight(step, epoch or 0) * compute(batch, step=step,
    epoch=epoch)`` so subclasses need not handle weight scheduling
    directly. The schedule receives both counters; its ``per_epoch``
    flag determines whether it advances by global step or by epoch.

    ``weight`` is typed as the
    :class:`~nvalchemi.training.losses.base.LossWeightSchedule` protocol,
    so any callable with the right signature and ``per_epoch`` attribute
    is accepted. Schedules are not round-tripped through a discriminated
    union; upstream ``TrainingStrategy`` reconstructs the schedule
    manually from its ``(instance, spec)`` pair when rebuilding a loss
    from its :class:`~nvalchemi.training.BaseSpec`.

    Arithmetic operators (``+``, ``*``) build a
    :class:`ComposedLossFunction`; see the module-level docstring for
    the precise composition semantics.

    Parameters
    ----------
    weight
        Per-step or per-epoch scalar schedule; defaults to
        ``ConstantWeight(value=1.0)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    weight: Annotated[
        SerializeAsAny[LossWeightSchedule],
        Field(
            default_factory=lambda: ConstantWeight(value=1.0),
            description=(
                "Per-step or per-epoch scalar schedule multiplied into "
                "compute(batch, step=step, epoch=epoch)."
            ),
        ),
    ]

    @abc.abstractmethod
    def compute(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Compute the unweighted loss tensor for ``batch``.

        Concrete subclasses read predictions and targets from ``batch``
        and must preserve autograd. ``step`` and ``epoch`` are available
        for subclasses that need them but are otherwise unused; the
        inherited :meth:`__call__` is responsible for multiplying by the
        scheduled weight.

        Parameters
        ----------
        batch
            Batched graph state carrying predictions and targets.
        step
            Current global training step (keyword-only).
        epoch
            Current training epoch, or ``None`` if unused
            (keyword-only).

        Returns
        -------
        torch.Tensor
            Unweighted loss value.
        """

    def __call__(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Multiply :meth:`compute` by the scheduled weight.

        Raises ``ValueError`` when ``self.weight.per_epoch is True`` and
        ``epoch is None``; raises ``TypeError`` when ``self.weight``
        violates the :class:`LossWeightSchedule` contract (its
        ``__call__`` must accept ``(step, epoch)`` and return a
        numeric scalar).
        """
        if self.weight.per_epoch and epoch is None:
            raise ValueError(
                "epoch must be provided when the loss weight schedule has "
                "per_epoch=True. Pass epoch=<current_epoch> to the loss, "
                "or set per_epoch=False on the schedule."
            )
        try:
            w = self.weight(step, epoch or 0)
        except TypeError as exc:
            raise TypeError(
                f"{type(self.weight).__name__} does not satisfy the "
                "LossWeightSchedule contract: __call__ must accept "
                "(step: int, epoch: int) and return a float."
            ) from exc
        if not isinstance(w, (int, float)):
            raise TypeError(
                f"{type(self.weight).__name__} returned {type(w).__name__}; "
                "LossWeightSchedule.__call__ must return float."
            )
        return w * self.compute(batch, step=step, epoch=epoch)

    # -----------------------------------------------------------------
    # Arithmetic: build / flatten ComposedLossFunction. See the
    # module-level docstring for the full semantics.
    # -----------------------------------------------------------------

    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other`` (see module docstring for semantics)."""
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
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar: Any) -> ComposedLossFunction:
        """Return ``self * scalar`` (see module docstring for semantics)."""
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
        """Delegate to :meth:`__mul__` (scalar multiplication is commutative)."""
        return self.__mul__(scalar)


def _is_identity_weight(schedule: LossWeightSchedule) -> bool:
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

        L(\\mathrm{batch}) = w_{\\mathrm{outer}}(\\mathrm{step},
                             \\mathrm{epoch}) \\cdot \\sum_i c_i \\cdot
                             L_i(\\mathrm{batch})

    where :math:`w_{\\mathrm{outer}}` is the inherited ``weight``
    schedule, :math:`c_i` is ``weights[i]``, and :math:`L_i` is
    ``components[i](batch, step=step, epoch=epoch)``. Each :math:`L_i`
    call already incorporates the component's own ``weight`` schedule,
    so the outer schedule is applied exactly once.

    See the module-level docstring for composition semantics governing
    :meth:`BaseLossFunction.__add__` and :meth:`BaseLossFunction.__mul__`.

    Parameters
    ----------
    components
        Loss terms to combine; must contain at least one element.
    weights
        Static scalars broadcast into the components; ``len(weights)``
        must equal ``len(components)``.
    weight
        Outer schedule applied once to the weighted sum (inherited).
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

    def compute(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Forward ``(batch, step, epoch)`` unchanged to each component and return the weighted sum (accumulated in-place to avoid a list allocation)."""
        total = self.weights[0] * self.components[0](batch, step=step, epoch=epoch)
        for w, comp in zip(self.weights[1:], self.components[1:], strict=True):
            total = total + w * comp(batch, step=step, epoch=epoch)
        return total
