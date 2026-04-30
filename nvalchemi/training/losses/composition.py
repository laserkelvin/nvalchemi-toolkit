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
"""Composable :class:`torch.nn.Module`-based loss-function abstractions.

Leaf loss terms are tensor-to-tensor :class:`BaseLossFunction` instances.
:class:`ComposedLossFunction` is a separate keyed-mapping aggregator that
sums those already-weighted leaves.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

import torch
from torch import nn

from nvalchemi.training.losses.base import LossWeightSchedule


class ComposedLossOutput(TypedDict):
    """Output returned by :class:`ComposedLossFunction`.

    The mapping always contains ``total_loss``. Additional component-name
    keys may also be present and map to weighted loss tensors.
    """

    total_loss: torch.Tensor


def _validate_prediction_target(
    loss: BaseLossFunction,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Validate that prediction and target tensors have matching shapes."""
    prediction_key = getattr(loss, "prediction_key", None)
    target_key = getattr(loss, "target_key", None)
    if pred.shape != target.shape:
        raise ValueError(
            f"{type(loss).__name__}: prediction and target shape mismatch; "
            f"prediction_key={prediction_key!r} has shape {tuple(pred.shape)}, "
            f"target_key={target_key!r} has shape {tuple(target.shape)}."
        )


class BaseLossFunction(nn.Module, abc.ABC):
    """Abstract :class:`torch.nn.Module` base for ALCHEMI loss functions.

    Concrete losses override :meth:`_forward` with tensor-first loss
    logic and return the unweighted loss tensor. The :meth:`forward`
    method will call the user's defined :meth:`_forward`, with shape
    validation on the predictions and targets, then applies the
    scheduled weighting value if specified.

    Addition returns a :class:`ComposedLossFunction`; ``sum([...])`` works
    via :meth:`__radd__`.

    Parameters
    ----------
    weight
        Optional scalar schedule. ``None`` (default) means an identity
        weight of ``1.0``.

    Attributes
    ----------
    weight
        The :class:`LossWeightSchedule` instance or ``None``.
    """

    def __init__(self, *, weight: LossWeightSchedule | None = None) -> None:
        """Initialize the base loss with an optional ``weight`` schedule."""
        super().__init__()
        self.weight: LossWeightSchedule | None = weight

    @abc.abstractmethod
    def _forward(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Return the unweighted loss tensor."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        step: int = 0,
        epoch: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Final wrapper: returns ``current_weight(step, epoch) * _forward(...)``."""
        _validate_prediction_target(self, pred, target)
        w = self.current_weight(step, epoch)
        return w * self._forward(pred, target, step=step, epoch=epoch, **kwargs)

    def current_weight(self, step: int = 0, epoch: int | None = None) -> float:
        """Evaluate the scalar weight to apply at ``(step, epoch)``.

        Returns 1.0 when no schedule is configured; otherwise evaluates
        and validates the configured schedule.

        Parameters
        ----------
        step
            Current global training step.
        epoch
            Current training epoch, or ``None`` when unused.

        Returns
        -------
        float
            Finite scalar weight.

        Raises
        ------
        ValueError
            If a ``per_epoch=True`` schedule is evaluated with
            ``epoch is None``, or the schedule returns a non-finite
            value.
        TypeError
            If the schedule returns a non-numeric value.
        """
        if self.weight is None:
            return 1.0
        schedule = self.weight
        context = type(self).__name__
        if schedule.per_epoch and epoch is None:
            raise ValueError(
                f"epoch must be provided when the {context} loss weight "
                "schedule has per_epoch=True. Pass epoch=<current_epoch> to "
                "the loss, or set per_epoch=False on the schedule."
            )
        try:
            value = schedule(step, epoch or 0)
        except TypeError as exc:
            raise TypeError(
                f"{type(schedule).__name__} does not satisfy the "
                "LossWeightSchedule contract: __call__ must accept "
                "(step: int, epoch: int) and return a float."
            ) from exc
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{type(schedule).__name__} returned {type(value).__name__}; "
                "LossWeightSchedule.__call__ must return float."
            )
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError(
                f"{type(schedule).__name__} for {context} returned non-finite "
                f"weight {coerced!r}; schedules must return finite floats."
            )
        return coerced

    def weight_factors(
        self, step: int = 0, epoch: int | None = None
    ) -> dict[str, float]:
        """Return ``{class_name: current_weight}`` — see :class:`ComposedLossFunction` for the composed form."""
        return {type(self).__name__: self.current_weight(step, epoch)}

    # Arithmetic dunders — return ComposedLossFunction.
    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other`` flattening any existing compositions."""
        if not isinstance(other, (BaseLossFunction, ComposedLossFunction)):
            return NotImplemented
        return ComposedLossFunction(components=_flatten(self) + _flatten(other))

    def __radd__(self, other: Any) -> BaseLossFunction | ComposedLossFunction:
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        return NotImplemented


def _flatten(
    loss: BaseLossFunction | ComposedLossFunction,
) -> tuple[BaseLossFunction, ...]:
    """Return leaf components pulled from a composition, or wrap a bare loss."""
    if isinstance(loss, ComposedLossFunction):
        return tuple(loss.components)
    return (loss,)


def _component_names(components: Sequence[BaseLossFunction]) -> tuple[str, ...]:
    """Return class names with suffixes applied to duplicate component types."""
    raw_names = tuple(type(comp).__name__ for comp in components)
    counts: dict[str, int] = {}
    for name in raw_names:
        counts[name] = counts.get(name, 0) + 1
    next_index: dict[str, int] = {}
    names: list[str] = []
    for name in raw_names:
        if counts[name] > 1:
            idx = next_index.get(name, 0)
            next_index[name] = idx + 1
            names.append(f"{name}_{idx}")
        else:
            names.append(name)
    return tuple(names)


class ComposedLossFunction(nn.Module):
    """Sum of :class:`BaseLossFunction` components.

    The role of this class is to rout keyed prediction/target mappings
    into each component's tensor-first ``forward`` method. Each component's
    schedule fires exactly once, inside that component's own ``forward``.

    Components live in an :class:`torch.nn.ModuleList` for
    ``.modules()`` / ``.state_dict()`` / nested-``__repr__`` support.

    Parameters
    ----------
    components
        Loss terms to combine; must contain at least one element.
    """

    def __init__(
        self,
        components: Sequence[BaseLossFunction | ComposedLossFunction],
    ) -> None:
        """Initialize with ``components``."""
        super().__init__()
        components = tuple(components)
        if len(components) == 0:
            raise ValueError("components must contain at least one loss term")
        for i, comp in enumerate(components):
            if not isinstance(comp, (BaseLossFunction, ComposedLossFunction)):
                raise TypeError(
                    f"components[{i}] must be a BaseLossFunction or "
                    f"ComposedLossFunction, got "
                    f"{type(comp).__name__}"
                )
        # flattening is needed in case we are merging composed losses
        flat_components: list[BaseLossFunction] = []
        for comp in components:
            flat_components.extend(_flatten(comp))

        self.components: nn.ModuleList = nn.ModuleList(flat_components)

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        step: int = 0,
        epoch: int | None = None,
        **kwargs: Any,
    ) -> ComposedLossOutput:
        """Return total and weighted per-component loss contributions."""
        total: torch.Tensor | None = None
        contributions: dict[str, torch.Tensor] = {}
        names = _component_names(tuple(self.components))

        for name, comp in zip(names, self.components, strict=True):
            prediction_key = getattr(comp, "prediction_key", None)
            target_key = getattr(comp, "target_key", None)
            if prediction_key is None:
                raise AttributeError(
                    f"{type(comp).__name__} cannot be used in "
                    "ComposedLossFunction without a prediction_key attribute."
                )
            if target_key is None:
                raise AttributeError(
                    f"{type(comp).__name__} cannot be used in "
                    "ComposedLossFunction without a target_key attribute."
                )
            try:
                pred = predictions[prediction_key]
            except KeyError as exc:
                raise KeyError(
                    f"{type(comp).__name__}: prediction mapping is missing "
                    f"key {prediction_key!r}"
                ) from exc
            try:
                target = targets[target_key]
            except KeyError as exc:
                raise KeyError(
                    f"{type(comp).__name__}: target mapping is missing "
                    f"key {target_key!r}"
                ) from exc
            if not isinstance(pred, torch.Tensor):
                raise TypeError(
                    f"{type(comp).__name__}: prediction mapping key "
                    f"{prediction_key!r} must resolve to torch.Tensor, "
                    f"got {type(pred).__name__}."
                )
            if not isinstance(target, torch.Tensor):
                raise TypeError(
                    f"{type(comp).__name__}: target mapping key "
                    f"{target_key!r} must resolve to torch.Tensor, "
                    f"got {type(target).__name__}."
                )
            _validate_prediction_target(comp, pred, target)
            term = comp(pred, target, step=step, epoch=epoch, **kwargs)

            contributions[name] = term
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError("ComposedLossFunction has no components.")
        contributions["total_loss"] = total

        # cast is mainly for type checking; this is to show that
        # we will guarantee a total_loss key
        return cast(ComposedLossOutput, contributions)

    def weight_factors(
        self, step: int = 0, epoch: int | None = None
    ) -> dict[str, float]:
        """Return a flat dict mapping each component's class name to its weight.

        Duplicate class names get numeric suffixes (``_0``, ``_1``, ...)
        applied to *all* colliding entries, not only the duplicates.
        """
        names = _component_names(tuple(self.components))
        return {
            name: comp.current_weight(step, epoch)
            for name, comp in zip(names, self.components, strict=True)
        }

    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other`` flattening any existing compositions."""
        if not isinstance(other, (BaseLossFunction, ComposedLossFunction)):
            return NotImplemented
        return ComposedLossFunction(components=_flatten(self) + _flatten(other))

    def __radd__(self, other: Any) -> ComposedLossFunction:
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        return NotImplemented

    def extra_repr(self) -> str:
        """Expose component count alongside the default nested-module repr."""
        return f"num_components={len(self.components)}"
