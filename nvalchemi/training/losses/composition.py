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

See :class:`BaseLossFunction` for the full call and arithmetic contract.
"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi.training.losses.base import LossWeightSchedule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nvalchemi.data.batch import Batch


def _validate_static_weights(weights: Any, expected_len: int) -> tuple[float, ...]:
    """Coerce ``weights`` to a tuple of ``expected_len`` finite floats."""
    try:
        items = list(weights)
    except TypeError as exc:
        raise ValueError(
            f"weights must be a list/tuple of floats; got {type(weights).__name__}."
        ) from exc
    for i, w in enumerate(items):
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(
                f"weights[{i}] must be a non-bool int/float; got {type(w).__name__}."
            )
        coerced = float(w)
        if not math.isfinite(coerced):
            raise ValueError(f"weights[{i}] must be finite; got {coerced!r}")
    if len(items) != expected_len:
        raise ValueError(
            f"weights length ({len(items)}) must equal components length "
            f"({expected_len})"
        )
    return tuple(float(w) for w in items)


class BaseLossFunction(nn.Module, abc.ABC):
    """Abstract :class:`torch.nn.Module` base for ALCHEMI loss functions.

    **Subclass contract.** Concrete losses override :meth:`_forward` to
    return the unweighted loss tensor. The public :meth:`forward` is
    final: it returns ``current_weight(step, epoch) * _forward(...)``.
    Do not override :meth:`forward` in a subclass — the language cannot
    enforce this, but doing so breaks the schedule contract.

    Arithmetic (``+``, ``*``, ``/``) returns a
    :class:`ComposedLossFunction`; ``sum([...])`` works via
    :meth:`__radd__`.

    Parameters
    ----------
    weight
        Optional scalar schedule. ``None`` (default) means an identity
        weight of ``1.0``; :meth:`current_weight` skips schedule
        evaluation entirely in that case.

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
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Return the unweighted loss tensor."""

    def forward(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Final wrapper: returns ``current_weight(step, epoch) * _forward(...)``."""
        if self.weight is None:
            # Identity-weight fast path: skip the `1.0 *` scalar multiply.
            return self._forward(batch, step=step, epoch=epoch)
        w = self.current_weight(step, epoch)
        if w == 1.0:
            return self._forward(batch, step=step, epoch=epoch)
        return w * self._forward(batch, step=step, epoch=epoch)

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
        if not isinstance(other, BaseLossFunction):
            return NotImplemented
        left_components, left_weights = _flatten(self)
        right_components, right_weights = _flatten(other)
        return ComposedLossFunction(
            components=left_components + right_components,
            static_weights=left_weights + right_weights,
        )

    def __radd__(self, other: Any) -> BaseLossFunction:
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar: Any) -> ComposedLossFunction:
        """Return ``self * scalar``; scales every component's static weight."""
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            return NotImplemented
        factor = float(scalar)
        if isinstance(self, ComposedLossFunction):
            return ComposedLossFunction(
                components=tuple(self.components),
                static_weights=tuple(w * factor for w in self.static_weights),
            )
        return ComposedLossFunction(components=(self,), static_weights=(factor,))

    def __rmul__(self, scalar: Any) -> ComposedLossFunction:
        """Return ``scalar * self``; delegates to :meth:`__mul__`."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Any) -> ComposedLossFunction:
        """Return ``self / scalar`` as ``(1 / scalar) * self``."""
        if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
            return NotImplemented
        divisor = float(scalar)
        if divisor == 0.0:
            raise ZeroDivisionError("cannot divide a loss by zero")
        return self.__mul__(1.0 / divisor)


def _flatten(
    loss: BaseLossFunction,
) -> tuple[tuple[BaseLossFunction, ...], tuple[float, ...]]:
    """Return ``(components, static_weights)`` pulled from a composition, or wrap a bare loss."""
    if isinstance(loss, ComposedLossFunction):
        return tuple(loss.components), tuple(loss.static_weights)
    return (loss,), (1.0,)


class ComposedLossFunction(BaseLossFunction):
    """Weighted sum of :class:`BaseLossFunction` components.

    A composition does NOT carry its own schedule: outer scheduling, if
    desired, is expressed via ``scalar * (a + b)`` or by multiplying
    the result at the call site. :meth:`current_weight` always returns
    ``1.0`` and raises if :attr:`weight` is ever set on a composition.
    Each component's schedule fires exactly once, inside that
    component's own ``forward``.

    Components live in an :class:`torch.nn.ModuleList` for
    ``.modules()`` / ``.state_dict()`` / nested-``__repr__`` support.

    Parameters
    ----------
    components
        Loss terms to combine; must contain at least one element.
    static_weights
        Static scalars broadcast into the components; length must equal
        ``len(components)``. Defaults to ``(1.0,) * len(components)``.
    """

    def __init__(
        self,
        components: Sequence[BaseLossFunction],
        static_weights: Sequence[float] | None = None,
    ) -> None:
        """Initialize with ``components`` and matching ``static_weights``."""
        super().__init__()
        components = tuple(components)
        if len(components) == 0:
            raise ValueError("components must contain at least one loss term")
        for i, comp in enumerate(components):
            if not isinstance(comp, BaseLossFunction):
                raise TypeError(
                    f"components[{i}] must be a BaseLossFunction, got "
                    f"{type(comp).__name__}"
                )
        self.components: nn.ModuleList = nn.ModuleList(components)
        raw = (1.0,) * len(components) if static_weights is None else static_weights
        self.static_weights: tuple[float, ...] = _validate_static_weights(
            raw, expected_len=len(components)
        )

    def _forward(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Return ``sum(static_w * comp(batch, step=step, epoch=epoch))``."""
        # __init__ enforces at least one component, so next() cannot raise.
        pairs = zip(self.static_weights, self.components, strict=True)
        first_w, first_comp = next(pairs)
        first_term = first_comp(batch, step=step, epoch=epoch)
        # Identity-weight fast path: skip the `1.0 *` on the first term.
        total = first_term if first_w == 1.0 else first_w * first_term
        for static_w, comp in pairs:
            term = comp(batch, step=step, epoch=epoch)
            total = total + (term if static_w == 1.0 else static_w * term)
        return total

    def current_weight(self, step: int = 0, epoch: int | None = None) -> float:  # noqa: ARG002
        """Return 1.0 — compositions do not carry their own schedule.

        Raises
        ------
        RuntimeError
            If :attr:`weight` has been set on this composition.
        """
        if self.weight is not None:
            raise RuntimeError(
                "ComposedLossFunction does not support its own schedule "
                "(weight must be None). Attach schedules to individual "
                "components, or scale the composition via `scalar * composed`."
            )
        return 1.0

    def weight_factors(
        self, step: int = 0, epoch: int | None = None
    ) -> dict[str, float]:
        """Return a flat dict mapping each component's class name to its effective coefficient.

        For a bare component ``c`` the entry is
        ``static_weights[i] * c.current_weight(step, epoch)``. Nested
        compositions are flattened recursively and their factors scaled
        by the outer ``static_weights[i]``. Duplicate class names get
        numeric suffixes (``_0``, ``_1``, ...) applied to *all*
        colliding entries, not only the duplicates. The suffix pass
        runs exactly once at the top level, so mixed
        suffixed/unsuffixed output is not possible.
        """
        pairs = self._flat_weight_factors(step, epoch)
        counts: dict[str, int] = {}
        for name, _ in pairs:
            counts[name] = counts.get(name, 0) + 1
        next_index: dict[str, int] = {}
        out: dict[str, float] = {}
        for name, coef in pairs:
            if counts[name] > 1:
                idx = next_index.get(name, 0)
                next_index[name] = idx + 1
                out[f"{name}_{idx}"] = coef
            else:
                out[name] = coef
        return out

    def _flat_weight_factors(
        self, step: int, epoch: int | None
    ) -> list[tuple[str, float]]:
        """Return raw ``(class_name, effective_coefficient)`` pairs without collision suffixing."""
        collected: list[tuple[str, float]] = []
        for static_w, comp in zip(self.static_weights, self.components, strict=True):
            if isinstance(comp, ComposedLossFunction):
                for name, coef in comp._flat_weight_factors(step, epoch):
                    collected.append((name, static_w * coef))
            else:
                collected.append(
                    (type(comp).__name__, static_w * comp.current_weight(step, epoch))
                )
        return collected

    def extra_repr(self) -> str:
        """Expose static weights alongside the default nested-module repr."""
        return f"static_weights={self.static_weights}"
