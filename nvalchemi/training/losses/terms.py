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
"""Concrete loss terms: :class:`EnergyLoss`, :class:`ForceLoss`, :class:`StressLoss`.

All three read targets from canonical batch fields (``energy`` /
``forces`` / ``stress``) and predictions from the ``predicted_*``
attribute convention that ``TrainingStrategy`` populates before
invoking the loss. Attribute names are plain ``__init__`` arguments
(``target_key`` / ``prediction_key``) so experiments can rewire them
without subclassing. Composition arithmetic is inherited from
:class:`~nvalchemi.training.losses.composition.BaseLossFunction`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.training.losses.base import LossWeightSchedule
from nvalchemi.training.losses.composition import BaseLossFunction
from nvalchemi.training.losses.reductions import frobenius_mse, per_graph_sum

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.data.batch import Batch


_MISSING: object = object()


def _required_batch_attr(
    batch: Batch,
    attr: str,
    *,
    error_message_factory: Callable[[bool], str],
) -> Any:
    """Return ``batch.<attr>`` or raise ``ValueError`` built lazily on miss.

    ``error_message_factory(is_none)`` is invoked only on the error path;
    its ``is_none`` argument distinguishes "attribute missing" from
    "attribute present but ``None``".
    """
    value = getattr(batch, attr, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(error_message_factory(value is not _MISSING))
    return value


def _get_batch_attr(
    batch: Batch,
    attr: str,
    *,
    loss_name: str,
    role: str,
) -> torch.Tensor:
    """Return ``batch.<attr>`` or raise a prediction/target-style ``ValueError``."""

    def _message(is_none: bool) -> str:
        state = "exists on batch and is None" if is_none else "is missing from batch"
        return (
            f"{loss_name} expected batch.{attr} to be populated but it "
            f"{state}. Populate batch.{attr}, or configure "
            f"{loss_name}({role}_key=...) for your batch schema."
        )

    return _required_batch_attr(batch, attr, error_message_factory=_message)


def _require_graph_metadata(
    batch: Batch,
    attr: str,
    *,
    loss_name: str,
    reason: str,
) -> torch.Tensor | int:
    """Return ``batch.<attr>`` or raise a metadata-style ``ValueError``."""

    def _message(is_none: bool) -> str:  # noqa: ARG001
        return (
            f"{loss_name} ({reason}) requires 'batch.{attr}'; populate it "
            f"on the Batch or reconfigure {loss_name} so it is not needed."
        )

    return _required_batch_attr(batch, attr, error_message_factory=_message)


def _prediction_and_target(
    batch: Batch,
    prediction_key: str,
    target_key: str,
    *,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetch ``(prediction, target)`` tensors and validate their shapes match."""
    pred = _get_batch_attr(
        batch, prediction_key, loss_name=loss_name, role="prediction"
    )
    target = _get_batch_attr(batch, target_key, loss_name=loss_name, role="target")
    if pred.shape != target.shape:
        raise ValueError(
            f"{loss_name}: prediction and target shape mismatch; "
            f"prediction_key={prediction_key!r} has shape {tuple(pred.shape)}, "
            f"target_key={target_key!r} has shape {tuple(target.shape)}."
        )
    return pred, target


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean-squared-error over finite target entries only.

    Uses branch-free tensor ops so the loss is safe under ``torch.compile``:
    no Python ``if`` on tensor values, no boolean indexing, no ``.item()``.
    Target positions with ``NaN`` contribute zero to both numerator and
    denominator; predictions at those positions receive zero gradient.
    When every target entry is ``NaN`` the denominator is clamped to ``1``
    so the loss is ``0.0`` rather than ``NaN``.

    Parameters
    ----------
    pred : torch.Tensor
        Prediction tensor. Expected to be fully finite.
    target : torch.Tensor
        Target tensor of the same shape as ``pred``; may contain ``NaN``
        at positions representing missing labels.

    Returns
    -------
    torch.Tensor
        Scalar mean of squared residuals over valid target entries.
    """
    valid = ~target.isnan()
    residual = torch.where(valid, pred - target, torch.zeros_like(pred))
    denom = valid.to(dtype=pred.dtype).sum().clamp_min(1.0)
    return residual.pow(2).sum() / denom


class EnergyLoss(BaseLossFunction):
    """Mean-squared-error loss on per-graph total energy.

    Energy is already a per-graph quantity of shape ``(B, 1)``, so no
    scatter reduction is needed: :meth:`_forward` returns a plain MSE
    over the batch. When ``per_atom=True``, both prediction and target
    are divided by ``batch.num_nodes_per_graph`` before the MSE, so
    large graphs don't dominate the loss.

    Parameters
    ----------
    target_key : str, default "energy"
        Batch attribute name for the target tensor.
    prediction_key : str, default "predicted_energy"
        Batch attribute name for the model output.
    per_atom : bool, default False
        Divide both energies by ``batch.num_nodes_per_graph`` before MSE.
    ignore_nan : bool, default False
        When ``True``, target entries equal to ``NaN`` are excluded from
        both loss value and gradient (a "nanmean"-style reduction).
        Intended for batches where some samples lack an energy label.
        Implemented with branch-free tensor ops for ``torch.compile``
        compatibility. When every target entry is ``NaN`` the loss is
        ``0.0``.
    weight : LossWeightSchedule, optional
        Scalar schedule applied in :meth:`forward`; ``None`` (default)
        means an identity weight of ``1.0``.
    """

    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        per_atom: bool = False,
        ignore_nan: bool = False,
        weight: LossWeightSchedule | None = None,
    ) -> None:
        """Configure attribute keys and per-atom normalization."""
        super().__init__(weight=weight)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.per_atom = per_atom
        self.ignore_nan = ignore_nan

    def _forward(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:  # noqa: ARG002
        """Return the (optionally per-atom-normalized) energy MSE over the batch."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="EnergyLoss"
        )
        if self.per_atom:
            raw_counts = _require_graph_metadata(
                batch,
                "num_nodes_per_graph",
                loss_name="EnergyLoss(per_atom=True)",
                reason="per_atom=True",
            )
            counts = raw_counts.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
            # ``num_nodes_per_graph`` should already be on ``pred.device`` for
            # peak performance; the ``.to(...)`` above is a safety net. If
            # profiling flags a host→device transfer here, the fix belongs in
            # ``Batch`` collation, not in the loss.
            pred = pred / counts
            target = target / counts
        if self.ignore_nan:
            return _masked_mse(pred, target)
        return (pred - target).pow(2).mean()

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"per_atom={self.per_atom!r}, "
            f"ignore_nan={self.ignore_nan!r}, "
            f"weight={self.weight!r}"
        )


class ForceLoss(BaseLossFunction):
    """Mean-squared-error loss on per-atom forces.

    Both branches return the mean-squared force *component* and differ
    only in whether each graph is weighted equally or each atom is:

    - ``normalize_by_atom_count=True`` (default): per-graph mean of
      squared-component error via :func:`per_graph_sum`, then mean
      over graphs. Small and large graphs contribute equally.
    - ``normalize_by_atom_count=False``: elementwise mean over the full
      ``(V, 3)`` tensor.

    Parameters
    ----------
    target_key : str, default "forces"
        Batch attribute name for the target tensor.
    prediction_key : str, default "predicted_forces"
        Batch attribute name for the model output.
    normalize_by_atom_count : bool, default True
        Divide per-graph force MSE by number of atoms before mean.
    ignore_nan : bool, default False
        When ``True``, target force components equal to ``NaN`` are
        excluded from both loss value and gradient. Intended for batches
        where some atoms/graphs lack force labels. Implemented with
        branch-free tensor ops for ``torch.compile`` compatibility. A
        graph whose entire force tensor is ``NaN`` contributes ``0.0``
        to the loss.
    weight : LossWeightSchedule, optional
        Scalar schedule applied in :meth:`forward`; ``None`` (default)
        means an identity weight of ``1.0``.
    """

    def __init__(
        self,
        *,
        target_key: str = "forces",
        prediction_key: str = "predicted_forces",
        normalize_by_atom_count: bool = True,
        ignore_nan: bool = False,
        weight: LossWeightSchedule | None = None,
    ) -> None:
        """Configure attribute keys and per-graph normalization."""
        super().__init__(weight=weight)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nan = ignore_nan

    def _forward(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:  # noqa: ARG002
        """Return the force-component MSE (optionally graph-balanced)."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="ForceLoss"
        )
        if not self.normalize_by_atom_count:
            if self.ignore_nan:
                return _masked_mse(pred, target)
            return (pred - target).pow(2).mean()
        batch_idx = _require_graph_metadata(
            batch,
            "batch_idx",
            loss_name="ForceLoss(normalize_by_atom_count=True)",
            reason="normalize_by_atom_count=True",
        )
        num_graphs = _require_graph_metadata(
            batch,
            "num_graphs",
            loss_name="ForceLoss(normalize_by_atom_count=True)",
            reason="normalize_by_atom_count=True",
        )
        if self.ignore_nan:
            # Per-component masking: the valid-component count per graph
            # already encodes the per-atom 3-component normalization, so
            # this branch does NOT trail a ``/3.0`` like the dense branch.
            valid = ~target.isnan()
            residual = torch.where(valid, pred - target, torch.zeros_like(pred))
            per_atom_se = residual.pow(2).sum(dim=-1)
            per_atom_valid = valid.to(dtype=pred.dtype).sum(dim=-1)
            per_graph_se_sum = per_graph_sum(
                per_atom_se, batch_idx, num_graphs=num_graphs
            )
            per_graph_valid = per_graph_sum(
                per_atom_valid, batch_idx, num_graphs=num_graphs
            )
            return (per_graph_se_sum / per_graph_valid.clamp_min(1.0)).mean()
        raw_counts = _require_graph_metadata(
            batch,
            "num_nodes_per_graph",
            loss_name="ForceLoss(normalize_by_atom_count=True)",
            reason="normalize_by_atom_count=True",
        )
        per_atom_se = (pred - target).pow(2).sum(dim=-1)
        per_graph_se_sum = per_graph_sum(per_atom_se, batch_idx, num_graphs=num_graphs)
        # See note in ``EnergyLoss._forward``: ``num_nodes_per_graph`` is
        # expected to already be on the same device as ``per_atom_se``. The
        # ``.to(...)`` guards mixed-device batches; if it becomes a
        # bottleneck the fix belongs in ``Batch`` collation.
        counts = raw_counts.to(
            device=per_atom_se.device, dtype=per_atom_se.dtype
        ).clamp_min(1)
        return (per_graph_se_sum / counts).mean() / 3.0

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nan={self.ignore_nan!r}, "
            f"weight={self.weight!r}"
        )


class StressLoss(BaseLossFunction):
    """Mean-squared-error loss on the per-graph stress tensor.

    Both pred and target are shape ``(B, 3, 3)``. The loss is the mean
    of the per-graph squared-Frobenius residual, computed via
    :func:`~nvalchemi.training.losses.reductions.frobenius_mse`.

    Parameters
    ----------
    target_key : str, default "stress"
        Batch attribute name for the target tensor.
    prediction_key : str, default "predicted_stress"
        Batch attribute name for the model output.
    ignore_nan : bool, default False
        When ``True``, target stress components equal to ``NaN`` are
        excluded from both loss value and gradient. Intended for batches
        that mix samples with and without stress labels. Implemented
        with branch-free tensor ops for ``torch.compile`` compatibility.
        A graph whose entire stress tensor is ``NaN`` contributes
        ``0.0`` to the loss.
    weight : LossWeightSchedule, optional
        Scalar schedule applied in :meth:`forward`; ``None`` (default)
        means an identity weight of ``1.0``.
    """

    def __init__(
        self,
        *,
        target_key: str = "stress",
        prediction_key: str = "predicted_stress",
        ignore_nan: bool = False,
        weight: LossWeightSchedule | None = None,
    ) -> None:
        """Configure attribute keys for target and prediction."""
        super().__init__(weight=weight)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.ignore_nan = ignore_nan

    def _forward(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:  # noqa: ARG002
        """Return the mean per-graph Frobenius MSE of the stress tensor."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="StressLoss"
        )
        if self.ignore_nan:
            # Per-component masking on ``(B, 3, 3)``: reduce squared
            # residuals and valid-component counts over the two trailing
            # dims, divide per-graph, then average over graphs. A graph
            # with all-NaN stress has numerator 0 and denominator clamped
            # to 1, contributing zero.
            valid = ~target.isnan()
            residual = torch.where(valid, pred - target, torch.zeros_like(pred))
            per_graph_num = residual.pow(2).sum(dim=(-2, -1))
            per_graph_den = valid.to(dtype=pred.dtype).sum(dim=(-2, -1)).clamp_min(1.0)
            return (per_graph_num / per_graph_den).mean()
        return frobenius_mse(pred, target).mean()

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"ignore_nan={self.ignore_nan!r}, "
            f"weight={self.weight!r}"
        )
