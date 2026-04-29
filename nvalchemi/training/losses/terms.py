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
"""Concrete loss-term implementations.

This module defines :class:`EnergyLoss`, :class:`ForceLoss`, and
:class:`StressLoss` — the three primary loss terms for ALCHEMI
potentials. They operate on a :class:`~nvalchemi.data.batch.Batch` and
take keyword-only ``step`` and ``epoch`` arguments. Targets are read
from the canonical ``batch`` fields (``energy`` / ``forces`` /
``stress``) and predictions from the ``predicted_*`` attribute
convention that ``TrainingStrategy`` populates on the batch before
invoking the loss. Both the target and prediction attribute names are
configurable Pydantic fields so experiments can rewire them without
subclassing.

All three losses inherit composition arithmetic from
:class:`~nvalchemi.training.losses.composition.BaseLossFunction`:
``1.0 * EnergyLoss() + 10.0 * ForceLoss() + 0.1 * StressLoss()`` builds
a :class:`~nvalchemi.training.losses.composition.ComposedLossFunction`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import torch
from pydantic import Field

from nvalchemi.training.losses.composition import BaseLossFunction
from nvalchemi.training.losses.reductions import frobenius_mse, per_graph_sum

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch


def _get_batch_attr(
    batch: Batch,
    attr: str,
    *,
    loss_name: str,
    role: str,
) -> torch.Tensor:
    """Return ``batch.<attr>`` or raise an actionable ``ValueError`` distinguishing missing-vs-``None``."""
    suggestion = (
        f"Populate batch.{attr}, or configure "
        f"{loss_name}({role}_key=...) for your batch schema."
    )
    if not hasattr(batch, attr):
        raise ValueError(
            f"{loss_name} expected batch.{attr} but the attribute is "
            f"missing from batch. {suggestion}"
        )
    value = getattr(batch, attr)
    if value is None:
        raise ValueError(
            f"{loss_name} expected batch.{attr} to be populated but it "
            f"exists on batch and is None. {suggestion}"
        )
    return value


def _prediction_and_target(
    batch: Batch,
    prediction_key: str,
    target_key: str,
    *,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetch ``(prediction, target)`` tensors from ``batch`` via :func:`_get_batch_attr`."""
    pred = _get_batch_attr(
        batch, prediction_key, loss_name=loss_name, role="prediction"
    )
    target = _get_batch_attr(batch, target_key, loss_name=loss_name, role="target")
    return pred, target


class EnergyLoss(BaseLossFunction):
    """Mean-squared-error loss on per-graph total energy.

    Energy is already a per-graph quantity of shape ``(B, 1)``, so no
    scatter reduction is needed: :meth:`compute` returns a plain MSE
    over the batch. When ``per_atom`` is ``True``, both prediction and
    target are divided by ``batch.num_nodes_per_graph`` before the MSE,
    yielding a per-atom-normalized energy loss.
    """

    target_key: Annotated[
        str,
        Field(description="Name of the target-energy attribute on batch."),
    ] = "energy"
    prediction_key: Annotated[
        str,
        Field(description="Name of the predicted-energy attribute on batch."),
    ] = "predicted_energy"
    per_atom: Annotated[
        bool,
        Field(
            description=(
                "If True, divide predicted and target energy by "
                "batch.num_nodes_per_graph before MSE."
            ),
        ),
    ] = False

    def compute(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Return the (optionally per-atom-normalized) energy MSE over the batch."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="EnergyLoss"
        )
        if self.per_atom:
            counts = batch.num_nodes_per_graph.to(
                device=pred.device, dtype=pred.dtype
            ).unsqueeze(-1)
            pred = pred / counts
            target = target / counts
        return (pred - target).pow(2).mean()


class ForceLoss(BaseLossFunction):
    """Mean-squared-error loss on per-atom forces.

    Both branches return the mean-squared force *component* and differ
    only in whether each graph is weighted equally or each atom is:

    - ``normalize_by_atom_count=True`` (default): per-graph mean of the
      squared-component error, then mean over graphs. Small and large
      graphs contribute equally to the final scalar. Uses
      :func:`~nvalchemi.training.losses.reductions.per_graph_sum` and
      reuses ``batch.num_nodes_per_graph`` (clamped to ≥ 1) as the
      denominator to avoid the extra ``bincount`` inside
      ``per_graph_mean``.
    - ``normalize_by_atom_count=False``: elementwise mean over the
      full ``(V, 3)`` tensor — each atom-component contributes equally
      regardless of graph.
    """

    target_key: Annotated[
        str,
        Field(description="Name of the target-forces attribute on batch."),
    ] = "forces"
    prediction_key: Annotated[
        str,
        Field(description="Name of the predicted-forces attribute on batch."),
    ] = "predicted_forces"
    normalize_by_atom_count: Annotated[
        bool,
        Field(
            description=(
                "If True, per-graph mean of squared error then mean over "
                "graphs (each graph weighted equally). If False, "
                "elementwise mean over all atoms and components."
            ),
        ),
    ] = True

    def compute(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Return the force-component MSE (optionally graph-balanced via per-atom-count normalization)."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="ForceLoss"
        )
        if not self.normalize_by_atom_count:
            return (pred - target).pow(2).mean()
        per_atom_se = (pred - target).pow(2).sum(dim=-1)
        per_graph_se_sum = per_graph_sum(
            per_atom_se, batch.batch_idx, num_graphs=batch.num_graphs
        )
        counts = batch.num_nodes_per_graph.to(
            device=per_atom_se.device, dtype=per_atom_se.dtype
        ).clamp_min(1)
        return (per_graph_se_sum / counts).mean() / 3.0


class StressLoss(BaseLossFunction):
    """Mean-squared-error loss on the per-graph stress tensor.

    Both pred and target are shape ``(B, 3, 3)``. The loss is the mean
    of the per-graph squared-Frobenius residual, computed via
    :func:`~nvalchemi.training.losses.reductions.frobenius_mse`.
    """

    target_key: Annotated[
        str,
        Field(description="Name of the target-stress attribute on batch."),
    ] = "stress"
    prediction_key: Annotated[
        str,
        Field(description="Name of the predicted-stress attribute on batch."),
    ] = "predicted_stress"

    def compute(
        self, batch: Batch, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:
        """Return the mean per-graph Frobenius MSE of the stress tensor."""
        pred, target = _prediction_and_target(
            batch, self.prediction_key, self.target_key, loss_name="StressLoss"
        )
        return frobenius_mse(pred, target).mean()
