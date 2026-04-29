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
"""Graph-aware scatter-based reduction primitives for loss functions.

All helpers operate on flat per-node tensors with a ``batch_idx`` mapping
each node to its graph and reduce via :func:`torch.Tensor.scatter_add_`
into a pre-allocated output tensor: no Python-level iteration over graphs,
and autograd flows through the scatter.

Per-graph denominators (node counts) are derived from ``batch_idx``. Callers
only need to pass ``num_graphs`` when ``batch_idx`` cannot encode the full
batch shape, such as trailing empty graphs or an entirely empty batch.

Note
---------------
Currently the methods use `torch.scatter_*` - the goal is to use
`nvalchemiops` segment operations once they support backwards.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias

import torch

from nvalchemi._typing import BatchIndices

if TYPE_CHECKING:
    from jaxtyping import Float

_NumGraphs: TypeAlias = int | torch.Tensor


def _resolve_batch_indices(
    batch_idx: BatchIndices,
    num_graphs: int | None,
    device: torch.device,
) -> tuple[BatchIndices, _NumGraphs]:
    """Return ``batch_idx`` on ``device`` and the graph count it implies."""
    batch_idx = batch_idx.to(device=device, dtype=torch.long)
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D, got shape {tuple(batch_idx.shape)}")
    if num_graphs is not None and num_graphs <= 0:
        raise ValueError(f"num_graphs must be positive, got {num_graphs}")
    if batch_idx.numel() == 0:
        if num_graphs is None:
            raise ValueError("Cannot infer num_graphs from empty batch_idx")
        return batch_idx, num_graphs
    if torch.compiler.is_compiling():
        return batch_idx, batch_idx.max() + 1 if num_graphs is None else num_graphs
    min_idx = int(batch_idx.min().item())
    if min_idx < 0:
        raise ValueError(f"batch_idx values must be non-negative, got {min_idx}")
    inferred_num_graphs = int(batch_idx.max().item()) + 1
    if num_graphs is not None and inferred_num_graphs > num_graphs:
        raise ValueError(
            f"batch_idx contains graph index {inferred_num_graphs - 1}, "
            f"but num_graphs={num_graphs}"
        )
    return batch_idx, inferred_num_graphs if num_graphs is None else num_graphs


def _check_leading_dim(
    values: torch.Tensor,
    batch_idx: BatchIndices,
    *,
    name: str,
) -> None:
    """Validate that ``values`` and ``batch_idx`` have matching leading dims."""
    if values.shape[0] != batch_idx.shape[0]:
        raise ValueError(
            f"{name} leading dim ({values.shape[0]}) must match "
            f"batch_idx length ({batch_idx.shape[0]})"
        )


def _per_graph_sum_resolved(
    values: torch.Tensor,
    batch_idx: BatchIndices,
    num_graphs: _NumGraphs,
) -> torch.Tensor:
    """Sum per-node values after ``batch_idx`` and ``num_graphs`` are resolved."""
    out_shape = (num_graphs, *values.shape[1:])
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    index = batch_idx.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    out.scatter_add_(0, index, values)
    return out


def _num_nodes_per_graph(
    batch_idx: BatchIndices,
    num_graphs: _NumGraphs,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Count nodes per graph from ``batch_idx``."""
    ones = torch.ones(batch_idx.shape[0], dtype=dtype, device=device)
    return _per_graph_sum_resolved(ones, batch_idx, num_graphs)


def per_graph_sum(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Sum per-node values into per-graph values via ``scatter_add_``.

    Trailing dims of ``values`` are preserved in the output.

    Parameters
    ----------
    values
        Per-node values; leading dim indexes nodes.
    batch_idx
        Graph index for each node.
    num_graphs
        Optional positive number of graphs. When omitted, inferred as
        ``batch_idx.max() + 1``. Pass explicitly to preserve trailing empty
        graphs or to reduce an empty ``batch_idx``.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph sums of shape ``(num_graphs, *values.shape[1:])``.

    Raises
    ------
    ValueError
        If ``values`` and ``batch_idx`` disagree on their leading dim, if
        ``num_graphs`` is invalid, or if ``num_graphs`` cannot be inferred.
    """
    _check_leading_dim(values, batch_idx, name="values")
    batch_idx, resolved_num_graphs = _resolve_batch_indices(
        batch_idx,
        num_graphs,
        values.device,
    )
    return _per_graph_sum_resolved(values, batch_idx, resolved_num_graphs)


def per_graph_mean(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Mean of per-node values across each graph.

    Empty graphs (zero nodes) are safe: their sum is zero and their count
    is clamped to ``1`` before the division, so they yield zero.

    Parameters
    ----------
    values
        Per-node values.
    batch_idx
        Graph index for each node.
    num_graphs
        Optional positive number of graphs. When omitted, inferred as
        ``batch_idx.max() + 1``. Pass explicitly to preserve trailing empty
        graphs or to reduce an empty ``batch_idx``.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph means.

    Raises
    ------
    ValueError
        If ``values`` and ``batch_idx`` disagree on their leading dim, if
        ``num_graphs`` is invalid, or if ``num_graphs`` cannot be inferred.
    """
    _check_leading_dim(values, batch_idx, name="values")
    batch_idx, resolved_num_graphs = _resolve_batch_indices(
        batch_idx,
        num_graphs,
        values.device,
    )
    totals = _per_graph_sum_resolved(values, batch_idx, resolved_num_graphs)
    counts = _num_nodes_per_graph(
        batch_idx,
        resolved_num_graphs,
        dtype=totals.dtype,
        device=totals.device,
    ).clamp_min(1.0)
    # Broadcast counts across trailing dims of totals.
    counts = counts.view(-1, *([1] * (totals.ndim - 1)))
    return totals / counts


def per_graph_mse(
    pred: Float[torch.Tensor, "V ..."],  # noqa: F722
    target: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B"]:  # noqa: F722
    """Per-graph MSE of ``pred`` vs ``target``.

    Computes ``sum squared error per graph / element count per graph``,
    where the denominator is ``nodes_in_graph * prod(trailing_dims)``.

    Parameters
    ----------
    pred, target
        Same-shape per-node tensors.
    batch_idx
        Graph index for each node.
    num_graphs
        Optional positive number of graphs. When omitted, inferred as
        ``batch_idx.max() + 1``. Pass explicitly to preserve trailing empty
        graphs or to reduce an empty ``batch_idx``.

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph MSE values.

    Raises
    ------
    ValueError
        If ``pred.shape != target.shape``, if ``pred`` and ``batch_idx``
        disagree on their leading dim, if ``num_graphs`` is invalid, or
        if ``num_graphs`` cannot be inferred.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    _check_leading_dim(pred, batch_idx, name="pred")
    batch_idx, resolved_num_graphs = _resolve_batch_indices(
        batch_idx,
        num_graphs,
        pred.device,
    )
    squared_error = (pred - target).pow(2)
    # Collapse trailing dims to one scalar per node, then scatter.
    squared_error_per_node = (
        squared_error.flatten(1).sum(dim=1) if squared_error.ndim > 1 else squared_error
    )
    squared_error_per_graph = _per_graph_sum_resolved(
        squared_error_per_node, batch_idx, resolved_num_graphs
    )
    trailing = math.prod(pred.shape[1:]) if pred.ndim > 1 else 1
    num_entries_per_graph = (
        _num_nodes_per_graph(
            batch_idx,
            resolved_num_graphs,
            dtype=squared_error_per_graph.dtype,
            device=squared_error_per_graph.device,
        )
        .mul(trailing)
        .clamp_min_(1.0)
    )
    return squared_error_per_graph / num_entries_per_graph


def frobenius_mse(
    pred: Float[torch.Tensor, "B 3 3"],  # noqa: F722
    target: Float[torch.Tensor, "B 3 3"],  # noqa: F722
) -> Float[torch.Tensor, "B"]:  # noqa: F722
    """Per-graph squared-Frobenius MSE over the last two dims.

    Returns ``((pred - target) ** 2).mean(dim=(-2, -1))`` — the squared
    Frobenius norm of the residual matrix, averaged over its entries.

    Parameters
    ----------
    pred, target
        Same-shape matrix-valued tensors (e.g. stress of shape ``(B, 3, 3)``).

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph Frobenius MSE.

    Raises
    ------
    ValueError
        If shapes differ or input has fewer than three dims.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    if pred.ndim < 3:
        raise ValueError(
            f"frobenius_mse expects at least 3 dims (B, ..., M1, M2); "
            f"got shape {tuple(pred.shape)}"
        )
    return (pred - target).pow(2).mean(dim=(-2, -1))
