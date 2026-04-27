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

Per-graph denominators (node counts) are taken as an explicit argument
rather than recomputed; callers typically pass :attr:`Batch.num_nodes_per_graph`
directly.

Notes
-----
- ``batch_idx`` values must lie in ``[0, num_graphs)``. Out-of-range
  indices surface as a :class:`RuntimeError` from ``scatter_add_``.
- On CUDA, ``scatter_add_`` accumulates via atomics, so results are
  nondeterministic at the last-bit level unless
  :func:`torch.use_deterministic_algorithms` is enabled (in which case it
  raises). The planned swap to ``nvalchemi.math.segment_ops`` will offer
  a deterministic variant.
"""

# TODO: swap ``scatter_add_`` for ``nvalchemi.math.segment_ops`` when available.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float, Integer


def per_graph_sum(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: Integer[torch.Tensor, "V"],  # noqa: F722
    num_graphs: int,
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
        Positive number of graphs; sets the output leading dim.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph sums of shape ``(num_graphs, *values.shape[1:])``.

    Raises
    ------
    ValueError
        If ``num_graphs <= 0`` or if ``values`` and ``batch_idx`` disagree
        on their leading dim.
    """
    if num_graphs <= 0:
        raise ValueError(f"num_graphs must be positive, got {num_graphs}")
    if values.shape[0] != batch_idx.shape[0]:
        raise ValueError(
            f"values leading dim ({values.shape[0]}) must match "
            f"batch_idx length ({batch_idx.shape[0]})"
        )
    out_shape = (num_graphs, *values.shape[1:])
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    # Expand batch_idx to match `values` trailing dims for scatter_add_.
    index = batch_idx.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    out.scatter_add_(0, index, values)
    return out


def per_graph_mean(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: Integer[torch.Tensor, "V"],  # noqa: F722
    num_graphs: int,
    num_nodes_per_graph: Integer[torch.Tensor, "B"],  # noqa: F722
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
        Number of graphs.
    num_nodes_per_graph
        Node count per graph; must have length ``num_graphs``.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph means.

    Raises
    ------
    ValueError
        If ``num_nodes_per_graph`` length does not equal ``num_graphs``.
    """
    if num_nodes_per_graph.shape[0] != num_graphs:
        raise ValueError(
            f"num_nodes_per_graph length ({num_nodes_per_graph.shape[0]}) "
            f"must equal num_graphs ({num_graphs})"
        )
    totals = per_graph_sum(values, batch_idx, num_graphs)
    counts = num_nodes_per_graph.to(totals.dtype).clamp_min(1.0)
    # Broadcast counts across trailing dims of totals.
    counts = counts.view(-1, *([1] * (totals.ndim - 1)))
    return totals / counts


def per_graph_mse(
    pred: Float[torch.Tensor, "V ..."],  # noqa: F722
    target: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: Integer[torch.Tensor, "V"],  # noqa: F722
    num_graphs: int,
    num_nodes_per_graph: Integer[torch.Tensor, "B"],  # noqa: F722
) -> Float[torch.Tensor, "B"]:  # noqa: F722
    """Per-graph MSE of ``pred`` vs ``target``.

    Computes ``sum squared error per graph / element count per graph``,
    where the denominator is ``num_nodes_per_graph * prod(trailing_dims)``.

    Parameters
    ----------
    pred, target
        Same-shape per-node tensors.
    batch_idx
        Graph index for each node.
    num_graphs
        Number of graphs.
    num_nodes_per_graph
        Node count per graph (e.g. ``Batch.num_nodes_per_graph``); must
        have length ``num_graphs``.

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph MSE values.

    Raises
    ------
    ValueError
        If ``pred.shape != target.shape`` or
        ``num_nodes_per_graph`` length does not equal ``num_graphs``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    if num_nodes_per_graph.shape[0] != num_graphs:
        raise ValueError(
            f"num_nodes_per_graph length ({num_nodes_per_graph.shape[0]}) "
            f"must equal num_graphs ({num_graphs})"
        )
    squared_error = (pred - target).pow(2)
    # Collapse trailing dims to one scalar per node, then scatter.
    squared_error_per_node = (
        squared_error.flatten(1).sum(dim=1) if squared_error.ndim > 1 else squared_error
    )
    squared_error_per_graph = per_graph_sum(
        squared_error_per_node, batch_idx, num_graphs
    )
    trailing = math.prod(pred.shape[1:]) if pred.ndim > 1 else 1
    num_entries_per_graph = (
        num_nodes_per_graph.to(
            device=squared_error_per_graph.device,
            dtype=squared_error_per_graph.dtype,
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
