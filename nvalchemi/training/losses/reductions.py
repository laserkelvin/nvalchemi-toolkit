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
into a pre-allocated output tensor: no Python-level iteration over
graphs, and autograd flows through the scatter. Per-graph denominators
are counted with :func:`torch.bincount` (single kernel, no auxiliary
allocation).

Common parameters
-----------------

All public reductions share the following signature:

- ``values`` (or ``pred`` / ``target``): per-node tensor whose leading
  dim indexes nodes; trailing dims are reduced or preserved depending on
  the specific helper.
- ``batch_idx``: 1-D ``BatchIndices`` mapping each node to its graph.
  For the GPU hot path, callers should ensure ``batch_idx`` is already a
  CUDA ``long`` tensor; the defensive ``.to(device=..., dtype=long)``
  cast below is a no-op for well-formed inputs.
- ``num_graphs`` (optional): when supplied, the helpers trust it and
  perform no validation scan of ``batch_idx``. This is the
  recommended hot-path calling convention because it avoids any
  GPU→CPU synchronization. When omitted, ``num_graphs`` is inferred
  as ``batch_idx.max().item() + 1``, which forces a device sync and
  should be avoided inside per-step training loops. Empty
  ``batch_idx`` always requires ``num_graphs`` to be supplied.

All public reductions raise :class:`ValueError` on shape mismatch, on
non-positive ``num_graphs``, or on inability to infer ``num_graphs``.

Note
----
Currently the methods use ``torch.scatter_*``; the goal is to use
``nvalchemiops`` segment operations once they support backwards.
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
    """Return ``batch_idx`` on ``device`` and the graph count it implies.

    When ``num_graphs`` is supplied, trust it and skip any scan of
    ``batch_idx`` — this is the hot path. When omitted, infer via
    ``batch_idx.max()``; under ``torch.compile`` the max stays a tensor,
    otherwise it is materialized on the host (forcing a device sync).
    """
    batch_idx = batch_idx.to(device=device, dtype=torch.long)
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D, got shape {tuple(batch_idx.shape)}")
    if num_graphs is not None:
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")
        return batch_idx, num_graphs
    if batch_idx.numel() == 0:
        raise ValueError(
            "Cannot infer num_graphs from empty batch_idx; "
            "pass num_graphs explicitly when reducing an empty batch."
        )
    if torch.compiler.is_compiling():
        return batch_idx, batch_idx.max() + 1
    return batch_idx, int(batch_idx.max().item()) + 1


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


def _prep_reduction(
    values: torch.Tensor,
    batch_idx: BatchIndices,
    num_graphs: int | None,
    *,
    name: str,
) -> tuple[BatchIndices, _NumGraphs]:
    """Validate leading dim and resolve ``(batch_idx, num_graphs)`` on values' device."""
    _check_leading_dim(values, batch_idx, name=name)
    return _resolve_batch_indices(batch_idx, num_graphs, values.device)


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
    """Count nodes per graph via :func:`torch.bincount` (single kernel, no scratch)."""
    minlength = int(num_graphs) if isinstance(num_graphs, int) else num_graphs
    counts = torch.bincount(batch_idx, minlength=minlength)
    return counts.to(device=device, dtype=dtype)


def per_graph_sum(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Sum per-node values into per-graph values via ``scatter_add_``.

    Trailing dims of ``values`` are preserved in the output. See the
    module docstring for ``batch_idx`` / ``num_graphs`` semantics and
    error conditions.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph sums of shape ``(num_graphs, *values.shape[1:])``.
    """
    batch_idx, resolved = _prep_reduction(values, batch_idx, num_graphs, name="values")
    return _per_graph_sum_resolved(values, batch_idx, resolved)


def per_graph_mean(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Mean of per-node values across each graph.

    Empty graphs (zero nodes) are safe: their sum is zero and their
    count is clamped to ``1`` before the division, so they yield zero.
    See the module docstring for shared parameter / error semantics.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph means.
    """
    batch_idx, resolved = _prep_reduction(values, batch_idx, num_graphs, name="values")
    totals = _per_graph_sum_resolved(values, batch_idx, resolved)
    counts = _num_nodes_per_graph(
        batch_idx,
        resolved,
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
    See the module docstring for shared parameter / error semantics;
    additionally raises :class:`ValueError` if ``pred.shape != target.shape``.

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph MSE values.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    batch_idx, resolved = _prep_reduction(pred, batch_idx, num_graphs, name="pred")
    squared_error = (pred - target).pow(2)
    # Collapse trailing dims to one scalar per node, then scatter.
    squared_error_per_node = (
        squared_error.flatten(1).sum(dim=1) if squared_error.ndim > 1 else squared_error
    )
    squared_error_per_graph = _per_graph_sum_resolved(
        squared_error_per_node, batch_idx, resolved
    )
    trailing = math.prod(pred.shape[1:]) if pred.ndim > 1 else 1
    num_entries_per_graph = (
        _num_nodes_per_graph(
            batch_idx,
            resolved,
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
