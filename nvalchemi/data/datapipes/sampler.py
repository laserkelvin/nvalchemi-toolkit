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
"""Size-aware batch sampler for datapipes training workflows."""

from __future__ import annotations

import bisect
import math
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import torch
from torch.utils.data import Sampler


class _SizedDataset(Protocol):
    """Protocol for datasets with lightweight size metadata."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` for a sample."""
        ...

    def get_size_metadata(self) -> list[tuple[int, int]]:
        """Return ``(num_atoms, num_edges)`` for all samples."""
        ...


CapacitySchedule: TypeAlias = int | Callable[[int, int], int]


def _coerce_positive_int(value: Any, *, name: str) -> int:
    """Coerce an integer-like value and require it to be positive."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must resolve to an integer, got {value!r}") from exc
    if integer < 1:
        raise ValueError(f"{name} must be >= 1, got {integer}")
    return integer


@dataclass(frozen=True)
class LinearCapacitySchedule:
    """Integer linear schedule for atom or graph batch capacity.

    Parameters
    ----------
    start : int
        Capacity at schedule index 0.
    end : int
        Capacity at ``num_steps`` and beyond.
    num_steps : int
        Positive number of steps or epochs over which to ramp.
    per_epoch : bool, default=False
        If ``True``, use the epoch counter instead of the per-epoch batch step.

    Attributes
    ----------
    start : int
        Initial capacity.
    end : int
        Final capacity.
    num_steps : int
        Ramp length in steps or epochs.
    per_epoch : bool
        Whether the schedule uses epochs.
    """

    start: int
    end: int
    num_steps: int
    per_epoch: bool = False

    def __post_init__(self) -> None:
        """Validate schedule parameters."""
        _coerce_positive_int(self.start, name="start")
        _coerce_positive_int(self.end, name="end")
        _coerce_positive_int(self.num_steps, name="num_steps")

    @property
    def max_value(self) -> int:
        """Return the largest value this schedule can produce."""
        return max(self.start, self.end)

    def __call__(self, step: int, epoch: int) -> int:
        """Return the capacity for ``step`` and ``epoch``."""
        idx = epoch if self.per_epoch else step
        if idx <= 0:
            return self.start
        if idx >= self.num_steps:
            return self.end
        value = self.start + (self.end - self.start) * (idx / self.num_steps)
        return max(1, math.floor(value))


@dataclass(frozen=True)
class CosineCapacitySchedule:
    """Integer half-cosine schedule for atom or graph batch capacity.

    Parameters
    ----------
    start : int
        Capacity at schedule index 0.
    end : int
        Capacity at ``num_steps`` and beyond.
    num_steps : int
        Positive number of steps or epochs over which to ramp.
    per_epoch : bool, default=False
        If ``True``, use the epoch counter instead of the per-epoch batch step.

    Attributes
    ----------
    start : int
        Initial capacity.
    end : int
        Final capacity.
    num_steps : int
        Ramp length in steps or epochs.
    per_epoch : bool
        Whether the schedule uses epochs.
    """

    start: int
    end: int
    num_steps: int
    per_epoch: bool = False

    def __post_init__(self) -> None:
        """Validate schedule parameters."""
        _coerce_positive_int(self.start, name="start")
        _coerce_positive_int(self.end, name="end")
        _coerce_positive_int(self.num_steps, name="num_steps")

    @property
    def max_value(self) -> int:
        """Return the largest value this schedule can produce."""
        return max(self.start, self.end)

    def __call__(self, step: int, epoch: int) -> int:
        """Return the capacity for ``step`` and ``epoch``."""
        idx = epoch if self.per_epoch else step
        if idx <= 0:
            return self.start
        if idx >= self.num_steps:
            return self.end
        frac = idx / self.num_steps
        curve = 0.5 * (1.0 - math.cos(math.pi * frac))
        value = self.start + (self.end - self.start) * curve
        return max(1, math.floor(value))


@dataclass(frozen=True)
class PiecewiseCapacitySchedule:
    """Integer piecewise-constant schedule for atom or graph batch capacity.

    Parameters
    ----------
    boundaries : tuple[int, ...]
        Strictly increasing non-negative schedule-index boundaries.
    values : tuple[int, ...]
        Capacity values for each interval. Must have length
        ``len(boundaries) + 1``.
    per_epoch : bool, default=False
        If ``True``, use the epoch counter instead of the per-epoch batch step.

    Attributes
    ----------
    boundaries : tuple[int, ...]
        Schedule-index boundaries.
    values : tuple[int, ...]
        Interval capacities.
    per_epoch : bool
        Whether the schedule uses epochs.
    """

    boundaries: tuple[int, ...]
    values: tuple[int, ...]
    per_epoch: bool = False

    def __post_init__(self) -> None:
        """Validate schedule parameters."""
        if len(self.values) != len(self.boundaries) + 1:
            raise ValueError(
                "values must have length len(boundaries) + 1; got "
                f"len(values)={len(self.values)}, "
                f"len(boundaries)={len(self.boundaries)}"
            )
        prev = -1
        for boundary in self.boundaries:
            if isinstance(boundary, bool):
                raise TypeError("boundaries must be integers, got bool")
            boundary = operator.index(boundary)
            if boundary < 0:
                raise ValueError(
                    f"boundaries must be non-negative; got {self.boundaries}"
                )
            if boundary <= prev:
                raise ValueError(
                    f"boundaries must be strictly increasing; got {self.boundaries}"
                )
            prev = boundary
        for index, value in enumerate(self.values):
            _coerce_positive_int(value, name=f"values[{index}]")

    @property
    def max_value(self) -> int:
        """Return the largest value this schedule can produce."""
        return max(self.values)

    def __call__(self, step: int, epoch: int) -> int:
        """Return the capacity for ``step`` and ``epoch``."""
        idx = epoch if self.per_epoch else step
        value_index = bisect.bisect_right(self.boundaries, idx)
        return self.values[value_index]


class SizeAwareBatchSampler(Sampler[list[int]]):
    """Yield index batches constrained by atom and graph budgets.

    This sampler is intended for training-style datapipe iteration.  It is
    re-iterable across epochs, yields lists of dataset indices, and leaves
    collation to :class:`nvalchemi.data.datapipes.DataLoader`.

    Parameters
    ----------
    dataset : object
        Dataset with ``__len__`` and ``get_metadata(index) -> (num_atoms, num_edges)``.
    max_atoms : int | Callable[[int, int], int]
        Maximum total atoms per emitted batch.  Callable schedules receive
        ``(step, epoch)`` where ``step`` is the sampler-local batch index.
    max_batch_size : int | Callable[[int, int], int] | None, default=None
        Maximum number of graphs per emitted batch.  ``None`` disables the
        graph-count constraint.
    shuffle : bool, default=False
        Whether to shuffle the sample order each epoch before packing.
    drop_last : bool, default=False
        Whether to drop the final under-filled graph-count batch.
    seed : int, default=0
        Base random seed used when ``shuffle=True``.
    num_replicas : int | None, default=None
        Number of distributed replicas.  Defaults to 1.
    rank : int | None, default=None
        Replica rank.  Defaults to 0.

    Attributes
    ----------
    dataset : object
        Dataset being sampled.
    max_atoms : int | Callable[[int, int], int]
        Atom capacity or schedule.
    max_batch_size : int | Callable[[int, int], int] | None
        Graph capacity or schedule.
    """

    def __init__(
        self,
        dataset: _SizedDataset,
        *,
        max_atoms: CapacitySchedule,
        max_batch_size: CapacitySchedule | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Initialize the size-aware batch sampler."""
        if not hasattr(dataset, "__len__"):
            raise TypeError("dataset must implement __len__")
        if not hasattr(dataset, "get_metadata"):
            raise TypeError(
                "dataset must implement get_metadata(index) -> (num_atoms, num_edges)"
            )
        self._validate_capacity(max_atoms, name="max_atoms")
        if max_batch_size is not None:
            self._validate_capacity(max_batch_size, name="max_batch_size")

        resolved_num_replicas = (
            1
            if num_replicas is None
            else _coerce_positive_int(num_replicas, name="num_replicas")
        )
        resolved_rank = 0 if rank is None else operator.index(rank)
        if not 0 <= resolved_rank < resolved_num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas), got rank={resolved_rank}, "
                f"num_replicas={resolved_num_replicas}"
            )

        self.dataset = dataset
        self.max_atoms = max_atoms
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = operator.index(seed)
        self.num_replicas = resolved_num_replicas
        self.rank = resolved_rank
        self.epoch = 0
        self._len_cache: dict[int, int] = {}

        if hasattr(dataset, "get_size_metadata"):
            self._sample_meta = list(dataset.get_size_metadata())
        else:
            self._sample_meta = [
                dataset.get_metadata(index) for index in range(len(dataset))
            ]
        if len(self._sample_meta) != len(dataset):
            raise RuntimeError(
                "size metadata length must match dataset length; got "
                f"{len(self._sample_meta)} metadata rows for {len(dataset)} samples"
            )
        self._validate_sample_sizes()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by schedules and deterministic shuffling.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield size-aware batches for the current epoch."""
        yield from self._pack_batches(self._epoch_indices())

    def __len__(self) -> int:
        """Return the number of batches for static capacities.

        Raises
        ------
        TypeError
            If either capacity is callable, because the number of emitted
            batches may depend on the per-iteration schedule.
        """
        if self._has_dynamic_capacity:
            raise TypeError(
                "SizeAwareBatchSampler length is unavailable when max_atoms or "
                "max_batch_size is scheduled per iteration."
            )
        if self.epoch not in self._len_cache:
            self._len_cache[self.epoch] = sum(
                1 for _ in self._pack_batches(self._epoch_indices())
            )
        return self._len_cache[self.epoch]

    @property
    def _has_dynamic_capacity(self) -> bool:
        """Return whether either capacity is provided by a callable schedule."""
        return callable(self.max_atoms) or callable(self.max_batch_size)

    @staticmethod
    def _validate_capacity(capacity: CapacitySchedule, *, name: str) -> None:
        """Validate a static capacity or callable schedule."""
        if callable(capacity):
            return
        try:
            _coerce_positive_int(capacity, name=name)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer or callable schedule") from exc

    def _capacity_upper_bound(self, capacity: CapacitySchedule | None) -> int | None:
        """Return a known capacity upper bound if one is available."""
        if capacity is None:
            return None
        if not callable(capacity):
            return _coerce_positive_int(capacity, name="capacity")
        if hasattr(capacity, "max_value"):
            return _coerce_positive_int(capacity.max_value, name="capacity.max_value")
        return None

    def _validate_sample_sizes(self) -> None:
        """Validate samples against known static or scheduled capacity bounds."""
        max_atoms_bound = self._capacity_upper_bound(self.max_atoms)
        if max_atoms_bound is None:
            return
        for index, (num_atoms, _num_edges) in enumerate(self._sample_meta):
            if num_atoms > max_atoms_bound:
                raise RuntimeError(
                    f"Sample {index} has {num_atoms} atoms, exceeding the maximum "
                    f"scheduled atom budget of {max_atoms_bound}."
                )

    def _resolve_capacity(
        self,
        capacity: CapacitySchedule,
        *,
        step: int,
        name: str,
    ) -> int:
        """Resolve a static or scheduled capacity for the current batch."""
        if callable(capacity):
            value = capacity(step, self.epoch)
        else:
            value = capacity
        return _coerce_positive_int(value, name=name)

    def _resolve_graph_budget(self, *, step: int, remaining_count: int) -> int:
        """Resolve the graph-count budget for the current batch."""
        if self.max_batch_size is None:
            return max(remaining_count, 1)
        return self._resolve_capacity(
            self.max_batch_size, step=step, name="max_batch_size"
        )

    def _epoch_indices(self) -> list[int]:
        """Return this rank's sample indices for the current epoch."""
        if self.shuffle and len(self.dataset) > 0:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return torch.randperm(len(self.dataset), generator=generator).tolist()
        return list(range(len(self.dataset)))

    def _next_batch(
        self,
        indices: list[int],
        cursor: int,
        *,
        step: int,
    ) -> tuple[list[int], int, int]:
        """Return the next order-preserving batch and updated cursor."""
        atom_budget = self._resolve_capacity(
            self.max_atoms, step=step, name="max_atoms"
        )
        graph_budget = self._resolve_graph_budget(
            step=step, remaining_count=len(indices) - cursor
        )
        batch: list[int] = []
        total_atoms = 0

        while cursor < len(indices) and len(batch) < graph_budget:
            index = indices[cursor]
            num_atoms, _num_edges = self._sample_meta[index]
            if num_atoms > atom_budget:
                if batch:
                    break
                raise RuntimeError(
                    f"No remaining sample fits max_atoms={atom_budget} at "
                    f"step={step}, epoch={self.epoch}; next sample has "
                    f"{num_atoms} atoms."
                )
            if total_atoms + num_atoms > atom_budget:
                break
            batch.append(index)
            total_atoms += num_atoms
            cursor += 1

        return batch, cursor, graph_budget

    def _should_drop_final_batch(
        self,
        batch: list[int],
        *,
        cursor: int,
        graph_budget: int,
        total_count: int,
    ) -> bool:
        """Return whether a final under-filled graph-count batch should be dropped."""
        return (
            self.drop_last
            and self.max_batch_size is not None
            and cursor >= total_count
            and len(batch) < graph_budget
        )

    def _pack_batches(self, indices: list[int]) -> Iterator[list[int]]:
        """Pack indices into batches under the current capacity schedules."""
        cursor = 0
        step = 0

        if self.num_replicas == 1:
            while cursor < len(indices):
                batch, cursor, graph_budget = self._next_batch(
                    indices, cursor, step=step
                )
                if not batch:
                    break
                if self._should_drop_final_batch(
                    batch,
                    cursor=cursor,
                    graph_budget=graph_budget,
                    total_count=len(indices),
                ):
                    break
                yield batch
                step += 1
            return

        while cursor < len(indices):
            group: list[list[int]] = []
            graph_budget = 0
            for _ in range(self.num_replicas):
                if cursor >= len(indices):
                    break
                batch, cursor, graph_budget = self._next_batch(
                    indices, cursor, step=step
                )
                if not batch:
                    break
                group.append(batch)

            if not group:
                break
            if self.drop_last and (
                len(group) < self.num_replicas
                or any(len(batch) < graph_budget for batch in group)
            ):
                break
            while len(group) < self.num_replicas:
                group.append(group[-1])
            yield group[self.rank]
            step += 1
