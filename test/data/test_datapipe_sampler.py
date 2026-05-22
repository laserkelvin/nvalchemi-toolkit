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
"""Tests for datapipes size-aware batch sampling."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import SequentialSampler

from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes import (
    DataLoader,
    Dataset,
    LinearCapacitySchedule,
    PiecewiseCapacitySchedule,
    SizeAwareBatchSampler,
)
from nvalchemi.data.datapipes.backends.base import Reader


class SizeOnlyDataset:
    """Dataset stub exposing only length and size metadata.

    Attributes
    ----------
    samples : list[tuple[int, int]]
        ``(num_atoms, num_edges)`` metadata for each sample.
    """

    def __init__(self, samples: list[tuple[int, int]]) -> None:
        """Initialize the metadata-only dataset."""
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def get_metadata(self, index: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` for ``index``."""
        return self.samples[index]


class AtomicListReader(Reader):
    """Reader that builds minimal atomic samples from atom counts.

    Attributes
    ----------
    atom_counts : list[int]
        Number of atoms in each sample.
    """

    def __init__(self, atom_counts: list[int]) -> None:
        """Initialize the reader from per-sample atom counts."""
        super().__init__()
        self.atom_counts = atom_counts

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sample as an AtomicData-compatible tensor dict."""
        num_atoms = self.atom_counts[index]
        return {
            "atomic_numbers": torch.ones(num_atoms, dtype=torch.long),
            "positions": torch.arange(num_atoms * 3, dtype=torch.float32).reshape(
                num_atoms, 3
            ),
            "atomic_masses": torch.ones(num_atoms, dtype=torch.float32),
            "forces": torch.zeros(num_atoms, 3, dtype=torch.float32),
            "energy": torch.zeros(1, 1, dtype=torch.float32),
        }

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.atom_counts)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return simple per-sample metadata."""
        return {"index": index}


class SizeMetadataReader(AtomicListReader):
    """Reader with explicit lightweight size metadata hooks.

    Attributes
    ----------
    load_calls : int
        Number of full sample loads.
    bulk_size_calls : int
        Number of bulk size metadata calls.
    """

    def __init__(self, atom_counts: list[int]) -> None:
        """Initialize the reader and counters."""
        super().__init__(atom_counts)
        self.load_calls = 0
        self.bulk_size_calls = 0

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Count full sample loads."""
        self.load_calls += 1
        return super()._load_sample(index)

    def _get_all_sample_sizes(self) -> list[tuple[int, int]]:
        """Return sizes without loading full samples."""
        self.bulk_size_calls += 1
        return [(num_atoms, 0) for num_atoms in self.atom_counts]


def _batch_atom_counts(
    batches: list[list[int]], samples: list[tuple[int, int]]
) -> list[int]:
    """Return total atom counts for sampled batches."""
    return [sum(samples[index][0] for index in batch) for batch in batches]


class TestSizeAwareBatchSampler:
    """Tests for size-aware batch planning."""

    def test_static_budgets_are_respected(self) -> None:
        """Every emitted batch respects atom and graph limits."""
        samples = [(2, 0), (3, 0), (4, 0), (1, 0)]
        sampler = SizeAwareBatchSampler(
            SizeOnlyDataset(samples), max_atoms=5, max_batch_size=2
        )

        batches = list(sampler)

        assert batches == [[0, 1], [2, 3]]
        assert _batch_atom_counts(batches, samples) == [5, 5]
        assert all(len(batch) <= 2 for batch in batches)
        assert len(sampler) == 2

    def test_sampler_uses_bulk_size_metadata(self) -> None:
        """Sampler construction prefers bulk size metadata over sample loads."""
        reader = SizeMetadataReader([2, 3, 4])
        dataset = Dataset(reader, device="cpu")

        SizeAwareBatchSampler(dataset, max_atoms=5, max_batch_size=2)

        assert reader.bulk_size_calls == 1
        assert reader.load_calls == 0
        dataset.close()

    def test_dynamic_atom_schedule_changes_batch_sizes(self) -> None:
        """Scheduled atom budgets are resolved once per emitted batch."""
        samples = [(2, 0), (2, 0), (2, 0), (2, 0)]
        sampler = SizeAwareBatchSampler(
            SizeOnlyDataset(samples),
            max_atoms=lambda step, epoch: 2 * (step + 1 + epoch),
            max_batch_size=10,
        )

        assert list(sampler) == [[0], [1, 2], [3]]
        with pytest.raises(TypeError, match="length is unavailable"):
            len(sampler)

    def test_dynamic_graph_schedule_changes_batch_sizes(self) -> None:
        """Scheduled graph budgets are resolved independently of atom budgets."""
        samples = [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0)]
        sampler = SizeAwareBatchSampler(
            SizeOnlyDataset(samples),
            max_atoms=10,
            max_batch_size=lambda step, epoch: step + epoch + 1,
        )

        assert list(sampler) == [[0], [1, 2], [3, 4]]
        with pytest.raises(TypeError, match="length is unavailable"):
            len(sampler)

    def test_atom_and_graph_schedules_compose(self) -> None:
        """The first binding scheduled limit closes the current batch."""
        samples = [(3, 0), (3, 0), (3, 0), (3, 0)]
        sampler = SizeAwareBatchSampler(
            SizeOnlyDataset(samples),
            max_atoms=PiecewiseCapacitySchedule(boundaries=(1,), values=(6, 9)),
            max_batch_size=PiecewiseCapacitySchedule(boundaries=(1,), values=(1, 3)),
        )

        assert list(sampler) == [[0], [1, 2, 3]]

    def test_oversized_sample_raises_for_known_capacity_bound(self) -> None:
        """Samples larger than the known maximum atom budget fail at construction."""
        dataset = SizeOnlyDataset([(6, 0)])

        with pytest.raises(RuntimeError, match="exceeding the maximum scheduled"):
            SizeAwareBatchSampler(dataset, max_atoms=LinearCapacitySchedule(2, 5, 3))

    def test_no_sample_fits_current_dynamic_budget_raises(self) -> None:
        """Unknown schedules fail clearly if the current budget cannot fit any sample."""
        sampler = SizeAwareBatchSampler(
            SizeOnlyDataset([(4, 0)]),
            max_atoms=lambda step, epoch: 3,
            max_batch_size=1,
        )

        with pytest.raises(RuntimeError, match="No remaining sample fits"):
            list(sampler)

    def test_shuffle_is_seeded_and_epoch_dependent(self) -> None:
        """Seeded shuffling is reproducible and changes with epoch."""
        samples = [(1, 0)] * 8
        dataset = SizeOnlyDataset(samples)
        sampler_a = SizeAwareBatchSampler(
            dataset, max_atoms=2, max_batch_size=2, shuffle=True, seed=123
        )
        sampler_b = SizeAwareBatchSampler(
            dataset, max_atoms=2, max_batch_size=2, shuffle=True, seed=123
        )

        epoch0 = list(sampler_a)
        assert epoch0 == list(sampler_b)

        sampler_a.set_epoch(1)
        assert list(sampler_a) != epoch0

    def test_explicit_rank_partitioning(self) -> None:
        """Explicit rank and replica settings partition samples deterministically."""
        samples = [(1, 0)] * 5
        dataset = SizeOnlyDataset(samples)
        rank0 = SizeAwareBatchSampler(
            dataset,
            max_atoms=10,
            max_batch_size=2,
            num_replicas=2,
            rank=0,
            drop_last=True,
        )
        rank1 = SizeAwareBatchSampler(
            dataset,
            max_atoms=10,
            max_batch_size=2,
            num_replicas=2,
            rank=1,
            drop_last=True,
        )

        assert list(rank0) == [[0, 2]]
        assert list(rank1) == [[1, 3]]


class TestDataLoaderBatchSamplerIntegration:
    """Integration tests for DataLoader ``batch_sampler`` support."""

    def test_dataloader_yields_batches_from_size_aware_sampler(self) -> None:
        """DataLoader collates the index lists emitted by the batch sampler."""
        dataset = Dataset(AtomicListReader([2, 3, 4, 1]), device="cpu")
        sampler = SizeAwareBatchSampler(dataset, max_atoms=5, max_batch_size=2)
        loader = DataLoader(dataset, batch_sampler=sampler, use_streams=False)

        batches = list(loader)

        assert len(loader) == 2
        assert all(isinstance(batch, Batch) for batch in batches)
        assert [batch.num_graphs for batch in batches] == [2, 2]
        assert [batch.num_nodes for batch in batches] == [5, 5]
        assert [batch.num_nodes_per_graph.tolist() for batch in batches] == [
            [2, 3],
            [4, 1],
        ]
        dataset.close()

    def test_dataloader_length_propagates_dynamic_sampler_typeerror(self) -> None:
        """Dynamic batch sampler length remains unavailable through DataLoader."""
        dataset = Dataset(AtomicListReader([2, 2, 2]), device="cpu")
        sampler = SizeAwareBatchSampler(
            dataset,
            max_atoms=4,
            max_batch_size=lambda step, epoch: step + 1,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, use_streams=False)

        with pytest.raises(TypeError, match="length is unavailable"):
            len(loader)
        dataset.close()

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"batch_size": 2}, "batch_size must be left at 1"),
            ({"shuffle": True}, "shuffle=True is incompatible"),
            ({"drop_last": True}, "drop_last=True is incompatible"),
            ({"sampler": SequentialSampler([0, 1])}, "sampler is incompatible"),
        ],
    )
    def test_dataloader_rejects_batch_sampler_conflicts(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        """Loader-level batching controls cannot be mixed with batch_sampler."""
        dataset = Dataset(AtomicListReader([2, 2]), device="cpu")
        sampler = SizeAwareBatchSampler(dataset, max_atoms=4, max_batch_size=2)

        with pytest.raises(ValueError, match=message):
            DataLoader(dataset, batch_sampler=sampler, **kwargs)
        dataset.close()

    def test_training_strategy_consumes_dynamic_loader(self) -> None:
        """TrainingStrategy runs over a dynamic loader without requiring len(loader)."""
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper
        from nvalchemi.training import EnergyLoss, ForceLoss
        from nvalchemi.training.optimizers import OptimizerConfig
        from nvalchemi.training.strategy import TrainingStrategy, default_training_fn

        dataset = Dataset(AtomicListReader([2, 2, 2]), device="cpu")
        sampler = SizeAwareBatchSampler(
            dataset,
            max_atoms=4,
            max_batch_size=lambda step, epoch: step + epoch + 1,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, use_streams=False)
        strategy = TrainingStrategy(
            models=DemoModelWrapper(DemoModel(num_atom_types=4, hidden_dim=4)),
            optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.Adam),
            num_epochs=1,
            training_fn=default_training_fn,
            loss_fn=EnergyLoss() + ForceLoss(normalize_by_atom_count=True),
        )

        strategy.run(loader)

        assert strategy.step_count == 2
        dataset.close()
