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
"""Step 1 tests for :mod:`nvalchemi.training.losses`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.training import (
    BaseLossFunction,
    ConstantWeight,
    CosineWeight,
    LinearWeight,
    LossWeightSchedule,
    PiecewiseWeight,
    create_model_spec,
    create_model_spec_from_json,
)
from nvalchemi.training.losses import (
    frobenius_mse,
    per_graph_mean,
    per_graph_mse,
    per_graph_sum,
)


@dataclass
class _StubCtx:
    """Minimal HookContext-compatible stub carrying only fields losses read."""

    step_count: int = 0
    epoch: int | None = 0
    batch: Any = None


class _ToyLoss(BaseLossFunction):
    """Concrete subclass returning a constant tensor — used for __call__ tests."""

    value: float = 1.0

    def compute(self, ctx: Any) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(self.value)


class TestReductions:
    """Tests for scatter-based graph-aware reductions."""

    def setup_method(self) -> None:
        # 3 graphs with 2, 3, 1 atoms respectively.
        self.batch_idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
        self.num_graphs = 3
        self.num_nodes_per_graph = torch.tensor([2, 3, 1], dtype=torch.long)

    def test_per_graph_sum_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_sum(vals, self.batch_idx, self.num_graphs)
        assert torch.allclose(got, torch.tensor([3.0, 12.0, 6.0]))

    def test_per_graph_sum_preserves_shape(self) -> None:
        vals = torch.randn(6, 3, requires_grad=True)
        got = per_graph_sum(vals, self.batch_idx, self.num_graphs)
        assert got.shape == (3, 3)
        assert got.grad_fn is not None

    def test_per_graph_mean_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_mean(
            vals, self.batch_idx, self.num_graphs, self.num_nodes_per_graph
        )
        assert torch.allclose(got, torch.tensor([1.5, 4.0, 6.0]))

    def test_per_graph_mse_matches_manual(self) -> None:
        pred = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # squared diffs per node: [0, 1, 4, 9, 16, 25]
        # per-graph sums: [1, 29, 25]; per-graph counts: [2, 3, 1]
        got = per_graph_mse(
            pred, target, self.batch_idx, self.num_graphs, self.num_nodes_per_graph
        )
        expected = torch.tensor([0.5, 29.0 / 3.0, 25.0])
        assert torch.allclose(got, expected)

    def test_per_graph_mse_3d_matches_reference(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(6, 3)
        target = torch.randn(6, 3)
        got = per_graph_mse(
            pred, target, self.batch_idx, self.num_graphs, self.num_nodes_per_graph
        )
        ref = torch.stack(
            [
                ((pred[:2] - target[:2]) ** 2).mean(),
                ((pred[2:5] - target[2:5]) ** 2).mean(),
                ((pred[5:6] - target[5:6]) ** 2).mean(),
            ]
        )
        assert torch.allclose(got, ref, atol=1e-6)

    def test_per_graph_mse_preserves_grad(self) -> None:
        pred = torch.randn(6, 3, requires_grad=True)
        target = torch.randn(6, 3)
        got = per_graph_mse(
            pred, target, self.batch_idx, self.num_graphs, self.num_nodes_per_graph
        )
        assert got.grad_fn is not None
        got.sum().backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_per_graph_mse_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must equal target shape"):
            per_graph_mse(
                torch.zeros(6, 3),
                torch.zeros(6, 2),
                self.batch_idx,
                self.num_graphs,
                self.num_nodes_per_graph,
            )

    def test_per_graph_mse_num_nodes_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must equal num_graphs"):
            per_graph_mse(
                torch.zeros(6),
                torch.zeros(6),
                self.batch_idx,
                self.num_graphs,
                torch.tensor([2, 3], dtype=torch.long),
            )

    def test_frobenius_mse_matches_manual(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            ]
        )
        target = torch.zeros(2, 3, 3)
        # Identity * k contributes 3*k^2 nonzero entries; mean over 9.
        got = frobenius_mse(pred, target)
        expected = torch.tensor([3.0 / 9.0, 12.0 / 9.0])
        assert torch.allclose(got, expected)

    def test_frobenius_mse_preserves_grad(self) -> None:
        pred = torch.randn(2, 3, 3, requires_grad=True)
        target = torch.randn(2, 3, 3)
        got = frobenius_mse(pred, target)
        assert got.grad_fn is not None
        got.sum().backward()
        assert pred.grad is not None

    def test_per_graph_sum_bad_num_graphs(self) -> None:
        with pytest.raises(ValueError, match="num_graphs must be positive"):
            per_graph_sum(torch.zeros(3), torch.zeros(3, dtype=torch.long), 0)


class TestSchedules:
    """Tests for the 4 weight schedules + protocol + round-trip."""

    def test_protocol_runtime_check(self) -> None:
        w = ConstantWeight(value=1.0)
        assert isinstance(w, LossWeightSchedule)

    def test_constant_weight(self) -> None:
        w = ConstantWeight(value=2.5)
        assert w(0, 0) == 2.5
        assert w(100, 3) == 2.5
        assert w(100_000, 99) == 2.5

    @pytest.mark.parametrize("cls", [LinearWeight, CosineWeight])
    def test_ramp_endpoints_and_clamp(
        self, cls: type[LinearWeight | CosineWeight]
    ) -> None:
        w = cls(start=0.0, end=1.0, num_steps=10)
        assert w(0, 0) == 0.0
        assert abs(w(10, 0) - 1.0) < 1e-6
        assert w(100, 0) == 1.0  # clamped above
        assert w(-5, 0) == 0.0  # clamped below

    def test_linear_midpoint(self) -> None:
        w = LinearWeight(start=0.0, end=1.0, num_steps=10)
        assert abs(w(5, 0) - 0.5) < 1e-6

    def test_cosine_midpoint(self) -> None:
        # Half-cosine midpoint is (start+end)/2 because cos(pi/2) = 0,
        # so frac = 0.5 * (1 - 0) = 0.5. This happens to match the linear
        # midpoint numerically for the 0->1 interval.
        w = CosineWeight(start=0.0, end=1.0, num_steps=10)
        assert abs(w(5, 0) - 0.5) < 1e-6

    @pytest.mark.parametrize(
        "boundaries,values,step,expected",
        [
            ((100,), (0.1, 0.9), 0, 0.1),
            ((100,), (0.1, 0.9), 99, 0.1),
            ((100,), (0.1, 0.9), 100, 0.9),
            ((100,), (0.1, 0.9), 500, 0.9),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 5, 0.0),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 10, 0.25),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 20, 0.5),
            ((10, 20, 30), (0.0, 0.25, 0.5, 1.0), 30, 1.0),
        ],
    )
    def test_piecewise_weight(
        self,
        boundaries: tuple[int, ...],
        values: tuple[float, ...],
        step: int,
        expected: float,
    ) -> None:
        w = PiecewiseWeight(boundaries=boundaries, values=values)
        assert w(step, 0) == expected

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (LinearWeight, {"start": 0.0, "end": 1.0, "num_steps": 0}),
            (LinearWeight, {"start": 0.0, "end": 1.0, "num_steps": -3}),
            (CosineWeight, {"start": 0.0, "end": 1.0, "num_steps": 0}),
            (
                PiecewiseWeight,
                {"boundaries": (10, 20), "values": (0.1, 0.5)},
            ),
            (
                PiecewiseWeight,
                {"boundaries": (10, 5), "values": (0.1, 0.5, 0.9)},
            ),
            (PiecewiseWeight, {"boundaries": (-1,), "values": (0.1, 0.5)}),
        ],
    )
    def test_schedule_validators_reject_bad_input(
        self, cls: type, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            cls(**kwargs)

    def test_schedule_frozen(self) -> None:
        w = ConstantWeight(value=1.0)
        with pytest.raises(ValidationError):
            w.value = 2.0  # type: ignore[misc]

    def test_piecewise_hashable(self) -> None:
        # Tuple-backed fields keep frozen instances hashable.
        w = PiecewiseWeight(boundaries=(10, 20), values=(0.1, 0.5, 0.9))
        assert hash(w) == hash(w)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (ConstantWeight, {"value": 0.5}),
            (LinearWeight, {"start": 0.1, "end": 0.9, "num_steps": 100}),
            (CosineWeight, {"start": 1.0, "end": 0.0, "num_steps": 50}),
            (
                PiecewiseWeight,
                {"boundaries": (10, 20), "values": (0.1, 0.5, 0.9)},
            ),
        ],
    )
    def test_schedule_basespec_roundtrip(
        self, cls: type, kwargs: dict[str, Any]
    ) -> None:
        spec = create_model_spec(cls, **kwargs)
        dumped = spec.model_dump_json()
        rebuilt_spec = create_model_spec_from_json(json.loads(dumped))
        built = rebuilt_spec.build()
        assert isinstance(built, cls)
        for k, v in kwargs.items():
            assert getattr(built, k) == v
        assert isinstance(built(5, 0), float)


class TestBaseLossFunction:
    """Tests for the abstract :class:`BaseLossFunction`."""

    def test_baseloss_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseLossFunction()

    def test_concrete_subclass_calls_compute_and_weight(self) -> None:
        loss = _ToyLoss(value=4.0, weight=ConstantWeight(value=2.5))
        ctx = _StubCtx(step_count=0, epoch=0)
        out = loss(ctx)
        assert torch.allclose(out, torch.tensor(10.0))

    def test_compute_and_weight_with_linear_schedule(self) -> None:
        loss = _ToyLoss(
            value=1.0, weight=LinearWeight(start=0.0, end=1.0, num_steps=10)
        )
        assert torch.allclose(loss(_StubCtx(step_count=0)), torch.tensor(0.0))
        assert torch.allclose(loss(_StubCtx(step_count=10)), torch.tensor(1.0))
        assert torch.allclose(loss(_StubCtx(step_count=5)), torch.tensor(0.5))

    def test_epoch_none_treated_as_zero(self) -> None:
        loss = _ToyLoss(value=2.0, weight=ConstantWeight(value=1.0))
        out = loss(_StubCtx(step_count=3, epoch=None))
        assert torch.allclose(out, torch.tensor(2.0))

    def test_baseloss_frozen(self) -> None:
        loss = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        with pytest.raises(ValidationError):
            loss.value = 2.0  # type: ignore[misc]

    def test_baseloss_default_weight_is_constant_one(self) -> None:
        loss = _ToyLoss(value=3.0)
        assert isinstance(loss.weight, ConstantWeight)
        assert loss.weight.value == 1.0
        assert torch.allclose(loss(_StubCtx(step_count=0)), torch.tensor(3.0))

    def test_baseloss_basespec_roundtrip(self) -> None:
        spec = create_model_spec(
            _ToyLoss, value=7.0, weight=LinearWeight(start=0.0, end=1.0, num_steps=4)
        )
        dumped = spec.model_dump_json()
        rebuilt_spec = create_model_spec_from_json(json.loads(dumped))
        built = rebuilt_spec.build()
        assert isinstance(built, _ToyLoss)
        assert built.value == 7.0
        assert isinstance(built.weight, LinearWeight)
        assert built.weight.start == 0.0
        assert built.weight.end == 1.0
        assert built.weight.num_steps == 4
        out = built(_StubCtx(step_count=2, epoch=0))
        assert torch.allclose(out, torch.tensor(0.5 * 7.0))
