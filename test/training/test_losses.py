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

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.training import (
    BaseLossFunction,
    ComposedLossFunction,
    ConstantWeight,
    LinearWeight,
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
        self.batch_idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)

    def test_per_graph_sum_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_sum(vals, self.batch_idx)
        assert torch.allclose(got, torch.tensor([3.0, 12.0, 6.0]))

    def test_per_graph_sum_preserves_shape(self) -> None:
        vals = torch.randn(6, 3, requires_grad=True)
        got = per_graph_sum(vals, self.batch_idx)
        assert got.shape == (3, 3)
        assert got.grad_fn is not None

    def test_per_graph_sum_explicit_num_graphs_keeps_trailing_empty(self) -> None:
        vals = torch.tensor([1.0, 2.0])
        batch_idx = torch.tensor([0, 0], dtype=torch.int32)
        got = per_graph_sum(vals, batch_idx, num_graphs=3)
        assert torch.allclose(got, torch.tensor([3.0, 0.0, 0.0]))

    def test_per_graph_mean_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_mean(vals, self.batch_idx)
        assert torch.allclose(got, torch.tensor([1.5, 4.0, 6.0]))

    def test_per_graph_mse_matches_manual(self) -> None:
        pred = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # squared diffs per node: [0, 1, 4, 9, 16, 25]
        # per-graph sums: [1, 29, 25]; per-graph counts: [2, 3, 1]
        got = per_graph_mse(pred, target, self.batch_idx)
        expected = torch.tensor([0.5, 29.0 / 3.0, 25.0])
        assert torch.allclose(got, expected)

    def test_per_graph_mse_3d_matches_reference(self, fixed_torch_seed: None) -> None:
        pred = torch.randn(6, 3)
        target = torch.randn(6, 3)
        got = per_graph_mse(pred, target, self.batch_idx)
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
        got = per_graph_mse(pred, target, self.batch_idx)
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
            )

    def test_per_graph_mse_num_graphs_too_small(self) -> None:
        with pytest.raises(ValueError, match="but num_graphs=2"):
            per_graph_mse(
                torch.zeros(6),
                torch.zeros(6),
                self.batch_idx,
                num_graphs=2,
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
            per_graph_sum(torch.zeros(3), torch.zeros(3, dtype=torch.int32), 0)


class TestReductionsCompile:
    """Tests for reduction compatibility with ``torch.compile``."""

    @staticmethod
    def _compile_kwargs(device: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"fullgraph": True}
        if device == "cuda":
            kwargs["backend"] = "cudagraphs"
        return kwargs

    @staticmethod
    def _batch_idx(device: str) -> torch.Tensor:
        return torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32, device=device)

    def test_per_graph_sum_compiles(self, device: str) -> None:
        values = torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3)
        batch_idx = self._batch_idx(device)
        compiled = torch.compile(per_graph_sum, **self._compile_kwargs(device))

        got = compiled(values, batch_idx, 3)
        expected = per_graph_sum(values, batch_idx, num_graphs=3)

        assert torch.allclose(got, expected)

    def test_per_graph_mean_compiles(self, device: str) -> None:
        values = torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3)
        batch_idx = self._batch_idx(device)
        compiled = torch.compile(per_graph_mean, **self._compile_kwargs(device))

        got = compiled(values, batch_idx, 3)
        expected = per_graph_mean(values, batch_idx, num_graphs=3)

        assert torch.allclose(got, expected)

    def test_per_graph_mse_compiles(self, device: str) -> None:
        pred = torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3)
        target = pred.flip(0)
        batch_idx = self._batch_idx(device)
        compiled = torch.compile(per_graph_mse, **self._compile_kwargs(device))

        got = compiled(pred, target, batch_idx, 3)
        expected = per_graph_mse(pred, target, batch_idx, num_graphs=3)

        assert torch.allclose(got, expected)

    def test_frobenius_mse_compiles(self, device: str) -> None:
        pred = torch.arange(18, dtype=torch.float32, device=device).reshape(2, 3, 3)
        target = pred.flip(0)
        compiled = torch.compile(frobenius_mse, **self._compile_kwargs(device))

        got = compiled(pred, target)
        expected = frobenius_mse(pred, target)

        assert torch.allclose(got, expected)


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

    def test_compute_and_weight_with_per_epoch_schedule(self) -> None:
        loss = _ToyLoss(
            value=1.0,
            weight=LinearWeight(
                start=0.0,
                end=1.0,
                num_steps=10,
                per_epoch=True,
            ),
        )
        assert torch.allclose(
            loss(_StubCtx(step_count=10, epoch=0)),
            torch.tensor(0.0),
        )
        assert torch.allclose(
            loss(_StubCtx(step_count=0, epoch=10)),
            torch.tensor(1.0),
        )

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


class _PositionsBatch:
    """Minimal batch stub exposing just ``positions`` for gradient tests."""

    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions


class _PositionsLoss(BaseLossFunction):
    """Toy loss whose compute() sums ``ctx.batch.positions`` (gradient-bearing)."""

    scale: float = 1.0

    def compute(self, ctx: Any) -> torch.Tensor:
        return self.scale * ctx.batch.positions.sum()


class TestComposedLossFunction:
    """Tests for composition arithmetic and the resulting loss object."""

    def setup_method(self) -> None:
        self.loss_a = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.loss_b = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.loss_c = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.ctx = _StubCtx(step_count=0, epoch=0)

    def test_add_two_losses(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed, ComposedLossFunction)
        assert composed.components == (self.loss_a, self.loss_b)
        assert composed.weights == (1.0, 1.0)
        assert isinstance(composed.weight, ConstantWeight)
        assert composed.weight.value == 1.0

    @pytest.mark.parametrize(
        "op",
        [lambda loss: 2.0 * loss, lambda loss: loss * 2.0],
        ids=["left", "right"],
    )
    def test_scalar_multiply_left_and_right(self, op: Any) -> None:
        composed = op(self.loss_a)
        assert isinstance(composed, ComposedLossFunction)
        assert composed.components == (self.loss_a,)
        assert composed.weights == (2.0,)

    def test_scalar_multiply_of_composition_scales_all_weights(self) -> None:
        composed = 2.0 * (self.loss_a + 3.0 * self.loss_b)
        assert composed.weights == (2.0, 6.0)
        # Outer weight is preserved (constant-1.0 here).
        assert isinstance(composed.weight, ConstantWeight)
        assert composed.weight.value == 1.0

    @pytest.mark.parametrize(
        "build",
        [
            lambda a, b, c: (a + b) + c,
            lambda a, b, c: a + (b + c),
        ],
        ids=["left_assoc", "right_assoc"],
    )
    def test_nested_addition_flattens(self, build: Any) -> None:
        composed = build(self.loss_a, self.loss_b, self.loss_c)
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3
        assert all(not isinstance(c, ComposedLossFunction) for c in composed.components)

    def test_sum_over_list(self) -> None:
        # sum() seeds with 0 → exercises __radd__.
        composed = sum([self.loss_a, self.loss_b, self.loss_c])
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3

    def test_weighted_sum_numerically_correct(self) -> None:
        composed = 2.0 * self.loss_a + 3.0 * self.loss_b
        out = composed(self.ctx)
        assert torch.allclose(out, torch.tensor(5.0), atol=1e-6)

    def test_component_weights_length_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="weights length"):
            ComposedLossFunction(
                components=(self.loss_a, self.loss_b),
                weights=(1.0,),
                weight=ConstantWeight(value=1.0),
            )

    def test_empty_components_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            ComposedLossFunction(
                components=(),
                weights=(),
                weight=ConstantWeight(value=1.0),
            )

    def test_gradient_flows_through_all_components(self) -> None:
        positions = torch.randn(4, 3, requires_grad=True)
        ctx = _StubCtx(step_count=0, epoch=0, batch=_PositionsBatch(positions))
        loss_a = _PositionsLoss(scale=2.0, weight=ConstantWeight(value=1.0))
        loss_b = _PositionsLoss(scale=3.0, weight=ConstantWeight(value=1.0))
        composed = loss_a + loss_b
        out = composed(ctx)
        out.backward()
        # d/dx sum(x) = 1 per element; composed multiplier = 2 + 3 = 5.
        expected_grad = torch.full_like(positions, 5.0)
        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, atol=1e-6)

    def test_composed_basespec_roundtrip(self) -> None:
        spec_a = create_model_spec(
            _ToyLoss, value=1.0, weight=ConstantWeight(value=1.0)
        )
        spec_b = create_model_spec(
            _ToyLoss, value=2.0, weight=ConstantWeight(value=1.0)
        )
        spec = create_model_spec(
            ComposedLossFunction,
            components=[spec_a, spec_b],
            weights=[1.0, 2.0],
            weight=ConstantWeight(value=0.5),
        )
        dumped = spec.model_dump_json()
        rebuilt = create_model_spec_from_json(json.loads(dumped)).build()
        assert isinstance(rebuilt, ComposedLossFunction)
        assert rebuilt.weights == (1.0, 2.0)
        assert isinstance(rebuilt.weight, ConstantWeight)
        assert rebuilt.weight.value == 0.5
        assert len(rebuilt.components) == 2
        assert all(isinstance(c, _ToyLoss) for c in rebuilt.components)
        assert [c.value for c in rebuilt.components] == [1.0, 2.0]
        # Eval matches: 0.5 * (1 * 1.0 + 2 * 2.0) = 2.5
        out = rebuilt(_StubCtx(step_count=0, epoch=0))
        assert torch.allclose(out, torch.tensor(2.5), atol=1e-6)

    def test_outer_schedule_applied_once(self) -> None:
        # Regression guard: ComposedLossFunction.compute() must NOT
        # multiply by self.weight — the inherited __call__ does.
        composed = ComposedLossFunction(
            components=(self.loss_a, self.loss_b),
            weights=(1.0, 1.0),
            weight=ConstantWeight(value=0.5),
        )
        out = composed(self.ctx)
        # Correct: 0.5 * (1.0*1.0 + 1.0*1.0) = 1.0
        # Double-weight bug would yield 0.5 * 0.5 * 2.0 = 0.5.
        assert torch.allclose(out, torch.tensor(1.0), atol=1e-6)

    def test_add_preserves_non_identity_schedule(self) -> None:
        # A scheduled composition must NOT be flattened by `+`; its
        # outer schedule would be silently dropped otherwise.
        scheduled = ComposedLossFunction(
            components=(self.loss_b, self.loss_c),
            weights=(1.0, 1.0),
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10),
        )
        composed = self.loss_a + scheduled
        assert len(composed.components) == 2
        # Second term is the whole scheduled composition, nested intact.
        assert composed.components[1] is scheduled
        assert composed.weights == (1.0, 1.0)
        # Symmetric on the left: scheduled + atomic.
        composed_sym = scheduled + self.loss_a
        assert len(composed_sym.components) == 2
        assert composed_sym.components[0] is scheduled

    def test_add_flattens_identity_schedule(self) -> None:
        # Identity outer weight → flatten as before.
        identity = ComposedLossFunction(
            components=(self.loss_b, self.loss_c),
            weights=(1.0, 1.0),
            weight=ConstantWeight(value=1.0),
        )
        composed = self.loss_a + identity
        assert len(composed.components) == 3
        assert composed.components == (self.loss_a, self.loss_b, self.loss_c)
        assert composed.weights == (1.0, 1.0, 1.0)

    def test_scalar_mul_preserves_non_identity_outer_schedule(self) -> None:
        scheduled = ComposedLossFunction(
            components=(self.loss_a, self.loss_b),
            weights=(1.0, 3.0),
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10),
        )
        scaled = 2.0 * scheduled
        assert scaled.weights == (2.0, 6.0)
        assert isinstance(scaled.weight, LinearWeight)
        assert scaled.weight.start == 0.0
        assert scaled.weight.end == 1.0
        assert scaled.weight.num_steps == 10

    @pytest.mark.parametrize("op", ["add", "mul"], ids=["add", "mul"])
    def test_not_implemented_for_bad_type(self, op: str) -> None:
        if op == "add":
            with pytest.raises(TypeError):
                _ = self.loss_a + "hello"  # type: ignore[operator]
        else:
            with pytest.raises(TypeError):
                _ = self.loss_a * "hello"  # type: ignore[operator]
