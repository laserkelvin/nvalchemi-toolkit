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
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.training import (
    BaseLossFunction,
    ComposedLossFunction,
    ConstantWeight,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    StressLoss,
    create_model_spec,
    create_model_spec_from_json,
)
from nvalchemi.training.losses import (
    frobenius_mse,
    per_graph_mean,
    per_graph_mse,
    per_graph_sum,
)


class _ToyLoss(BaseLossFunction):
    """Concrete subclass returning a constant tensor — used for __call__ tests."""

    value: float = 1.0

    def compute(
        self, batch: Any, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:  # noqa: ARG002
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
        # When ``num_graphs`` is supplied, reductions trust it without
        # scanning ``batch_idx`` (to avoid GPU syncs in the training hot
        # path). An overflowing index is caught downstream by
        # ``scatter_add_`` itself, which raises ``RuntimeError``.
        with pytest.raises(RuntimeError, match="out of bounds"):
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
        out = loss(SimpleNamespace(), step=0, epoch=0)
        assert torch.allclose(out, torch.tensor(10.0))

    def test_compute_and_weight_with_linear_schedule(self) -> None:
        loss = _ToyLoss(
            value=1.0, weight=LinearWeight(start=0.0, end=1.0, num_steps=10)
        )
        batch = SimpleNamespace()
        assert torch.allclose(loss(batch, step=0), torch.tensor(0.0))
        assert torch.allclose(loss(batch, step=10), torch.tensor(1.0))
        assert torch.allclose(loss(batch, step=5), torch.tensor(0.5))

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
        batch = SimpleNamespace()
        assert torch.allclose(
            loss(batch, step=10, epoch=0),
            torch.tensor(0.0),
        )
        assert torch.allclose(
            loss(batch, step=0, epoch=10),
            torch.tensor(1.0),
        )

    def test_epoch_none_treated_as_zero(self) -> None:
        loss = _ToyLoss(value=2.0, weight=ConstantWeight(value=1.0))
        out = loss(SimpleNamespace(), step=3, epoch=None)
        assert torch.allclose(out, torch.tensor(2.0))

    def test_baseloss_frozen(self) -> None:
        loss = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        with pytest.raises(ValidationError):
            loss.value = 2.0  # type: ignore[misc]

    def test_baseloss_default_weight_is_constant_one(self) -> None:
        loss = _ToyLoss(value=3.0)
        assert isinstance(loss.weight, ConstantWeight)
        assert loss.weight.value == 1.0
        assert torch.allclose(loss(SimpleNamespace(), step=0), torch.tensor(3.0))

    def test_baseloss_basespec_roundtrip_without_weight(self) -> None:
        # ``BaseLossFunction.weight`` is typed as the
        # ``LossWeightSchedule`` protocol (not a Pydantic discriminated
        # union), so JSON round-trip through ``create_model_spec`` does
        # NOT rehydrate the schedule automatically. Upstream
        # ``TrainingStrategy`` reconstructs the schedule manually from
        # its ``(instance, spec)`` pair when rebuilding a loss. This
        # test locks in the body of the loss round-tripping cleanly and
        # leaves schedule rehydration to that feature.
        spec = create_model_spec(_ToyLoss, value=7.0)
        dumped = spec.model_dump_json()
        rebuilt_spec = create_model_spec_from_json(json.loads(dumped))
        built = rebuilt_spec.build()
        assert isinstance(built, _ToyLoss)
        assert built.value == 7.0
        # Default weight is identity, so __call__ returns compute() as-is.
        assert isinstance(built.weight, ConstantWeight)
        assert built.weight.value == 1.0
        out = built(SimpleNamespace(), step=2, epoch=0)
        assert torch.allclose(out, torch.tensor(7.0))

    def test_per_epoch_schedule_with_none_epoch_raises(self) -> None:
        loss = _ToyLoss(
            value=1.0,
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10, per_epoch=True),
        )
        with pytest.raises(ValueError, match="per_epoch=True"):
            loss(SimpleNamespace(), step=3, epoch=None)

    def test_non_numeric_schedule_return_raises_actionable_typeerror(self) -> None:
        class _BadReturnSchedule:
            per_epoch: bool = False

            def __call__(self, step: int, epoch: int) -> str:  # noqa: ARG002
                return "oops"

        loss = _ToyLoss(value=1.0, weight=_BadReturnSchedule())
        with pytest.raises(
            TypeError,
            match=r"_BadReturnSchedule returned str; "
            r"LossWeightSchedule\.__call__ must return float",
        ):
            loss(SimpleNamespace(), step=0, epoch=0)


class _PositionsLoss(BaseLossFunction):
    """Toy loss whose compute() sums ``batch.positions`` (gradient-bearing)."""

    scale: float = 1.0

    def compute(
        self, batch: Any, *, step: int = 0, epoch: int | None = None
    ) -> torch.Tensor:  # noqa: ARG002
        return self.scale * batch.positions.sum()


class TestComposedLossFunction:
    """Tests for composition arithmetic and the resulting loss object."""

    def setup_method(self) -> None:
        self.loss_a = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.loss_b = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.loss_c = _ToyLoss(value=1.0, weight=ConstantWeight(value=1.0))
        self.batch = SimpleNamespace()

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
        out = composed(self.batch, step=0, epoch=0)
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
        batch = SimpleNamespace(positions=positions)
        loss_a = _PositionsLoss(scale=2.0, weight=ConstantWeight(value=1.0))
        loss_b = _PositionsLoss(scale=3.0, weight=ConstantWeight(value=1.0))
        composed = loss_a + loss_b
        out = composed(batch, step=0, epoch=0)
        out.backward()
        # d/dx sum(x) = 1 per element; composed multiplier = 2 + 3 = 5.
        expected_grad = torch.full_like(positions, 5.0)
        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, atol=1e-6)

    def test_composed_basespec_roundtrip(self) -> None:
        # ``BaseLossFunction.weight`` is typed as the ``LossWeightSchedule``
        # protocol, so neither the outer schedule on ``ComposedLossFunction``
        # nor each component's own ``weight`` round-trips automatically.
        # Upstream ``TrainingStrategy`` rebuilds schedules manually from
        # their ``(instance, spec)`` pairs. This test locks in the body
        # (``components``, ``weights``) round-tripping cleanly, then
        # reattaches the outer schedule after ``.build()`` and checks
        # end-to-end evaluation.
        spec_a = create_model_spec(_ToyLoss, value=1.0)
        spec_b = create_model_spec(_ToyLoss, value=2.0)
        spec = create_model_spec(
            ComposedLossFunction,
            components=[spec_a, spec_b],
            weights=[1.0, 2.0],
        )
        dumped = spec.model_dump_json()
        rebuilt = create_model_spec_from_json(json.loads(dumped)).build()
        assert isinstance(rebuilt, ComposedLossFunction)
        assert rebuilt.weights == (1.0, 2.0)
        assert len(rebuilt.components) == 2
        assert all(isinstance(c, _ToyLoss) for c in rebuilt.components)
        assert [c.value for c in rebuilt.components] == [1.0, 2.0]
        # Outer schedule is identity after round-trip; re-scale manually
        # to emulate what ``TrainingStrategy`` will do.
        scaled = 0.5 * rebuilt
        out = scaled(SimpleNamespace(), step=0, epoch=0)
        # 0.5 * (1*1.0 + 2*2.0) = 2.5
        assert torch.allclose(out, torch.tensor(2.5), atol=1e-6)

    def test_outer_schedule_applied_once(self) -> None:
        # Regression guard: ComposedLossFunction.compute() must NOT
        # multiply by self.weight — the inherited __call__ does.
        composed = ComposedLossFunction(
            components=(self.loss_a, self.loss_b),
            weights=(1.0, 1.0),
            weight=ConstantWeight(value=0.5),
        )
        out = composed(self.batch, step=0, epoch=0)
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


class TestConcreteLosses:
    """Tests for :class:`EnergyLoss`, :class:`ForceLoss`, :class:`StressLoss`."""

    def setup_method(self) -> None:
        # Mixed-size batch: 3 graphs with 3, 5, 2 atoms respectively.
        self.nodes_per_graph = [3, 5, 2]
        self.num_graphs = 3
        self.num_nodes = sum(self.nodes_per_graph)
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
        self.num_nodes_per_graph = torch.tensor(self.nodes_per_graph, dtype=torch.long)

    def _batch(self, **extra: torch.Tensor) -> SimpleNamespace:
        """Build a ``SimpleNamespace`` batch carrying ``extra`` attributes."""
        return SimpleNamespace(
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
            num_nodes_per_graph=self.num_nodes_per_graph,
            **extra,
        )

    def test_energy_loss_gradient_matches_analytic(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_graphs, 1)
        pred = (target + torch.randn_like(target) * 0.1).detach().requires_grad_()
        batch = self._batch(energy=target, predicted_energy=pred)
        EnergyLoss()(batch).backward()
        # MSE over (B, 1): d/d pred = 2*(pred - target) / B.
        expected_grad = 2.0 * (pred.detach() - target) / self.num_graphs
        assert pred.grad is not None
        assert torch.allclose(pred.grad, expected_grad, atol=1e-6)

    def test_energy_loss_per_atom_divides_both(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])  # per-graph energies
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        batch = self._batch(energy=target, predicted_energy=pred)
        got = EnergyLoss(per_atom=True)(batch)
        # Per-atom pred: [2, 3, 4]; target: [1, 2, 2]; diffs: [1, 1, 2].
        # Mean of squared diffs over B=3: (1 + 1 + 4) / 3 = 2.0.
        assert torch.allclose(got, torch.tensor(2.0), atol=1e-6)

    def test_energy_loss_per_atom_accepts_cpu_counts_on_cuda(
        self, gpu_device: str
    ) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]], device=gpu_device)
        pred = torch.tensor([[6.0], [15.0], [8.0]], device=gpu_device)
        batch = self._batch(energy=target, predicted_energy=pred)

        got = EnergyLoss(per_atom=True)(batch)

        assert got.device.type == "cuda"
        assert torch.allclose(got, torch.tensor(2.0, device=gpu_device), atol=1e-6)

    def test_force_loss_matches_hand_computed(self) -> None:
        # 2 graphs with 3 and 2 atoms for a small hand-traceable case.
        batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        num_nodes_per_graph = torch.tensor([3, 2], dtype=torch.long)
        target = torch.zeros(5, 3)
        pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # graph 0 atom 0: |f|^2 = 1
                [0.0, 2.0, 0.0],  # graph 0 atom 1: |f|^2 = 4
                [0.0, 0.0, 3.0],  # graph 0 atom 2: |f|^2 = 9
                [1.0, 1.0, 1.0],  # graph 1 atom 0: |f|^2 = 3
                [2.0, 0.0, 0.0],  # graph 1 atom 1: |f|^2 = 4
            ]
        )
        batch = SimpleNamespace(
            batch_idx=batch_idx,
            num_graphs=2,
            num_nodes_per_graph=num_nodes_per_graph,
            forces=target,
            predicted_forces=pred,
        )

        # normalize_by_atom_count=True: per-graph mean of |f|^2 then mean
        # over graphs, then / 3 for per-component.
        # graph 0 mean |f|^2 = (1+4+9)/3 = 14/3
        # graph 1 mean |f|^2 = (3+4)/2 = 7/2
        # mean over graphs = (14/3 + 7/2) / 2 = (28/6 + 21/6) / 2 = 49/12
        # divided by 3 components = 49/36
        got_norm = ForceLoss(normalize_by_atom_count=True)(batch)
        assert torch.allclose(got_norm, torch.tensor(49.0 / 36.0), atol=1e-6)

        # normalize=False: elementwise mean over the (V, 3) tensor.
        # sum of squares = 1+4+9+3+4 = 21 across 5*3 = 15 entries -> 21/15 = 1.4.
        got_global = ForceLoss(normalize_by_atom_count=False)(batch)
        assert torch.allclose(got_global, torch.tensor(21.0 / 15.0), atol=1e-6)

    def test_force_loss_gradient_flows(self) -> None:
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        target = torch.randn(self.num_nodes, 3)
        batch = self._batch(forces=target, predicted_forces=pred)
        ForceLoss()(batch).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_stress_loss_matches_elementwise_mse(self, fixed_torch_seed: None) -> None:
        pred = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        target = torch.randn(self.num_graphs, 3, 3)
        batch = self._batch(stress=target, predicted_stress=pred)
        got = StressLoss()(batch)
        # Frobenius MSE averaged over graphs == elementwise MSE.
        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(got, expected, atol=1e-6)
        got.backward()
        assert pred.grad is not None

    # Missing-attribute errors: parametrize across losses.
    @pytest.mark.parametrize(
        ("loss_factory", "batch_kwargs", "match"),
        [
            pytest.param(
                lambda: EnergyLoss(),
                {"energy": torch.zeros(3, 1)},  # predicted_energy omitted
                r"EnergyLoss expected batch\.predicted_energy.*missing",
                id="energy_missing_prediction",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {"predicted_forces": torch.zeros(10, 3)},  # forces omitted
                r"ForceLoss expected batch\.forces.*missing",
                id="force_missing_target",
            ),
            pytest.param(
                lambda: StressLoss(),
                {"stress": torch.zeros(3, 3, 3)},  # predicted_stress omitted
                r"StressLoss expected batch\.predicted_stress.*missing",
                id="stress_missing_prediction",
            ),
        ],
    )
    def test_missing_attribute_raises_actionable_error(
        self,
        loss_factory: Any,
        batch_kwargs: dict[str, torch.Tensor],
        match: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**batch_kwargs)
        with pytest.raises(ValueError, match=match):
            loss(batch)

    def test_attribute_present_but_none_raises_distinct_message(self) -> None:
        # Explicit None is distinct from missing — error mentions that.
        batch = self._batch(energy=torch.zeros(3, 1), predicted_energy=None)
        with pytest.raises(
            ValueError,
            match=r"exists on batch and is None",
        ):
            EnergyLoss()(batch)

    def test_composed_losses_backprop_to_all_inputs(self) -> None:
        pred_energy = torch.randn(self.num_graphs, 1, requires_grad=True)
        pred_forces = torch.randn(self.num_nodes, 3, requires_grad=True)
        pred_stress = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        batch = self._batch(
            energy=torch.randn(self.num_graphs, 1),
            forces=torch.randn(self.num_nodes, 3),
            stress=torch.randn(self.num_graphs, 3, 3),
            predicted_energy=pred_energy,
            predicted_forces=pred_forces,
            predicted_stress=pred_stress,
        )
        composed = 1.0 * EnergyLoss() + 10.0 * ForceLoss() + 0.1 * StressLoss()
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3
        composed(batch).backward()
        # Each branch must contribute a non-zero gradient to its input.
        for grad in (pred_energy.grad, pred_forces.grad, pred_stress.grad):
            assert grad is not None
            assert not torch.all(grad == 0)

    def test_energy_loss_basespec_roundtrip(self) -> None:
        # Only body fields round-trip; schedules are rebuilt elsewhere.
        spec = create_model_spec(EnergyLoss, per_atom=True)
        dumped = spec.model_dump_json()
        rebuilt = create_model_spec_from_json(json.loads(dumped)).build()
        assert isinstance(rebuilt, EnergyLoss)
        assert rebuilt.per_atom is True
        assert rebuilt.target_key == "energy"
        assert rebuilt.prediction_key == "predicted_energy"

        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        batch = self._batch(energy=target, predicted_energy=pred)
        assert torch.allclose(EnergyLoss(per_atom=True)(batch), rebuilt(batch))

    def test_force_loss_reads_from_configured_prediction_key(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        renamed_pred = torch.ones(self.num_nodes, 3)
        batch = self._batch(forces=target, my_model_forces=renamed_pred)
        got = ForceLoss(prediction_key="my_model_forces")(batch)
        # |pred - target|^2 sum over 3 components = 3 per atom.
        # per-graph mean = 3; mean over graphs = 3; / 3 = 1.0.
        assert torch.allclose(got, torch.tensor(1.0), atol=1e-6)
