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
from torch import nn

from nvalchemi.training import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    ConstantWeight,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    StressLoss,
)
from nvalchemi.training._spec import create_model_spec, create_model_spec_from_json
from nvalchemi.training.losses import (
    assert_same_shape,
    frobenius_mse,
    per_graph_mean,
    per_graph_mse,
    per_graph_sum,
)


class _ToyLoss(BaseLossFunction):
    # Concrete subclass returning a constant tensor — used in composition tests.

    def __init__(
        self,
        value: float = 1.0,
        *,
        weight: Any = None,
    ) -> None:
        super().__init__(weight=weight)
        self.value = float(value)
        self.prediction_key = "prediction"
        self.target_key = "target"

    def _forward(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(self.value)


class _PositionsLoss(BaseLossFunction):
    # Toy loss whose ``_forward`` sums ``pred`` (gradient-bearing).

    def __init__(self, scale: float = 1.0, *, weight: Any = None) -> None:
        super().__init__(weight=weight)
        self.scale = float(scale)
        self.prediction_key = "positions"
        self.target_key = "positions"

    def _forward(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:  # noqa: ARG002
        return self.scale * pred.sum()


class _ReturnSchedule:
    # Schedule whose ``__call__`` returns a configurable value.

    per_epoch: bool = False

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, step: int, epoch: int) -> Any:  # noqa: ARG002
        return self.value


def _dummy_loss_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(0.0), torch.tensor(0.0)


def _dummy_loss_mappings() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pred, target = _dummy_loss_tensors()
    return {"prediction": pred}, {"target": target}


def _full_loss_batch() -> SimpleNamespace:
    # Standard 3-graph layout covering energy + forces + stress.
    num_graphs = 3
    num_nodes = 6
    return SimpleNamespace(
        batch_idx=torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32),
        num_graphs=num_graphs,
        num_nodes_per_graph=torch.tensor([2, 3, 1], dtype=torch.long),
        energy=torch.tensor([[1.0], [2.0], [3.0]]),
        predicted_energy=torch.tensor([[1.5], [2.5], [3.5]]),
        forces=torch.zeros(num_nodes, 3),
        predicted_forces=torch.ones(num_nodes, 3),
        stress=torch.zeros(num_graphs, 3, 3),
        predicted_stress=torch.ones(num_graphs, 3, 3),
    )


def _loss_metadata(batch: SimpleNamespace) -> dict[str, Any]:
    return {
        name: getattr(batch, name)
        for name in ("batch_idx", "num_graphs", "num_nodes_per_graph")
        if hasattr(batch, name)
    }


def _tensor_mapping(batch: SimpleNamespace) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
    }


def _call_from_batch(
    loss: BaseLossFunction | ComposedLossFunction,
    batch: SimpleNamespace,
    **metadata: Any,
) -> ComposedLossOutput:
    composed = (
        loss
        if isinstance(loss, ComposedLossFunction)
        else ComposedLossFunction(components=(loss,))
    )
    tensors = _tensor_mapping(batch)
    return composed(tensors, tensors, **(_loss_metadata(batch) | metadata))


class TestReductions:
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
    @staticmethod
    def _compile_kwargs(device: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"fullgraph": True}
        if device == "cuda":
            kwargs["backend"] = "cudagraphs"
        return kwargs

    @staticmethod
    def _batch_idx(device: str) -> torch.Tensor:
        return torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32, device=device)

    @pytest.mark.parametrize(
        ("fn", "args_factory"),
        [
            pytest.param(
                per_graph_sum,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
                    batch_idx,
                    3,
                ),
                id="per_graph_sum",
            ),
            pytest.param(
                per_graph_mean,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
                    batch_idx,
                    3,
                ),
                id="per_graph_mean",
            ),
            pytest.param(
                per_graph_mse,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
                    torch.arange(18, dtype=torch.float32, device=device)
                    .reshape(6, 3)
                    .flip(0),
                    batch_idx,
                    3,
                ),
                id="per_graph_mse",
            ),
            pytest.param(
                frobenius_mse,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(
                        2, 3, 3
                    ),
                    torch.arange(18, dtype=torch.float32, device=device)
                    .reshape(2, 3, 3)
                    .flip(0),
                ),
                id="frobenius_mse",
            ),
        ],
    )
    def test_reduction_compiles(
        self,
        fn: Any,
        args_factory: Any,
        device: str,
    ) -> None:
        batch_idx = self._batch_idx(device)
        args = args_factory(device, batch_idx)
        compiled = torch.compile(fn, **self._compile_kwargs(device))

        got = compiled(*args)
        expected = fn(*args)

        assert torch.allclose(got, expected)


class TestBaseLossFunction:
    # ``forward(pred, target, ...)`` is final and returns
    # ``current_weight(step, epoch) * _forward(pred, target, ...)``.
    # Subclasses override ``_forward``; composed calls dispatch to each
    # component via its own ``forward``, so each schedule fires once.

    def test_baseloss_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseLossFunction()

    def test_baseloss_is_nn_module(self) -> None:
        loss = _ToyLoss(value=1.0)
        assert isinstance(loss, nn.Module)

    def test_baseloss_default_weight_is_none(self) -> None:
        loss = _ToyLoss(value=3.0)
        assert loss.weight is None

    def test_baseloss_forward_delegates_to_private_forward(self) -> None:
        loss = _ToyLoss(value=2.5)
        direct = loss._forward(*_dummy_loss_tensors())
        via_call = loss(*_dummy_loss_tensors())
        assert torch.allclose(direct, via_call)

    def test_none_weight_current_weight_is_one(self) -> None:
        loss = _ToyLoss(value=4.0)  # no weight
        assert loss.current_weight(step=7, epoch=3) == 1.0
        # forward returns the unweighted _forward result.
        assert torch.allclose(
            loss(*_dummy_loss_tensors(), step=7, epoch=3), torch.tensor(4.0)
        )

    def test_current_weight_with_constant_schedule(self) -> None:
        loss = _ToyLoss(value=1.0, weight=ConstantWeight(value=2.5))
        assert loss.current_weight(step=0, epoch=0) == 2.5

    def test_current_weight_per_epoch_none_epoch_raises(self) -> None:
        loss = _ToyLoss(
            value=1.0,
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10, per_epoch=True),
        )
        with pytest.raises(ValueError, match="per_epoch=True"):
            loss.current_weight(step=0, epoch=None)

    @pytest.mark.parametrize(
        ("bad_value", "match"),
        [
            pytest.param(float("nan"), r"non-finite weight nan", id="nan"),
            pytest.param(float("inf"), r"non-finite weight inf", id="inf"),
            pytest.param(float("-inf"), r"non-finite weight -inf", id="neg_inf"),
        ],
    )
    def test_current_weight_non_finite_raises(
        self, bad_value: float, match: str
    ) -> None:
        loss = _ToyLoss(value=1.0, weight=_ReturnSchedule(bad_value))
        with pytest.raises(ValueError, match=match):
            loss.current_weight(step=0, epoch=0)

    def test_current_weight_non_numeric_raises(self) -> None:
        loss = _ToyLoss(value=1.0, weight=_ReturnSchedule("oops"))
        with pytest.raises(
            TypeError,
            match=r"_ReturnSchedule returned str; "
            r"LossWeightSchedule\.__call__ must return float",
        ):
            loss.current_weight(step=0, epoch=0)

    @pytest.mark.parametrize(
        ("weight_factory", "step", "epoch", "expected"),
        [
            pytest.param(
                lambda: ConstantWeight(value=2.5),
                0,
                0,
                10.0,
                id="constant_scalar",
            ),
            pytest.param(
                lambda: LinearWeight(start=0.0, end=1.0, num_steps=10),
                0,
                None,
                0.0,
                id="linear_start",
            ),
            pytest.param(
                lambda: LinearWeight(start=0.0, end=1.0, num_steps=10),
                5,
                None,
                2.0,
                id="linear_midpoint",
            ),
            pytest.param(
                lambda: LinearWeight(start=0.0, end=1.0, num_steps=10),
                10,
                None,
                4.0,
                id="linear_end",
            ),
            pytest.param(
                lambda: LinearWeight(start=0.0, end=1.0, num_steps=4, per_epoch=True),
                99,
                2,
                2.0,
                id="per_epoch_uses_epoch",
            ),
        ],
    )
    def test_baseloss_call_applies_own_schedule(
        self,
        weight_factory: Any,
        step: int,
        epoch: int | None,
        expected: float,
    ) -> None:
        loss = _ToyLoss(value=4.0, weight=weight_factory())
        got = loss(*_dummy_loss_tensors(), step=step, epoch=epoch)
        assert torch.allclose(got, torch.tensor(expected), atol=1e-6)

    def test_baseloss_epoch_none_treated_as_zero_for_step_schedule(self) -> None:
        # per_epoch=False schedules allow epoch=None; internal
        # ``epoch or 0`` coercion means the schedule reads only ``step``.
        loss = _ToyLoss(
            value=1.0, weight=LinearWeight(start=0.0, end=10.0, num_steps=10)
        )
        got = loss(*_dummy_loss_tensors(), step=7, epoch=None)
        assert torch.allclose(got, torch.tensor(7.0), atol=1e-6)

    def test_baseloss_per_epoch_schedule_with_none_epoch_raises(self) -> None:
        loss = _ToyLoss(
            value=1.0,
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10, per_epoch=True),
        )
        with pytest.raises(ValueError, match="per_epoch=True"):
            loss(*_dummy_loss_tensors(), step=3, epoch=None)

    def test_baseloss_to_device_smoke(self) -> None:
        # Stateless loss still supports ``.to()`` via nn.Module.
        loss = EnergyLoss()
        moved = loss.to("meta")
        assert isinstance(moved, nn.Module)
        assert moved is loss  # .to() is in-place for nn.Module

    def test_baseloss_state_dict_empty(self) -> None:
        loss = EnergyLoss()
        assert len(loss.state_dict()) == 0
        assert list(loss.parameters()) == []
        assert list(loss.buffers()) == []


class TestLossRepr:
    @pytest.mark.parametrize(
        ("loss_factory", "class_name", "substrings"),
        [
            pytest.param(
                lambda: EnergyLoss(per_atom=True),
                "EnergyLoss",
                (
                    "target_key='energy'",
                    "prediction_key='predicted_energy'",
                    "per_atom=True",
                    "weight=None",
                ),
                id="energy",
            ),
            pytest.param(
                lambda: ForceLoss(normalize_by_atom_count=False),
                "ForceLoss",
                ("normalize_by_atom_count=False", "weight=None"),
                id="force",
            ),
            pytest.param(
                lambda: StressLoss(weight=ConstantWeight(value=2.0)),
                "StressLoss",
                ("target_key='stress'", "ConstantWeight"),
                id="stress_scheduled",
            ),
        ],
    )
    def test_concrete_loss_repr_contains_hyperparameters(
        self,
        loss_factory: Any,
        class_name: str,
        substrings: tuple[str, ...],
    ) -> None:
        text = repr(loss_factory())
        assert class_name in text
        for substring in substrings:
            assert substring in text, (substring, text)

    def test_composed_repr_shows_nested_components(self) -> None:
        composed = EnergyLoss() + ForceLoss()
        text = repr(composed)
        assert "ComposedLossFunction" in text
        assert "EnergyLoss" in text
        assert "ForceLoss" in text
        # nn.ModuleList numbers its children; "(0):" is the first entry.
        assert "(0)" in text

    def test_extra_repr_non_empty_on_concrete(self) -> None:
        for loss in (EnergyLoss(), ForceLoss(), StressLoss()):
            assert loss.extra_repr() != ""


class TestComposedLossFunction:
    def setup_method(self) -> None:
        self.loss_a = _ToyLoss(value=1.0)
        self.loss_b = _ToyLoss(value=1.0)
        self.loss_c = _ToyLoss(value=1.0)

    def test_add_two_losses(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed, ComposedLossFunction)
        assert tuple(composed.components) == (self.loss_a, self.loss_b)

    def test_composed_has_no_weight_attribute_of_its_own(self) -> None:
        # Regardless of what the components carry, a composition's own
        # schedule is absent. Component weights are the only weighting path.
        components = (
            _ToyLoss(value=1.0, weight=ConstantWeight(value=2.0)),
            _ToyLoss(value=1.0, weight=LinearWeight(start=0.0, end=1.0, num_steps=5)),
        )
        composed = ComposedLossFunction(components=components)
        assert not hasattr(composed, "weight")

    def test_composed_is_nn_module(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed, nn.Module)
        modules = list(composed.modules())
        assert self.loss_a in modules
        assert self.loss_b in modules

    def test_composed_components_stored_as_module_list(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed.components, nn.ModuleList)

    @pytest.mark.parametrize(
        "op",
        [lambda loss: 2.0 * loss, lambda loss: loss * 2.0, lambda loss: loss / 2.0],
        ids=["left_mul", "right_mul", "div"],
    )
    def test_scalar_arithmetic_is_not_supported(self, op: Any) -> None:
        with pytest.raises(TypeError):
            op(self.loss_a)

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
        loss_a = _ToyLoss(value=1.0, weight=ConstantWeight(value=2.0))
        loss_b = _ToyLoss(value=1.0, weight=ConstantWeight(value=3.0))
        composed = loss_a + loss_b
        out = composed(*_dummy_loss_mappings(), step=0, epoch=0)
        assert set(out) == {"total_loss", "_ToyLoss_0", "_ToyLoss_1"}
        assert torch.allclose(out["_ToyLoss_0"], torch.tensor(2.0))
        assert torch.allclose(out["_ToyLoss_1"], torch.tensor(3.0))
        assert torch.allclose(out["total_loss"], torch.tensor(5.0), atol=1e-6)

    def test_composed_is_pure_sum_of_weighted_components(self) -> None:
        # composed = a*v1 + b*v2 where each component's schedule is applied
        # exactly once inside its own forward.
        a, b, v1, v2 = 2.0, 3.0, 5.0, 7.0
        comp1 = _ToyLoss(value=v1, weight=ConstantWeight(value=a))
        comp2 = _ToyLoss(value=v2, weight=ConstantWeight(value=b))
        composed = ComposedLossFunction(components=(comp1, comp2))
        out = composed(*_dummy_loss_mappings(), step=0, epoch=0)
        expected = a * v1 + b * v2
        assert torch.allclose(out["total_loss"], torch.tensor(expected), atol=1e-6)

    def test_empty_components_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ComposedLossFunction(components=())

    def test_non_loss_component_rejected(self) -> None:
        with pytest.raises(
            TypeError,
            match="components\\[0\\] must be a BaseLossFunction or ComposedLossFunction",
        ):
            ComposedLossFunction(
                components=("not-a-loss",),  # type: ignore[arg-type]
            )

    def test_gradient_flows_through_all_components(self) -> None:
        positions = torch.randn(4, 3, requires_grad=True)
        loss_a = _PositionsLoss(scale=2.0)
        loss_b = _PositionsLoss(scale=3.0)
        composed = loss_a + loss_b
        out = composed(
            {"positions": positions},
            {"positions": torch.zeros_like(positions)},
            step=0,
            epoch=0,
        )
        out["total_loss"].backward()
        # d/dx sum(x) = 1 per element; composed multiplier = 2 + 3 = 5.
        expected_grad = torch.full_like(positions, 5.0)
        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, atol=1e-6)

    def test_component_schedule_applied_inside_composition(self) -> None:
        # Each component's schedule is applied exactly once — inside
        # its own forward.
        weighted = _ToyLoss(value=4.0, weight=ConstantWeight(value=2.5))
        composed = ComposedLossFunction(components=(weighted,))
        out = composed(*_dummy_loss_mappings(), step=0, epoch=0)
        # 2.5 (schedule) * 4.0 (_forward) = 10.0
        assert torch.allclose(out["total_loss"], torch.tensor(10.0), atol=1e-6)

    def test_linear_schedule_on_component_in_composition(self) -> None:
        scheduled = _ToyLoss(
            value=1.0,
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10),
        )
        composed = ComposedLossFunction(components=(scheduled,))
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=0)["total_loss"],
            torch.tensor(0.0),
            atol=1e-6,
        )
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=10)["total_loss"],
            torch.tensor(1.0),
            atol=1e-6,
        )
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=5)["total_loss"],
            torch.tensor(0.5),
            atol=1e-6,
        )

    def test_per_epoch_schedule_with_none_epoch_raises_in_composition(self) -> None:
        scheduled = _ToyLoss(
            value=1.0,
            weight=LinearWeight(start=0.0, end=1.0, num_steps=10, per_epoch=True),
        )
        composed = ComposedLossFunction(components=(scheduled,))
        with pytest.raises(ValueError, match="per_epoch=True"):
            composed(*_dummy_loss_mappings(), step=3, epoch=None)

    def test_nested_composition_applies_each_schedule_exactly_once(self) -> None:
        # Nesting should not cause duplicate schedule application.
        leaf = _ToyLoss(value=2.0, weight=ConstantWeight(value=3.0))
        inner = ComposedLossFunction(components=(leaf,))
        outer = ComposedLossFunction(components=(inner,))
        out = outer(*_dummy_loss_mappings(), step=0, epoch=0)
        # 3 * 2 = 6
        assert torch.allclose(out["total_loss"], torch.tensor(6.0), atol=1e-6)
        assert torch.allclose(out["_ToyLoss"], torch.tensor(6.0))

    @pytest.mark.parametrize("op", ["add", "mul"], ids=["add", "mul"])
    def test_not_implemented_for_bad_type(self, op: str) -> None:
        if op == "add":
            with pytest.raises(TypeError):
                _ = self.loss_a + "hello"  # type: ignore[operator]
        else:
            with pytest.raises(TypeError):
                _ = self.loss_a * "hello"  # type: ignore[operator]


class TestWeightFactors:
    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            pytest.param(
                lambda: _ToyLoss(),
                {"_ToyLoss": 1.0},
                id="bare_none_weight",
            ),
            pytest.param(
                lambda: _ToyLoss(weight=ConstantWeight(value=0.5)),
                {"_ToyLoss": 0.5},
                id="bare_constant_schedule",
            ),
            pytest.param(
                lambda: ComposedLossFunction(
                    components=(
                        EnergyLoss(weight=ConstantWeight(value=2.0)),
                        ForceLoss(weight=ConstantWeight(value=3.0)),
                    ),
                ),
                {"EnergyLoss": 2.0, "ForceLoss": 3.0},
                id="composed_component_weights",
            ),
            pytest.param(
                lambda: ComposedLossFunction(
                    components=(
                        EnergyLoss(weight=ConstantWeight(value=0.5)),
                        ForceLoss(weight=ConstantWeight(value=0.25)),
                    ),
                ),
                {"EnergyLoss": 0.5, "ForceLoss": 0.25},
                id="composed_component_schedules",
            ),
        ],
    )
    def test_weight_factors_simple_cases(
        self, factory: Any, expected: dict[str, float]
    ) -> None:
        assert factory().weight_factors(step=0, epoch=0) == expected

    def test_weight_factors_no_args_smoke(self) -> None:
        # Both ``current_weight`` and ``weight_factors`` take default
        # ``step=0, epoch=None`` so introspection helpers don't demand args.
        loss = _ToyLoss(weight=ConstantWeight(value=0.5))
        assert loss.current_weight() == 0.5
        assert loss.weight_factors() == {"_ToyLoss": 0.5}
        composed = ComposedLossFunction(
            components=(EnergyLoss(weight=ConstantWeight(value=2.0)),)
        )
        assert composed.weight_factors() == {"EnergyLoss": 2.0}

    def test_weight_factors_class_name_collision_gets_indexed_suffix(self) -> None:
        composed = ComposedLossFunction(
            components=(StressLoss(), StressLoss()),
        )
        got = composed.weight_factors(step=0, epoch=0)
        assert set(got) == {"StressLoss_0", "StressLoss_1"}
        assert got["StressLoss_0"] == 1.0
        assert got["StressLoss_1"] == 1.0

    def test_weight_factors_three_way_collision_across_nested_composition(self) -> None:
        # Inner composition contains two ``StressLoss`` instances; wrapping in
        # an outer composition with another ``StressLoss`` must collapse to
        # three collision-suffixed keys — NOT to a mix like
        # ``{"StressLoss_0", "StressLoss_1", "StressLoss"}`` from per-level
        # suffixing.
        inner = ComposedLossFunction(components=(StressLoss(), StressLoss()))
        outer = ComposedLossFunction(components=(inner, StressLoss()))
        got = outer.weight_factors(step=0, epoch=0)
        assert set(got) == {"StressLoss_0", "StressLoss_1", "StressLoss_2"}
        assert all(v == 1.0 for v in got.values())

    def test_weight_factors_nested_composition_flattens(self) -> None:
        inner = ComposedLossFunction(
            components=(EnergyLoss(weight=ConstantWeight(value=0.5)),),
        )
        outer = ComposedLossFunction(
            components=(inner, ForceLoss(weight=ConstantWeight(value=4.0))),
        )
        assert outer.weight_factors(step=0, epoch=0) == {
            "EnergyLoss": 0.5,
            "ForceLoss": 4.0,
        }


class TestConcreteLosses:
    def setup_method(self) -> None:
        # Mixed-size batch: 3 graphs with 3, 5, 2 atoms respectively.
        self.nodes_per_graph = [3, 5, 2]
        self.num_graphs = 3
        self.num_nodes = sum(self.nodes_per_graph)
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
        self.num_nodes_per_graph = torch.tensor(self.nodes_per_graph, dtype=torch.long)

    def _batch(self, **extra: torch.Tensor) -> SimpleNamespace:
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
        EnergyLoss()(pred, target).backward()
        # MSE over (B, 1): d/d pred = 2*(pred - target) / B.
        expected_grad = 2.0 * (pred.detach() - target) / self.num_graphs
        assert pred.grad is not None
        assert torch.allclose(pred.grad, expected_grad, atol=1e-6)

    def test_energy_loss_per_atom_divides_both(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])  # per-graph energies
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        got = EnergyLoss(per_atom=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        # Per-atom pred: [2, 3, 4]; target: [1, 2, 2]; diffs: [1, 1, 2].
        # Mean of squared diffs over B=3: (1 + 1 + 4) / 3 = 2.0.
        assert torch.allclose(got, torch.tensor(2.0), atol=1e-6)

    def test_energy_loss_per_atom_accepts_padded_node_mask(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])  # per-graph energies
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        node_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, False, False, False],
            ]
        )
        got = EnergyLoss(per_atom=True)(pred, target, num_nodes_per_graph=node_mask)
        # The padded mask has row counts [3, 5, 2], matching the dense-count test.
        assert torch.allclose(got, torch.tensor(2.0), atol=1e-6)

    def test_energy_loss_per_atom_accepts_cpu_counts_on_cuda(
        self, gpu_device: str
    ) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]], device=gpu_device)
        pred = torch.tensor([[6.0], [15.0], [8.0]], device=gpu_device)
        got = EnergyLoss(per_atom=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )

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
        # normalize_by_atom_count=True: per-graph mean of |f|^2 then mean
        # over graphs, then / 3 for per-component.
        # graph 0 mean |f|^2 = (1+4+9)/3 = 14/3
        # graph 1 mean |f|^2 = (3+4)/2 = 7/2
        # mean over graphs = (14/3 + 7/2) / 2 = (28/6 + 21/6) / 2 = 49/12
        # divided by 3 components = 49/36
        got_norm = ForceLoss(normalize_by_atom_count=True)(
            pred,
            target,
            batch_idx=batch_idx,
            num_graphs=2,
            num_nodes_per_graph=num_nodes_per_graph,
        )
        assert torch.allclose(got_norm, torch.tensor(49.0 / 36.0), atol=1e-6)

        # normalize=False: elementwise mean over the (V, 3) tensor.
        # sum of squares = 1+4+9+3+4 = 21 across 5*3 = 15 entries -> 21/15 = 1.4.
        got_global = ForceLoss(normalize_by_atom_count=False)(pred, target)
        assert torch.allclose(got_global, torch.tensor(21.0 / 15.0), atol=1e-6)

    def test_force_loss_padded_layout_matches_flat_hand_computed(self) -> None:
        target = torch.zeros(2, 3, 3)
        pred = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],  # graph 0 atom 0: |f|^2 = 1
                    [0.0, 2.0, 0.0],  # graph 0 atom 1: |f|^2 = 4
                    [0.0, 0.0, 3.0],  # graph 0 atom 2: |f|^2 = 9
                ],
                [
                    [1.0, 1.0, 1.0],  # graph 1 atom 0: |f|^2 = 3
                    [2.0, 0.0, 0.0],  # graph 1 atom 1: |f|^2 = 4
                    [99.0, 99.0, 99.0],  # padding; must be ignored
                ],
            ]
        )
        target[1, 2] = float("nan")
        num_nodes_per_graph = torch.tensor([3, 2], dtype=torch.long)

        got_norm = ForceLoss(normalize_by_atom_count=True)(
            pred, target, num_nodes_per_graph=num_nodes_per_graph
        )
        assert torch.allclose(got_norm, torch.tensor(49.0 / 36.0), atol=1e-6)

        got_global = ForceLoss(normalize_by_atom_count=False)(
            pred, target, num_nodes_per_graph=num_nodes_per_graph
        )
        assert torch.allclose(got_global, torch.tensor(21.0 / 15.0), atol=1e-6)

    def test_force_loss_padded_layout_accepts_node_mask(self) -> None:
        target = torch.zeros(2, 3, 3)
        pred = torch.ones(2, 3, 3)
        pred[1, 2] = 100.0
        target[1, 2] = float("nan")
        node_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
            ]
        )

        got = ForceLoss(normalize_by_atom_count=True)(
            pred, target, num_nodes_per_graph=node_mask
        )

        assert torch.allclose(got, torch.tensor(1.0), atol=1e-6)

    def test_force_loss_gradient_flows(self) -> None:
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        target = torch.randn(self.num_nodes, 3)
        ForceLoss()(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        ).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_stress_loss_matches_elementwise_mse(self, fixed_torch_seed: None) -> None:
        pred = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        target = torch.randn(self.num_graphs, 3, 3)
        got = StressLoss()(pred, target)
        # Frobenius MSE averaged over graphs == elementwise MSE.
        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(got, expected, atol=1e-6)
        got.backward()
        assert pred.grad is not None

    @pytest.mark.parametrize(
        ("loss_factory", "batch_kwargs", "missing_attr"),
        [
            pytest.param(
                lambda: EnergyLoss(),
                {"energy": torch.zeros(3, 1)},  # predicted_energy omitted
                "predicted_energy",
                id="energy_missing_prediction",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {"predicted_forces": torch.zeros(10, 3)},  # forces omitted
                "forces",
                id="force_missing_target",
            ),
            pytest.param(
                lambda: StressLoss(),
                {"stress": torch.zeros(3, 3, 3)},  # predicted_stress omitted
                "predicted_stress",
                id="stress_missing_prediction",
            ),
        ],
    )
    def test_missing_mapping_key_raises_key_error(
        self,
        loss_factory: Any,
        batch_kwargs: dict[str, torch.Tensor],
        missing_attr: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**batch_kwargs)
        with pytest.raises(KeyError, match=missing_attr):
            _call_from_batch(loss, batch)

    def test_mapping_key_resolving_to_none_raises_type_error(self) -> None:
        loss = ComposedLossFunction(components=(EnergyLoss(),))
        predictions = {"predicted_energy": None}  # type: ignore[dict-item]
        targets = {"energy": torch.zeros(3, 1)}
        with pytest.raises(TypeError, match="predicted_energy"):
            loss(predictions, targets)

    @pytest.mark.parametrize(
        ("loss_factory", "batch_kwargs", "loss_name"),
        [
            pytest.param(
                lambda: EnergyLoss(),
                {
                    "energy": torch.zeros(3, 2),  # incompatible trailing dim
                    "predicted_energy": torch.zeros(3, 3),
                },
                "EnergyLoss",
                id="energy_trailing_mismatch",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {
                    "forces": torch.zeros(10, 2),  # trailing 2 vs 3 not broadcastable
                    "predicted_forces": torch.zeros(10, 3),
                },
                "ForceLoss",
                id="force_component_mismatch",
            ),
            pytest.param(
                lambda: StressLoss(),
                {
                    "stress": torch.zeros(3, 2),  # trailing 2 vs 3 not broadcastable
                    "predicted_stress": torch.zeros(3, 3, 3),
                },
                "StressLoss",
                id="stress_rank_mismatch",
            ),
        ],
    )
    def test_prediction_target_shape_mismatch_raises(
        self,
        loss_factory: Any,
        batch_kwargs: dict[str, torch.Tensor],
        loss_name: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**batch_kwargs)
        with pytest.raises(
            ValueError,
            match=rf"{loss_name}: prediction and target shape mismatch",
        ):
            _call_from_batch(loss, batch)

    @pytest.mark.parametrize(
        ("loss_factory", "tensor_kwargs", "missing_attr"),
        [
            pytest.param(
                lambda: EnergyLoss(per_atom=True),
                {
                    "energy": torch.zeros(3, 1),
                    "predicted_energy": torch.zeros(3, 1),
                },
                "num_nodes_per_graph",
                id="energy_per_atom_missing_num_nodes",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {
                    "forces": torch.zeros(10, 3),
                    "predicted_forces": torch.zeros(10, 3),
                },
                "batch_idx",
                id="force_missing_batch_idx",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {
                    "forces": torch.zeros(10, 3),
                    "predicted_forces": torch.zeros(10, 3),
                },
                "num_graphs",
                id="force_missing_num_graphs",
            ),
            pytest.param(
                lambda: ForceLoss(),
                {
                    "forces": torch.zeros(3, 5, 3),
                    "predicted_forces": torch.zeros(3, 5, 3),
                },
                "num_nodes_per_graph",
                id="force_missing_num_nodes",
            ),
        ],
    )
    def test_missing_loss_metadata_raises_value_error(
        self,
        loss_factory: Any,
        tensor_kwargs: dict[str, torch.Tensor],
        missing_attr: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**tensor_kwargs)
        # _batch sets all graph metadata fields; drop the one under test
        # to exercise the tensor-first metadata requirement path.
        delattr(batch, missing_attr)
        with pytest.raises(ValueError, match=missing_attr):
            _call_from_batch(loss, batch)

    def test_composed_losses_backprop_to_all_inputs(self) -> None:
        batch = _full_loss_batch()
        for name in ("energy", "forces", "stress"):
            setattr(batch, name, torch.randn_like(getattr(batch, name)))
        for name in ("predicted_energy", "predicted_forces", "predicted_stress"):
            setattr(
                batch, name, torch.randn_like(getattr(batch, name)).requires_grad_()
            )

        composed = (
            EnergyLoss()
            + ForceLoss(weight=ConstantWeight(value=10.0))
            + StressLoss(weight=ConstantWeight(value=0.1))
        )
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3
        out = _call_from_batch(composed, batch)
        assert set(out) == {"total_loss", "EnergyLoss", "ForceLoss", "StressLoss"}
        out["total_loss"].backward()
        for grad in (
            batch.predicted_energy.grad,
            batch.predicted_forces.grad,
            batch.predicted_stress.grad,
        ):
            assert grad is not None
            assert not torch.all(grad == 0)

    def test_force_loss_reads_from_configured_prediction_key(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        renamed_pred = torch.ones(self.num_nodes, 3)
        predictions = {"my_model_forces": renamed_pred}
        targets = {"forces": target}
        loss = ComposedLossFunction(
            components=(ForceLoss(prediction_key="my_model_forces"),)
        )
        got = loss(
            predictions,
            targets,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # |pred - target|^2 sum over 3 components = 3 per atom.
        # per-graph mean = 3; mean over graphs = 3; / 3 = 1.0.
        assert torch.allclose(got["total_loss"], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(got["ForceLoss"], torch.tensor(1.0))


class TestIgnoreNaN:
    """Tests for the opt-in ``ignore_nan`` masking in concrete losses.

    Targets with ``NaN`` represent missing labels and must not contribute
    to loss value or gradient. Predictions are assumed finite. The
    implementation uses branch-free tensor ops, so behavior is the same
    as the eager path these tests assert.
    """

    def setup_method(self) -> None:
        # Reuse the 3,5,2-atom layout from ``TestConcreteLosses`` so per-atom
        # normalization and multi-graph masking paths are all exercised.
        self.nodes_per_graph = [3, 5, 2]
        self.num_graphs = 3
        self.num_nodes = sum(self.nodes_per_graph)
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
        self.num_nodes_per_graph = torch.tensor(self.nodes_per_graph, dtype=torch.long)

    def _batch(self, **extra: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
            num_nodes_per_graph=self.num_nodes_per_graph,
            **extra,
        )

    # ---- EnergyLoss ---------------------------------------------------

    def test_energy_loss_default_propagates_nan(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [2.5], [3.5]])
        got = EnergyLoss()(pred, target)
        assert torch.isnan(got)

    def test_energy_loss_ignore_nan_masks_missing_targets(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [2.5], [3.5]])
        got = EnergyLoss(ignore_nan=True)(pred, target)
        # Valid entries contribute (0.5)^2 and (0.5)^2; two valid entries.
        expected = torch.tensor((0.25 + 0.25) / 2.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_loss_ignore_nan_zero_gradient_at_nan_positions(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [10.0], [3.5]], requires_grad=True)
        EnergyLoss(ignore_nan=True)(pred, target).backward()
        assert pred.grad is not None
        # The NaN-target entry must receive exactly zero gradient.
        assert pred.grad[1].item() == 0.0
        # Other entries must receive finite, non-zero gradient.
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[0].item() != 0.0
        assert pred.grad[2].item() != 0.0

    def test_energy_loss_ignore_nan_all_nan_gives_zero(self) -> None:
        target = torch.full((self.num_graphs, 1), float("nan"))
        pred = torch.randn(self.num_graphs, 1, requires_grad=True)
        got = EnergyLoss(ignore_nan=True)(pred, target)
        assert torch.allclose(got, torch.tensor(0.0))
        got.backward()
        assert pred.grad is not None
        assert torch.all(pred.grad == 0.0)

    def test_energy_loss_ignore_nan_per_atom_applies_normalization_first(self) -> None:
        # Per-atom normalization must be applied before masking so the
        # valid-entry MSE is computed on per-atom values, not raw energies.
        target = torch.tensor([[3.0], [float("nan")], [4.0]])  # per-atom: 1, -, 2
        pred = torch.tensor([[6.0], [15.0], [8.0]])  # per-atom: 2, 3, 4
        got = EnergyLoss(per_atom=True, ignore_nan=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        # Valid per-atom diffs: (2-1)=1 and (4-2)=2; MSE over 2 entries.
        expected = torch.tensor((1.0 + 4.0) / 2.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_loss_ignore_nan_off_matches_baseline(self) -> None:
        target = torch.randn(self.num_graphs, 1)
        pred = torch.randn(self.num_graphs, 1)
        baseline = EnergyLoss()(pred, target)
        opt_in = EnergyLoss(ignore_nan=True)(pred, target)
        assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- ForceLoss ----------------------------------------------------

    def test_force_loss_default_propagates_nan(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[4, 1] = float("nan")
        pred = torch.ones(self.num_nodes, 3)
        assert torch.isnan(
            ForceLoss(normalize_by_atom_count=True)(
                pred,
                target,
                batch_idx=self.batch_idx,
                num_graphs=self.num_graphs,
            )
        )
        assert torch.isnan(ForceLoss(normalize_by_atom_count=False)(pred, target))

    def test_force_loss_ignore_nan_global_masks_missing_components(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[4, 1] = float("nan")  # one component missing
        pred = torch.ones(self.num_nodes, 3)
        got = ForceLoss(normalize_by_atom_count=False, ignore_nan=True)(pred, target)
        # V*3 - 1 = 29 valid entries, each contributing (1 - 0)^2 = 1.
        expected = torch.tensor(29.0 / 29.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nan_per_graph_all_nan_graph_zero_contribution(
        self,
    ) -> None:
        # Graph 1 (atoms 3..7) has fully-NaN force labels; graphs 0 and 2
        # are fully labeled. The all-NaN graph must contribute zero to the
        # mean over graphs.
        target = torch.zeros(self.num_nodes, 3)
        target[3:8] = float("nan")
        pred = torch.ones(self.num_nodes, 3)
        got = ForceLoss(normalize_by_atom_count=True, ignore_nan=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # Graph 0: 3 atoms * 3 components all valid, each (1-0)^2 = 1,
        # per-graph loss = 9/9 = 1. Graph 2: 2 atoms * 3 components all
        # valid, per-graph loss = 6/6 = 1. Graph 1: all NaN, loss = 0.
        # Mean over 3 graphs = (1 + 0 + 1) / 3.
        expected = torch.tensor(2.0 / 3.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nan_per_graph_partial_mask(self) -> None:
        # Single component missing on one atom; check the per-graph
        # denominator reflects 3*n_atoms - missing, not 3*n_atoms.
        target = torch.zeros(self.num_nodes, 3)
        target[0, 0] = float("nan")  # graph 0, atom 0, x component
        pred = torch.ones(self.num_nodes, 3)
        got = ForceLoss(normalize_by_atom_count=True, ignore_nan=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # Graph 0: 8 valid components (out of 9), each contributes 1; loss = 8/8 = 1.
        # Graph 1: 15/15 = 1. Graph 2: 6/6 = 1. Mean = 1.0.
        expected = torch.tensor(1.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nan_zero_gradient_at_nan_positions(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[0, 0] = float("nan")
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        ForceLoss(normalize_by_atom_count=True, ignore_nan=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        ).backward()
        assert pred.grad is not None
        assert pred.grad[0, 0].item() == 0.0
        # At least one other component receives non-zero gradient.
        assert torch.isfinite(pred.grad).all()
        assert (pred.grad != 0.0).any()

    def test_force_loss_ignore_nan_off_matches_baseline(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_nodes, 3)
        pred = torch.randn(self.num_nodes, 3)
        for norm in (True, False):
            metadata = (
                {"batch_idx": self.batch_idx, "num_graphs": self.num_graphs}
                if norm
                else {}
            )
            baseline = ForceLoss(normalize_by_atom_count=norm)(pred, target, **metadata)
            opt_in = ForceLoss(normalize_by_atom_count=norm, ignore_nan=True)(
                pred, target, **metadata
            )
            assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- StressLoss ---------------------------------------------------

    def test_stress_loss_default_propagates_nan(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[1, 2, 2] = float("nan")
        pred = torch.ones(self.num_graphs, 3, 3)
        assert torch.isnan(StressLoss()(pred, target))

    def test_stress_loss_ignore_nan_all_nan_graph_zero_contribution(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[1] = float("nan")  # full graph 1 unlabeled
        pred = torch.ones(self.num_graphs, 3, 3)
        got = StressLoss(ignore_nan=True)(pred, target)
        # Graph 0: 9 valid entries each (1-0)^2 = 1, per-graph loss = 9/9 = 1.
        # Graph 1: all NaN, loss = 0. Graph 2: loss = 1.
        # Mean = (1 + 0 + 1) / 3.
        expected = torch.tensor(2.0 / 3.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_stress_loss_ignore_nan_partial_mask(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[0, 0, 0] = float("nan")  # one entry missing in graph 0
        pred = torch.ones(self.num_graphs, 3, 3)
        got = StressLoss(ignore_nan=True)(pred, target)
        # Graph 0: 8 valid entries of (1-0)^2 = 1 -> loss = 8/8 = 1.
        # Graphs 1, 2: loss = 1. Mean = 1.0.
        expected = torch.tensor(1.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_stress_loss_ignore_nan_zero_gradient_at_nan_positions(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[0, 0, 0] = float("nan")
        pred = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        StressLoss(ignore_nan=True)(pred, target).backward()
        assert pred.grad is not None
        assert pred.grad[0, 0, 0].item() == 0.0
        assert torch.isfinite(pred.grad).all()

    def test_stress_loss_ignore_nan_off_matches_baseline(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_graphs, 3, 3)
        pred = torch.randn(self.num_graphs, 3, 3)
        baseline = StressLoss()(pred, target)
        opt_in = StressLoss(ignore_nan=True)(pred, target)
        assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- Repr ---------------------------------------------------------

    def test_ignore_nan_appears_in_extra_repr(self) -> None:
        for loss in (
            EnergyLoss(ignore_nan=True),
            ForceLoss(ignore_nan=True),
            StressLoss(ignore_nan=True),
        ):
            assert "ignore_nan=True" in repr(loss)


class TestLossModelSpec:
    """Tests for :func:`create_model_spec` round-trip on concrete losses.

    These tests verify the generic spec workflow works end-to-end for
    :class:`EnergyLoss`, :class:`ForceLoss`, and :class:`StressLoss`:
    ``create_model_spec(cls, **kwargs)`` → ``model_dump_json`` → ``json.loads``
    → :func:`create_model_spec_from_json` → ``spec.build()``. The rebuilt
    instance must preserve ``__init__`` kwargs and stay functionally
    equivalent on tensor inputs.

    Schedule kwargs (``weight=...``) are exercised via the **nested-spec**
    pattern: ``weight=create_model_spec(ConstantWeight, value=...)``. The
    generic spec builder does not round-trip a bare Pydantic schedule
    instance through JSON, since the dumped dict has no ``cls_path``
    discriminator. Nesting the schedule as its own spec is the supported
    workflow.
    """

    def _roundtrip(self, spec: Any) -> Any:
        dumped = json.loads(spec.model_dump_json())
        return create_model_spec_from_json(dumped)

    @pytest.mark.parametrize(
        ("cls", "kwargs"),
        [
            pytest.param(EnergyLoss, {}, id="energy_defaults"),
            pytest.param(
                EnergyLoss,
                {"per_atom": True, "ignore_nan": True},
                id="energy_per_atom_ignore_nan",
            ),
            pytest.param(
                EnergyLoss,
                {"target_key": "u_ref", "prediction_key": "u_hat"},
                id="energy_renamed_keys",
            ),
            pytest.param(ForceLoss, {}, id="force_defaults"),
            pytest.param(
                ForceLoss,
                {"normalize_by_atom_count": False, "ignore_nan": True},
                id="force_global_ignore_nan",
            ),
            pytest.param(StressLoss, {}, id="stress_defaults"),
            pytest.param(StressLoss, {"ignore_nan": True}, id="stress_ignore_nan"),
        ],
    )
    def test_loss_basespec_roundtrip_without_schedule(
        self, cls: type[BaseLossFunction], kwargs: dict[str, Any]
    ) -> None:
        """JSON round-trip rebuilds a loss with matching kwargs."""
        spec = create_model_spec(cls, **kwargs)
        rebuilt = self._roundtrip(spec)
        built = rebuilt.build()
        assert isinstance(built, cls)
        for k, v in kwargs.items():
            assert getattr(built, k) == v
        # Schedule-less losses carry ``weight=None``.
        assert built.weight is None

    @pytest.mark.parametrize(
        "cls",
        [EnergyLoss, ForceLoss, StressLoss],
        ids=["energy", "force", "stress"],
    )
    def test_loss_basespec_roundtrip_with_nested_schedule(
        self, cls: type[BaseLossFunction]
    ) -> None:
        """Nested ``create_model_spec`` on the schedule survives JSON round-trip."""
        schedule_spec = create_model_spec(ConstantWeight, value=2.5)
        spec = create_model_spec(cls, weight=schedule_spec)
        rebuilt = self._roundtrip(spec)
        built = rebuilt.build()
        assert isinstance(built, cls)
        assert isinstance(built.weight, ConstantWeight)
        assert built.weight.value == 2.5

    def test_loss_spec_preserves_timestamp(self) -> None:
        """Rehydrated spec keeps the original timestamp byte-for-byte."""
        spec = create_model_spec(EnergyLoss, per_atom=True)
        rebuilt = self._roundtrip(spec)
        assert rebuilt.timestamp == spec.timestamp

    def test_rebuilt_loss_is_functionally_equivalent(self) -> None:
        """A round-tripped loss produces the same value as the original."""
        pred = torch.randn(3, 1)
        target = torch.randn(3, 1)
        original = EnergyLoss(ignore_nan=True)
        spec = create_model_spec(
            EnergyLoss,
            ignore_nan=True,
            weight=create_model_spec(ConstantWeight, value=3.0),
        )
        rebuilt = self._roundtrip(spec).build()

        # Original has no weight, rebuilt has ConstantWeight(3.0) — underlying
        # ``_forward`` must agree; the ``3.0 *`` is applied by ``forward``.
        assert torch.allclose(
            original(pred, target), rebuilt._forward(pred, target), atol=1e-6
        )
        assert torch.allclose(
            rebuilt(pred, target), 3.0 * original(pred, target), atol=1e-6
        )

    def test_bare_schedule_instance_does_not_round_trip(self) -> None:
        """Document the unsupported path: bare Pydantic schedule can't be rehydrated.

        Passing a ``ConstantWeight`` instance directly (instead of a nested
        spec) dumps it as a plain ``{"value": ..., "per_epoch": ...}`` dict
        with no ``cls_path`` discriminator. The rehydration step therefore
        tries to feed the dict straight into the spec's ``weight`` field,
        which is typed as :class:`LossWeightSchedule`, and Pydantic
        rejects it. Users should wrap schedules in ``create_model_spec``
        to serialize them.
        """
        spec = create_model_spec(StressLoss, weight=ConstantWeight(value=2.5))
        dumped = json.loads(spec.model_dump_json())
        with pytest.raises(Exception, match="LossWeightSchedule|validation"):
            create_model_spec_from_json(dumped)


class TestShapeValidationOptIn:
    def test_bare_subclass_does_not_shape_check(self) -> None:
        loss = _ToyLoss(value=1.0)
        pred = torch.randn(3, 1)
        target = torch.randn(4, 5, 7)  # deliberately mismatched
        got = loss(pred, target)
        assert torch.allclose(got, torch.tensor(1.0))

    def test_energy_loss_raises_on_shape_mismatch(self) -> None:
        loss = EnergyLoss()
        pred = torch.zeros(3, 2)
        target = torch.zeros(3, 3)  # trailing 2 vs 3 not broadcastable
        with pytest.raises(
            ValueError,
            match="EnergyLoss: prediction and target shape mismatch",
        ):
            loss(pred, target)

    def test_assert_same_shape_public_helper(self) -> None:
        pred = torch.zeros(3, 2)
        target = torch.zeros(3, 2)
        assert_same_shape(
            pred,
            target,
            name="MyLoss",
            prediction_key="p",
            target_key="t",
        )
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target shape mismatch; "
            r"prediction_key='p' has shape \(3, 2\), "
            r"target_key='t' has shape \(3, 3\)",
        ):
            assert_same_shape(
                pred,
                torch.zeros(3, 3),
                name="MyLoss",
                prediction_key="p",
                target_key="t",
            )

    def test_assert_same_shape_omits_key_fragments_when_none(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target shape mismatch; "
            r"prediction has shape \(3, 2\), target has shape \(3, 3\)",
        ):
            assert_same_shape(
                torch.zeros(3, 2),
                torch.zeros(3, 3),
                name="MyLoss",
            )

    def test_assert_same_shape_accepts_broadcastable(self) -> None:
        # (B, 1) vs (B, 3) is broadcast-compatible; must not raise.
        assert_same_shape(
            torch.zeros(4, 1),
            torch.zeros(4, 3),
            name="MyLoss",
            prediction_key="p",
            target_key="t",
        )

    def test_assert_same_shape_rejects_dtype_mismatch(self) -> None:
        pred = torch.zeros(3, 2, dtype=torch.float32)
        target = torch.zeros(3, 2, dtype=torch.float64)
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target dtype mismatch; "
            r"prediction_key='p' has dtype torch\.float32, "
            r"target_key='t' has dtype torch\.float64",
        ):
            assert_same_shape(
                pred,
                target,
                name="MyLoss",
                prediction_key="p",
                target_key="t",
            )

    def test_assert_same_shape_dtype_check_runs_before_shape_check(self) -> None:
        # Both dtype AND shape mismatch — must surface dtype error, not shape.
        pred = torch.zeros(3, 2, dtype=torch.float32)
        target = torch.zeros(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="dtype mismatch"):
            assert_same_shape(
                pred,
                target,
                name="MyLoss",
            )
