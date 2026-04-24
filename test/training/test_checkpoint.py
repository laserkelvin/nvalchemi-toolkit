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
"""Tests for :mod:`nvalchemi.training._checkpoint`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nvalchemi.training._checkpoint import load_checkpoint, save_checkpoint
from nvalchemi.training._spec import create_model_spec


class SwiGLU(nn.Module):
    """Custom activation: SwiGLU-style gated activation with a learnable scale.

    Splits the input channel dimension in half, applies SiLU to one half and
    multiplies by the other, then scales by a learnable parameter. Exercises
    :func:`create_model_spec`/:func:`save_checkpoint` against a module that
    owns its own :class:`~torch.nn.Parameter` rather than delegating to a
    stock layer.
    """

    def __init__(self, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return self.scale * (a * torch.nn.functional.silu(b))


class CustomMLPBlock(nn.Module):
    """Pre-norm MLP with a custom activation, a SwiGLU expansion, and dropout.

    The block is deliberately non-trivial: it stacks :class:`LayerNorm`, an
    expansion :class:`Linear` that feeds :class:`SwiGLU` (which halves the
    channel dimension), a projection :class:`Linear`, and :class:`Dropout`,
    with an optional residual connection. It stress tests the spec layer in
    three ways:

    1. ``__init__`` takes a mix of ints, floats, booleans, and a
       :class:`torch.dtype` — the latter routed through the custom type
       serializer registry.
    2. The module owns parameters at multiple nesting depths (top-level
       :class:`Linear` weights, plus the :class:`SwiGLU` scale parameter).
    3. The forward pass is stateless w.r.t. shape up to the channel
       dimension, so round-tripped weights must reproduce outputs exactly
       (modulo dropout, which we disable by calling ``.eval()``).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.1,
        eps: float = 1e-5,
        activation_scale: float = 1.0,
        use_residual: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if hidden_features % 2 != 0:
            raise ValueError(
                f"hidden_features must be even for SwiGLU, got {hidden_features}"
            )
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_residual = use_residual

        self.norm = nn.LayerNorm(in_features, eps=eps, dtype=dtype)
        self.expand = nn.Linear(in_features, hidden_features, dtype=dtype)
        self.activation = SwiGLU(init_scale=activation_scale)
        self.project = nn.Linear(hidden_features // 2, in_features, dtype=dtype)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.expand(h)
        h = self.activation(h)
        h = self.project(h)
        h = self.dropout(h)
        return x + h if self.use_residual else h


@dataclass
class NotModule:
    arg_a: int
    arg_b: str


class TestCheckpointRoundtrip:
    """Save/load round-trip behavior of :func:`save_checkpoint`/:func:`load_checkpoint`."""

    def test_save_load_basic(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(tmp_path, model, spec)
        assert idx == 0

        qualname_dir = tmp_path / "Linear"
        assert (qualname_dir / "spec.json").is_file()
        assert (qualname_dir / "checkpoints" / "0.pt").is_file()

        reloaded, reloaded_spec = load_checkpoint(qualname_dir)
        assert isinstance(reloaded, nn.Linear)
        assert torch.allclose(reloaded.weight, model.weight)
        assert torch.allclose(reloaded.bias, model.bias)
        assert reloaded_spec.init_hash == spec.init_hash

    def test_autoincrement(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx0 = save_checkpoint(tmp_path, model, spec)
        idx1 = save_checkpoint(tmp_path, model, spec)
        idx2 = save_checkpoint(tmp_path, model, spec)
        assert (idx0, idx1, idx2) == (0, 1, 2)

    def test_explicit_index_returned(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(tmp_path, model, spec, checkpoint_index=5)
        assert idx == 5
        assert (tmp_path / "Linear" / "checkpoints" / "5.pt").is_file()

    def test_spec_consistency_check(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec_a = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, model, spec_a)

        # Different hyperparameters at same root -> ValueError.
        spec_b = create_model_spec(nn.Linear, in_features=8, out_features=2)
        model_b = nn.Linear(8, 2)
        with pytest.raises(ValueError) as exc:
            save_checkpoint(tmp_path, model_b, spec_b)
        msg = str(exc.value)
        assert "in_features" in msg

    def test_load_latest_with_minus_one(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        for i in (0, 2, 5):
            save_checkpoint(tmp_path, model, spec, checkpoint_index=i)

        qualname_dir = tmp_path / "Linear"
        # Latest checkpoint by index should be index 5. Verify by comparing
        # file contents: overwrite index 5 with a mutated model, then
        # default-load and check weights match the mutated version.
        mutated = nn.Linear(4, 2)
        with torch.no_grad():
            mutated.weight.copy_(mutated.weight + 100.0)
        save_checkpoint(tmp_path, mutated, spec, checkpoint_index=5)

        reloaded, _ = load_checkpoint(qualname_dir, checkpoint_index=-1)
        assert torch.allclose(reloaded.weight, mutated.weight)

    def test_load_explicit_index(self, tmp_path: Path) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        model_a = nn.Linear(4, 2)
        model_b = nn.Linear(4, 2)
        # Ensure the two models are distinguishable.
        with torch.no_grad():
            model_b.weight.copy_(model_b.weight + 10.0)

        save_checkpoint(tmp_path, model_a, spec, checkpoint_index=1)
        save_checkpoint(tmp_path, model_b, spec, checkpoint_index=2)

        qualname_dir = tmp_path / "Linear"
        loaded_a, _ = load_checkpoint(qualname_dir, checkpoint_index=1)
        loaded_b, _ = load_checkpoint(qualname_dir, checkpoint_index=2)
        assert torch.allclose(loaded_a.weight, model_a.weight)
        assert torch.allclose(loaded_b.weight, model_b.weight)

    def test_load_missing_spec(self, tmp_path: Path) -> None:
        """Check that exception is raised when spec.json is missing"""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, model, spec)

        tmp_path.joinpath("Linear/spec.json").unlink()
        with pytest.raises(FileNotFoundError, match="does not contain a `spec.json`"):
            load_checkpoint(tmp_path)

    def test_non_module(self, tmp_path: Path) -> None:
        """load_checkpoint must refuse specs that build non-``nn.Module`` objects.

        Stages a ``spec.json`` for :class:`NotModule` (a plain dataclass) on
        disk, then points :func:`load_checkpoint` at it. The non-module
        guard fires after ``spec.build()`` and before any ``.pt`` file is
        read, so no checkpoint payload is required.
        """
        spec = create_model_spec(NotModule, arg_a=5, arg_b="hello")
        qualname_dir = tmp_path / "NotModule"
        qualname_dir.mkdir()
        (qualname_dir / "spec.json").write_text(spec.model_dump_json(indent=2))

        with pytest.raises(RuntimeError, match="a subclass of `nn.Module`"):
            load_checkpoint(qualname_dir)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available on this host"
    )
    def test_save_load_model_on_gpu(self, tmp_path: Path) -> None:
        """Round-trip a model whose parameters live on CUDA.

        The checkpoint layer stores raw tensors via ``torch.save`` and
        reloads with ``weights_only=True``; tensors retain their original
        device on save, and the reconstructed model (built by the spec on
        CPU) must still be loadable from a CUDA-resident state_dict.
        """
        device = torch.device("cuda")
        model = nn.Linear(4, 2).to(device)
        # Mutate so we can distinguish weights from a freshly-initialized reload.
        with torch.no_grad():
            model.weight.add_(1.25)
            model.bias.add_(-0.5)

        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(tmp_path, model, spec)
        assert idx == 0

        qualname_dir = tmp_path / "Linear"
        # Sanity: the saved tensors were CUDA-resident.
        saved = torch.load(qualname_dir / "checkpoints" / "0.pt", weights_only=True)
        assert saved["weight"].is_cuda
        assert saved["bias"].is_cuda

        reloaded, _ = load_checkpoint(qualname_dir)
        # Values match regardless of device; compare on CPU.
        assert torch.allclose(reloaded.weight.cpu(), model.weight.cpu())
        assert torch.allclose(reloaded.bias.cpu(), model.bias.cpu())

    def test_load_empty_checkpoints_dir(self, tmp_path: Path) -> None:
        # Create a valid spec.json but no .pt files.
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        qualname_dir = tmp_path / "Linear"
        (qualname_dir / "checkpoints").mkdir(parents=True)
        (qualname_dir / "spec.json").write_text(spec.model_dump_json(indent=2))

        with pytest.raises(FileNotFoundError, match="No checkpoints"):
            load_checkpoint(qualname_dir)

    def test_load_weights_only_true_used(self, tmp_path: Path) -> None:
        """Security: every ``torch.load`` must pass ``weights_only=True``."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, model, spec)
        qualname_dir = tmp_path / "Linear"

        # Patch the symbol as looked up inside _checkpoint.py.
        import nvalchemi.training._checkpoint as ckpt_mod

        real_load = ckpt_mod.torch.load
        with patch.object(ckpt_mod.torch, "load", wraps=real_load) as mock_load:
            load_checkpoint(qualname_dir)

        assert mock_load.call_count >= 1
        for call in mock_load.call_args_list:
            assert call.kwargs.get("weights_only") is True, (
                f"torch.load called without weights_only=True: {call}"
            )


class TestCheckpointCustomMLPBlock:
    """Stress tests: serialize a non-trivial custom block end-to-end.

    The target is :class:`CustomMLPBlock` — a pre-norm MLP wrapping a custom
    :class:`SwiGLU` activation, an expansion/projection :class:`Linear` pair,
    :class:`LayerNorm`, and :class:`Dropout`. These tests exercise the spec +
    checkpoint pipeline against a module that mixes several param types
    (weights, bias, learnable activation scale, LayerNorm affine params) and
    several kwarg types (int, float, bool, :class:`torch.dtype`).
    """

    @staticmethod
    def _make_spec(**overrides: object):
        kwargs: dict[str, object] = dict(
            in_features=8,
            hidden_features=16,
            dropout=0.25,
            eps=1e-6,
            activation_scale=0.5,
            use_residual=True,
            dtype=torch.float32,
        )
        kwargs.update(overrides)
        return kwargs, create_model_spec(CustomMLPBlock, **kwargs)

    def test_save_load_roundtrip_preserves_all_params(self, tmp_path: Path) -> None:
        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)
        # Perturb every parameter so defaults can't masquerade as matches.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        idx = save_checkpoint(tmp_path, model, spec)
        assert idx == 0

        qualname_dir = tmp_path / "CustomMLPBlock"
        assert (qualname_dir / "spec.json").is_file()
        assert (qualname_dir / "checkpoints" / "0.pt").is_file()

        reloaded, reloaded_spec = load_checkpoint(qualname_dir)
        assert isinstance(reloaded, CustomMLPBlock)
        assert reloaded_spec.init_hash == spec.init_hash

        # Every named parameter must round-trip bit-exactly.
        original_params = dict(model.named_parameters())
        reloaded_params = dict(reloaded.named_parameters())
        assert set(original_params) == set(reloaded_params)
        for name, tensor in original_params.items():
            assert torch.equal(reloaded_params[name], tensor), (
                f"parameter {name!r} differs after round-trip"
            )

        # And every named buffer (LayerNorm has none by default, but guard
        # against future additions).
        assert (
            dict(model.named_buffers()).keys() == dict(reloaded.named_buffers()).keys()
        )

    def test_roundtrip_preserves_forward_output(self, tmp_path: Path) -> None:
        kwargs, spec = self._make_spec(dropout=0.5)
        model = CustomMLPBlock(**kwargs).eval()  # .eval() disables dropout
        save_checkpoint(tmp_path, model, spec)

        reloaded, _ = load_checkpoint(tmp_path / "CustomMLPBlock")
        reloaded.eval()

        x = torch.randn(4, kwargs["in_features"])
        with torch.no_grad():
            y_original = model(x)
            y_reloaded = reloaded(x)
        assert torch.equal(y_original, y_reloaded)

    def test_spec_json_is_pure_json(self, tmp_path: Path) -> None:
        """spec.json must contain only JSON-native types; no pickled blobs."""
        import json

        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)
        save_checkpoint(tmp_path, model, spec)

        raw = (tmp_path / "CustomMLPBlock" / "spec.json").read_text()
        parsed = json.loads(raw)  # must not raise
        # dtype should be serialized as a string, not a pickled object.
        assert parsed["dtype"] == "torch.float32"
        # Key metadata fields are present.
        for key in ("cls_path", "timestamp", "init_hash"):
            assert key in parsed
        assert parsed["cls_path"].endswith(".CustomMLPBlock")

    def test_dtype_kwarg_round_trips_through_spec(self, tmp_path: Path) -> None:
        kwargs, spec = self._make_spec(dtype=torch.float64)
        model = CustomMLPBlock(**kwargs)
        save_checkpoint(tmp_path, model, spec)

        reloaded, reloaded_spec = load_checkpoint(tmp_path / "CustomMLPBlock")
        # The dtype kwarg must survive JSON round-trip as a torch.dtype.
        assert reloaded_spec.dtype is torch.float64
        # And it must actually be applied to the reconstructed parameters.
        assert reloaded.expand.weight.dtype is torch.float64
        assert reloaded.norm.weight.dtype is torch.float64

    def test_activation_parameter_is_checkpointed(self, tmp_path: Path) -> None:
        """The learnable ``SwiGLU.scale`` param must survive the round-trip."""
        kwargs, spec = self._make_spec(activation_scale=0.5)
        model = CustomMLPBlock(**kwargs)
        with torch.no_grad():
            model.activation.scale.fill_(7.5)

        save_checkpoint(tmp_path, model, spec)
        reloaded, _ = load_checkpoint(tmp_path / "CustomMLPBlock")
        assert torch.equal(reloaded.activation.scale, torch.tensor(7.5))

    def test_autoincrement_multiple_checkpoints(self, tmp_path: Path) -> None:
        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)

        # Simulate 3 "training steps" by mutating the projection weight
        # between saves, then verify each saved checkpoint reloads as
        # itself (not as a later state).
        snapshots: list[torch.Tensor] = []
        for step in range(3):
            with torch.no_grad():
                model.project.weight.add_(float(step) + 1.0)
            snapshots.append(model.project.weight.detach().clone())
            idx = save_checkpoint(tmp_path, model, spec)
            assert idx == step

        qualname_dir = tmp_path / "CustomMLPBlock"
        for step, snapshot in enumerate(snapshots):
            reloaded, _ = load_checkpoint(qualname_dir, checkpoint_index=step)
            assert torch.equal(reloaded.project.weight, snapshot), (
                f"checkpoint {step} did not reload its own weights"
            )

    def test_spec_mismatch_on_hyperparameter_change(self, tmp_path: Path) -> None:
        """Saving a second spec with different hyperparameters must fail."""
        kwargs_a, spec_a = self._make_spec(hidden_features=16)
        model_a = CustomMLPBlock(**kwargs_a)
        save_checkpoint(tmp_path, model_a, spec_a)

        kwargs_b, spec_b = self._make_spec(hidden_features=32)
        model_b = CustomMLPBlock(**kwargs_b)
        with pytest.raises(ValueError) as exc:
            save_checkpoint(tmp_path, model_b, spec_b)
        assert "hidden_features" in str(exc.value)

    def test_invalid_hyperparameter_still_fails_at_build(self, tmp_path: Path) -> None:
        """A spec with hyperparameters the class rejects must fail at build().

        ``hidden_features=15`` is odd, which ``CustomMLPBlock.__init__``
        explicitly rejects. The spec itself is constructible (spec creation
        does not invoke the target class), but :func:`load_checkpoint`
        should surface the class's own ``ValueError``.
        """
        # Build + save by hand so we can write a malformed spec without
        # going through ``CustomMLPBlock(...)``.
        spec = create_model_spec(
            CustomMLPBlock,
            in_features=8,
            hidden_features=15,  # invalid: odd
            dropout=0.1,
            eps=1e-5,
            activation_scale=1.0,
            use_residual=True,
            dtype=torch.float32,
        )
        qualname_dir = tmp_path / "CustomMLPBlock"
        (qualname_dir / "checkpoints").mkdir(parents=True)
        (qualname_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
        # A dummy .pt so load_checkpoint gets past the "no checkpoints" check
        # ... except we expect the failure to happen earlier, at build().
        torch.save({}, qualname_dir / "checkpoints" / "0.pt")

        with pytest.raises(ValueError, match="hidden_features must be even"):
            load_checkpoint(qualname_dir)
