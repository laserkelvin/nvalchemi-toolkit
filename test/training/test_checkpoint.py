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

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nvalchemi.training._checkpoint import load_checkpoint, save_checkpoint
from nvalchemi.training._spec import create_model_spec


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

    def test_load_wrong_path_footgun(self, tmp_path: Path) -> None:
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, model, spec)

        # Caller mistakenly passes the parent, not the qualname subdir.
        with pytest.raises(FileNotFoundError) as exc:
            load_checkpoint(tmp_path)
        msg = str(exc.value)
        assert "qualname subdirectory" in msg
        assert str(tmp_path / "Linear") in msg

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
