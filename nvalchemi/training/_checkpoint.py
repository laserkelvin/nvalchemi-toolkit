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
"""No-pickle checkpoint layer over :class:`~nvalchemi.training._spec.BaseSpec`.

This module pairs a JSON-serialized :class:`BaseSpec` with a ``state_dict``
on disk so that an MLIP (or any :class:`torch.nn.Module`) can be saved and
reloaded without relying on :mod:`pickle`. Both halves are required:
``spec.json`` records *how to build* the model, and the numbered ``.pt``
files record the *weights* of successive checkpoints.

Layout
------
A single call to :func:`save_checkpoint` writes::

    {root_folder}/
      {ModelQualname}/
        spec.json
        checkpoints/
          0.pt
          1.pt
          ...

The ``{ModelQualname}`` subdirectory is derived from
``type(model).__qualname__`` with any ``.`` replaced by ``_`` so that
nested-class qualnames (``Outer.Inner``) do not introduce extra path
segments. This layout lets multiple distinct models coexist under one
``root_folder`` without interfering.

To reload, callers pass the qualname subdirectory directly — i.e. the
directory that *contains* ``spec.json``::

    save_checkpoint("runs/exp1", model, spec)         # writes runs/exp1/MyMLIP/...
    model2, spec2 = load_checkpoint("runs/exp1/MyMLIP")

The caller always knows which model they want to reload; discovery would
add ambiguity without a compelling use case.

Security rationale
------------------
Model weights are stored with ``torch.save(model.state_dict(), ...)`` — a
plain tensor bundle, not a pickle of a Python object graph — and reloaded
with ``torch.load(..., weights_only=True)``. This is the only pickle-free
code path PyTorch offers for weight bundles and is mandatory here: loading
a checkpoint from an untrusted source must not be able to execute arbitrary
code.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> import torch.nn as nn
>>> from nvalchemi.training._spec import create_model_spec
>>> with tempfile.TemporaryDirectory() as tmp:
...     model = nn.Linear(4, 2)
...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
...     idx = save_checkpoint(tmp, model, spec)
...     model2, spec2 = load_checkpoint(Path(tmp) / "Linear")
...     assert idx == 0 and isinstance(model2, nn.Linear)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json


def _ckpt_indices(ckpt_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if p.stem.isdigit())


def save_checkpoint(
    root_folder: Path | str,
    model: nn.Module,
    spec: BaseSpec,
    checkpoint_index: int = -1,
) -> int:
    """Save a model ``state_dict`` alongside its :class:`BaseSpec`.

    See the module docstring for the on-disk layout and security rationale.

    Parameters
    ----------
    root_folder
        Parent directory. The qualname subdirectory is created under it.
    model
        Module whose ``state_dict`` is persisted. The model itself is
        never pickled.
    spec
        Spec describing how to rebuild ``model``. On the first save its
        JSON form is written; on subsequent saves it is compared to the
        existing ``spec.json`` (ignoring ``timestamp``) and a mismatch
        raises :class:`ValueError`.
    checkpoint_index
        Index of the checkpoint file. ``-1`` (the default) autoincrements
        past the highest existing index, or starts at ``0`` if none exist.

    Returns
    -------
    int
        The checkpoint index that was written.

    Raises
    ------
    ValueError
        If an existing ``spec.json`` disagrees with ``spec`` on any field
        other than ``timestamp``.

    Examples
    --------
    >>> import tempfile, torch.nn as nn
    >>> from nvalchemi.training._spec import create_model_spec
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
    ...     save_checkpoint(tmp, nn.Linear(4, 2), spec)
    0
    """
    qualname_dir = Path(root_folder) / type(model).__qualname__.replace(".", "_")
    ckpt_dir = qualname_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    spec_path = qualname_dir / "spec.json"
    spec_json = spec.model_dump_json(indent=2)
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        new_spec = json.loads(spec_json)
        existing.pop("timestamp", None)
        new_spec.pop("timestamp", None)
        if existing != new_spec:
            diffs = sorted(
                k
                for k in set(existing) | set(new_spec)
                if existing.get(k) != new_spec.get(k)
            )
            preview = ", ".join(
                f"{k}: {existing.get(k)!r} -> {new_spec.get(k)!r}" for k in diffs[:3]
            )
            suffix = f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""
            raise ValueError(
                f"spec.json at {spec_path} disagrees with the spec being saved. "
                f"Differing fields: {preview}{suffix}."
            )
    else:
        spec_path.write_text(spec_json)

    if checkpoint_index == -1:
        existing_idx = _ckpt_indices(ckpt_dir)
        checkpoint_index = (existing_idx[-1] + 1) if existing_idx else 0

    torch.save(model.state_dict(), ckpt_dir / f"{checkpoint_index}.pt")
    return checkpoint_index


def load_checkpoint(
    root_folder: Path | str,
    checkpoint_index: int = -1,
) -> tuple[nn.Module, BaseSpec]:
    """Load a model and its spec written by :func:`save_checkpoint`.

    See the module docstring for the on-disk layout and security rationale.
    ``root_folder`` is the qualname subdirectory that directly contains
    ``spec.json`` — *not* the parent passed to :func:`save_checkpoint`.

    Parameters
    ----------
    root_folder
        Directory containing ``spec.json`` and a ``checkpoints/``
        subdirectory with one or more ``{N}.pt`` files.
    checkpoint_index
        Index of the checkpoint file to load. ``-1`` (the default) selects
        the highest available index.

    Returns
    -------
    tuple of (torch.nn.Module, BaseSpec)
        The reconstructed module with weights loaded, and the spec used to
        build it.

    Raises
    ------
    FileNotFoundError
        If ``root_folder/spec.json`` does not exist, or if
        ``checkpoint_index == -1`` but no ``.pt`` files are present in
        ``root_folder/checkpoints``.
    ValueError
        If the spec JSON is malformed; see
        :func:`~nvalchemi.training._spec.create_model_spec_from_json`.

    Examples
    --------
    >>> from pathlib import Path
    >>> model, spec = load_checkpoint(Path(tmp) / "Linear")  # doctest: +SKIP
    """
    root = Path(root_folder)
    spec_path = root / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {root} does not contain a `spec.json`"
            " file which is required to reconstruct the model"
            " architecture. Make sure this file exists and in"
            " the future use the `save_checkpoint` method for serialization."
        )
    spec_json = json.loads(spec_path.read_text())
    spec = create_model_spec_from_json(spec_json)
    model = spec.build()
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            f"Specification at {spec_path} was expected to instantiate"
            f" a subclass of `nn.Module`, but got {type(model)}."
        )

    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.exists() and not ckpt_dir.is_dir():
        raise RuntimeError(f"Expected {ckpt_dir} to exist and be a folder.")
    if checkpoint_index == -1:
        existing = _ckpt_indices(ckpt_dir)
        if not existing:
            other = (
                sorted(p.name for p in ckpt_dir.glob("*.pt"))[:3]
                if ckpt_dir.is_dir()
                else []
            )
            hint = f" (found non-numeric .pt files: {other})" if other else ""
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}{hint}")
        checkpoint_index = existing[-1]

    weights = torch.load(ckpt_dir / f"{checkpoint_index}.pt", weights_only=True)
    model.load_state_dict(weights)
    return model, spec
