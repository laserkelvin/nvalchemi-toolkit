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
"""Multi-component, manifest-based checkpoint layer.

This module saves and loads checkpoints for multiple named models,
optimizers, and schedulers without relying on :mod:`pickle`. A top-level
``manifest.json`` coordinates all components and their associations.

Layout
------
A single call to :func:`save_checkpoint` writes::

    {root_folder}/
      manifest.json
      models/{name}/
        spec.json
        checkpoints/{N}.pt
      optimizers/{name}/          # optional
        spec.json
        checkpoints/{N}.pt
      schedulers/{name}/          # optional
        spec.json
        checkpoints/{N}.pt

The ``manifest.json`` records which components are present, the latest
checkpoint index, and optional associations that wire optimizers to models
and schedulers to optimizers::

    {
      "checkpoint_index": 0,
      "models": ["student", "teacher"],
      "optimizers": ["student_opt"],
      "schedulers": ["student_sched"],
      "associations": {
        "student": {
          "optimizers": ["student_opt"],
          "schedulers": ["student_sched"]
        }
      }
    }

The ``associations`` key specifies connectivity between models and
their respective optimizer(s) and LR scheduler(s). This can be explicitly
provided by the user, or automatically inferred by matching parameters
with optimizers/LR schedulers.

Examples
--------
Single model::

    save_checkpoint("runs/exp1", models={"main": (model, spec)})
    result = load_checkpoint("runs/exp1")
    model, spec = result.models["main"]

Knowledge distillation (two models + optimizer + scheduler)::

    save_checkpoint(
        "runs/kd",
        models={"student": (student, s_spec), "teacher": (teacher, t_spec)},
        optimizers={"s_opt": (optimizer, opt_spec)},
        schedulers={"s_sched": (scheduler, sched_spec)},
        # associations can be inferred automatically from param_groups
    )
    result = load_checkpoint("runs/kd")
    student, _ = result.models["student"]
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn as nn
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from nvalchemi.training._spec import BaseSpec, create_model_spec_from_json

# ---------------------------------------------------------------------------
# Dual-mode field helpers
# ---------------------------------------------------------------------------


def _component_before(v: Any) -> dict[str, Any]:
    """Accept ``list[str]`` (from JSON) or ``dict`` (from code) for component fields."""
    if isinstance(v, list):
        # From disk: list of names → placeholder dict (values populated later)
        return {name: None for name in v}
    return v


def _component_serialize(d: dict[str, Any]) -> list[str]:
    """Serialize a component dict to a sorted list of its keys."""
    return sorted(d.keys())


# ---------------------------------------------------------------------------
# Manifest schema + runtime container (unified)
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1
"""Current manifest schema version.  Bump when manifest structure changes."""

# Type aliases for the runtime dict shapes
_ModelDict = dict[str, tuple[nn.Module, BaseSpec] | None]
_OptimizerDict = dict[str, tuple[torch.optim.Optimizer, BaseSpec] | None]
_SchedulerDict = dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec] | None]


class CheckpointManifest(BaseModel):
    """Unified checkpoint manifest and runtime container.

    This Pydantic model serves a dual role:

    1. **On-disk schema** — ``manifest.json`` stores component names as
       sorted string lists together with metadata and associations.
    2. **Runtime container** — after :func:`load_checkpoint` hydrates the
       components, the same instance carries live ``(object, spec)`` tuples.

    The ``models``, ``optimizers``, and ``schedulers`` fields accept
    either a ``list[str]`` (from JSON) or a ``dict[str, tuple]`` (from
    code).  Serialization always produces sorted name lists via
    :class:`~pydantic.PlainSerializer`.

    Attributes
    ----------
    schema_version
        Schema version. Defaults to the current ``_SCHEMA_VERSION``.
        When manifest structure changes, bump ``_SCHEMA_VERSION`` and
        add a migration step in :meth:`read`.
    checkpoint_index
        The latest checkpoint index written.
    models
        Component dict keyed by name. At runtime each value is a
        ``(nn.Module, BaseSpec)`` tuple; on disk, serialized as a
        sorted ``list[str]`` of names.
    optimizers
        Same dual-mode dict for optimizers (empty by default).
    schedulers
        Same dual-mode dict for schedulers (empty by default).
    associations
        Model-centric linkage: maps a model name to
        ``{"optimizers": [...], "schedulers": [...]}``.

    Examples
    --------
    >>> manifest = CheckpointManifest(
    ...     checkpoint_index=0, models={"main": None},
    ... )
    >>> manifest.model_dump()["models"]
    ['main']
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: Annotated[
        int, Field(default=_SCHEMA_VERSION, description="Manifest schema version.")
    ]
    checkpoint_index: Annotated[
        int, Field(description="Latest checkpoint index written.")
    ]
    models: Annotated[
        _ModelDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(description="Model components keyed by name."),
    ]
    optimizers: Annotated[
        _OptimizerDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(default_factory=dict, description="Optimizer components keyed by name."),
    ]
    schedulers: Annotated[
        _SchedulerDict,
        BeforeValidator(_component_before),
        PlainSerializer(_component_serialize, return_type=list[str]),
        Field(default_factory=dict, description="Scheduler components keyed by name."),
    ]
    associations: Annotated[
        dict[str, dict[str, list[str]]],
        Field(
            default_factory=dict,
            description="Model-centric linkage to optimizers/schedulers.",
        ),
    ]

    @staticmethod
    def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
        """Migrate an older manifest dict to the current schema version.

        Parameters
        ----------
        raw
            Parsed ``manifest.json`` content.

        Returns
        -------
        dict[str, Any]
            Dict conforming to the current ``_SCHEMA_VERSION``, ready
            for :meth:`pydantic.BaseModel.model_validate`.

        Raises
        ------
        ValueError
            If the manifest's schema version is newer than supported.
        """
        version = raw.get("schema_version", 0)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema version {version} is newer than supported "
                f"({_SCHEMA_VERSION}). Upgrade nvalchemi to load this checkpoint."
            )
        # Future migrations chain here:
        # if version < 1:
        #     raw = _migrate_v0_to_v1(raw)
        raw["schema_version"] = _SCHEMA_VERSION
        return raw

    @classmethod
    def read(cls, root: Path) -> CheckpointManifest:
        """Read, migrate, and validate ``manifest.json`` from *root*.

        Parameters
        ----------
        root
            Checkpoint root directory containing ``manifest.json``.

        Returns
        -------
        CheckpointManifest
            Validated manifest instance. Component dicts contain
            placeholder ``None`` values until hydrated by
            :func:`load_checkpoint`.

        Raises
        ------
        FileNotFoundError
            If ``manifest.json`` does not exist.
        ValueError
            If the manifest's schema version is newer than supported.
        pydantic.ValidationError
            If the manifest JSON does not conform to the schema.
        """
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json found in {root}. Use save_checkpoint to "
                f"create a checkpoint first."
            )
        raw = json.loads(manifest_path.read_text())
        migrated = cls._migrate(raw)
        return cls.model_validate(migrated)

    def write(self, root: Path) -> None:
        """Write this manifest to ``{root}/manifest.json``.

        Parameters
        ----------
        root
            Checkpoint root directory.
        """
        (root / "manifest.json").write_text(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ckpt_indices(ckpt_dir: Path) -> list[int]:
    """Return sorted integer stems from ``*.pt`` files in *ckpt_dir*."""
    return sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if p.stem.isdigit())


def _check_spec_consistency(spec_path: Path, spec: BaseSpec) -> None:
    """Write *spec* to *spec_path* on first call; raise on mismatch thereafter.

    Parameters
    ----------
    spec_path
        Path to the ``spec.json`` file.
    spec
        The spec to write or compare against the existing file.

    Raises
    ------
    ValueError
        If the existing ``spec.json`` disagrees with *spec* on any field
        other than ``timestamp``.
    """
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
                f"spec.json at {spec_path} disagrees with the spec being "
                f"saved. Differing fields: {preview}{suffix}."
            )
    else:
        spec_path.write_text(spec_json)


def _save_component(
    root: Path,
    category: str,
    name: str,
    state_dict: dict[str, Any],
    spec: BaseSpec,
    checkpoint_index: int,
) -> None:
    """Write *spec* and *state_dict* under ``root/category/name/``."""
    comp_dir = root / category / name
    ckpt_dir = comp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _check_spec_consistency(comp_dir / "spec.json", spec)
    torch.save(state_dict, ckpt_dir / f"{checkpoint_index}.pt")


def _infer_associations(
    models: dict[str, tuple[nn.Module, BaseSpec]],
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]],
    schedulers: dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]],
) -> dict[str, dict[str, list[str]]]:
    """Infer model-centric associations from optimizer ``param_groups``.

    For each optimizer, collect the ``data_ptr()`` values of every parameter
    in its ``param_groups`` and match against each model's ``parameters()``.
    The optimizer is associated with every model that owns at least one of
    those parameters.

    Schedulers are linked to their optimizer via
    ``scheduler.optimizer is optimizer`` identity checks.

    Parameters
    ----------
    models
        ``{name: (module, spec)}`` mapping.
    optimizers
        ``{name: (optimizer, spec)}`` mapping.
    schedulers
        ``{name: (scheduler, spec)}`` mapping.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Model-centric associations, e.g.
        ``{"student": {"optimizers": ["s_opt"], "schedulers": ["s_sched"]}}``.
    """
    # Build data_ptr → model_name index
    ptr_to_model: dict[int, str] = {}
    for model_name, (module, _) in models.items():
        for p in module.parameters():
            ptr_to_model[p.data_ptr()] = model_name

    # Map each optimizer to every model that owns at least one parameter
    opt_to_models: dict[str, list[str]] = {}
    for opt_name, (optimizer, _) in optimizers.items():
        matched: dict[str, bool] = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                owner = ptr_to_model.get(p.data_ptr())
                if owner is not None:
                    matched[owner] = True
        if matched:
            opt_to_models[opt_name] = list(matched)

    # Map each scheduler to its optimizer (identity check)
    sched_to_opt: dict[str, str] = {}
    for sched_name, (scheduler, _) in schedulers.items():
        for opt_name, (optimizer, _) in optimizers.items():
            if scheduler.optimizer is optimizer:  # type: ignore[attr-defined]
                sched_to_opt[sched_name] = opt_name
                break

    # Build model-centric structure
    assoc: dict[str, dict[str, list[str]]] = {}
    for opt_name, model_names in opt_to_models.items():
        for model_name in model_names:
            assoc.setdefault(model_name, {"optimizers": [], "schedulers": []})
            assoc[model_name]["optimizers"].append(opt_name)
    for sched_name, opt_name in sched_to_opt.items():
        model_names = opt_to_models.get(opt_name, [])
        for model_name in model_names:
            assoc.setdefault(model_name, {"optimizers": [], "schedulers": []})
            assoc[model_name]["schedulers"].append(sched_name)

    return assoc


def _find_associated_model_params(
    optimizer_name: str,
    associations: dict[str, dict[str, list[str]]],
    models: dict[str, tuple[nn.Module, BaseSpec]],
) -> Iterator[torch.nn.Parameter]:
    """Return chained parameters from all models associated with *optimizer_name*."""
    matched: list[str] = []
    for model_name, assoc in associations.items():
        if optimizer_name in assoc.get("optimizers", []):
            matched.append(model_name)
    if matched:
        return itertools.chain.from_iterable(
            models[name][0].parameters() for name in matched
        )
    # Fallback: if exactly one model exists, use it
    if len(models) == 1:
        return next(iter(models.values()))[0].parameters()
    raise ValueError(
        f"Cannot determine which model's parameters to use for optimizer "
        f"{optimizer_name!r}. Provide associations or use a single model."
    )


def _find_associated_optimizer(
    scheduler_name: str,
    associations: dict[str, dict[str, list[str]]],
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]],
) -> torch.optim.Optimizer:
    """Return the optimizer whose associations include *scheduler_name*."""
    for assoc in associations.values():
        if scheduler_name in assoc.get("schedulers", []):
            for opt_name in assoc.get("optimizers", []):
                if opt_name in optimizers:
                    return optimizers[opt_name][0]
    # Fallback: if exactly one optimizer exists, use it
    if len(optimizers) == 1:
        return next(iter(optimizers.values()))[0]
    raise ValueError(
        f"Cannot determine which optimizer to use for scheduler "
        f"{scheduler_name!r}. Provide associations or use a single optimizer."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    root_folder: Path | str,
    models: dict[str, tuple[nn.Module, BaseSpec]],
    optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] | None = None,
    schedulers: (
        dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]] | None
    ) = None,
    associations: dict[str, dict[str, list[str]]] | None = None,
    checkpoint_index: int = -1,
) -> int:
    """Save a multi-component checkpoint with a manifest.

    Each component (model, optimizer, scheduler) is saved as a
    ``state_dict`` under its own named subdirectory. A ``manifest.json``
    at the root records all component names and their associations.

    Parameters
    ----------
    root_folder
        Root directory for the checkpoint tree.
    models
        Mapping of model name to ``(module, spec)`` pairs. At least one
        model is required.
    optimizers
        Optional mapping of optimizer name to ``(optimizer, spec)`` pairs.
    schedulers
        Optional mapping of scheduler name to ``(scheduler, spec)`` pairs.
    associations
        Optional model-centric linkage mapping a model name to
        ``{"optimizers": [...], "schedulers": [...]}``. When ``None``
        (default), associations are inferred automatically by matching
        optimizer ``param_groups`` to model parameters via ``data_ptr()``
        identity, and schedulers to optimizers via object identity.
    checkpoint_index
        Index for the checkpoint files. ``-1`` (default) auto-increments
        from the manifest's last index, or starts at ``0``.

    Returns
    -------
    int
        The checkpoint index that was written.

    Raises
    ------
    ValueError
        If an existing ``spec.json`` disagrees with the spec being saved
        (ignoring ``timestamp``).

    Examples
    --------
    >>> import tempfile, torch.nn as nn
    >>> from nvalchemi.training._spec import create_model_spec
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
    ...     save_checkpoint(tmp, models={"main": (nn.Linear(4, 2), spec)})
    0
    """
    root = Path(root_folder)
    optimizers = optimizers or {}
    schedulers = schedulers or {}
    if associations is None:
        associations = _infer_associations(models, optimizers, schedulers)

    # Resolve checkpoint index
    if checkpoint_index == -1:
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            prev = CheckpointManifest.read(root)
            checkpoint_index = prev.checkpoint_index + 1
        else:
            checkpoint_index = 0

    # Save each component category
    for name, (module, spec) in models.items():
        _save_component(
            root, "models", name, module.state_dict(), spec, checkpoint_index
        )

    for name, (opt, spec) in optimizers.items():
        _save_component(
            root, "optimizers", name, opt.state_dict(), spec, checkpoint_index
        )

    for name, (sched, spec) in schedulers.items():
        _save_component(
            root, "schedulers", name, sched.state_dict(), spec, checkpoint_index
        )

    # Write manifest — pass live dicts directly; PlainSerializer extracts keys
    manifest = CheckpointManifest(
        checkpoint_index=checkpoint_index,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        associations=associations,
    )
    manifest.write(root)
    return checkpoint_index


def load_checkpoint(
    root_folder: Path | str,
    checkpoint_index: int = -1,
    map_location: str | torch.device | None = None,
    model_names: Iterable[str] | None = None,
) -> CheckpointManifest:
    """Load a multi-component checkpoint written by :func:`save_checkpoint`.

    Components are rebuilt in dependency order: models first, then
    optimizers (which need model parameters), then schedulers (which need
    an optimizer instance). Associations from the manifest wire each
    optimizer to the correct model and each scheduler to the correct
    optimizer.

    Parameters
    ----------
    root_folder
        Root directory containing ``manifest.json``.
    checkpoint_index
        Index of the checkpoint to load. ``-1`` (default) loads the
        latest index recorded in the manifest.
    map_location
        Forwarded to every :func:`torch.load` call. When not ``None``,
        each loaded model is additionally moved via
        ``model.to(map_location)``. Optimizers and schedulers have their
        state placed by ``torch.load`` alone (they lack a standard
        ``.to()`` API).
    model_names
        If given, load only the models with these names together with the
        optimizers and schedulers wired to them through
        ``manifest.associations``. Accepts any iterable of strings
        (typically a set). ``None`` (default) loads every component on
        disk. The returned manifest's ``associations`` still reflects the
        full on-disk mapping, so callers can inspect what was not loaded.

    Returns
    -------
    CheckpointManifest
        Manifest with hydrated ``models``, ``optimizers``, ``schedulers``
        dicts containing live ``(object, spec)`` tuples, plus
        ``associations`` and ``checkpoint_index``. When ``model_names``
        is set, the hydrated dicts contain only the selected subset.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` is missing or a checkpoint ``.pt`` file
        does not exist.
    KeyError
        If any name in ``model_names`` does not appear in
        ``manifest.models``.
    RuntimeError
        If a model spec does not build an :class:`~torch.nn.Module`.

    Examples
    --------
    >>> import tempfile, torch.nn as nn
    >>> from nvalchemi.training._spec import create_model_spec
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
    ...     _ = save_checkpoint(tmp, models={"main": (nn.Linear(4, 2), spec)})
    ...     result = load_checkpoint(tmp)
    ...     isinstance(result.models["main"][0], nn.Linear)
    True

    Loading onto CPU regardless of the original device::

        result = load_checkpoint("runs/exp1", map_location="cpu")

    Selecting a subset of models (e.g., teacher and student but not the
    third auxiliary model)::

        result = load_checkpoint("runs/kd", model_names={"teacher", "student"})
    """
    root = Path(root_folder)
    manifest = CheckpointManifest.read(root)

    if checkpoint_index == -1:
        checkpoint_index = manifest.checkpoint_index

    associations = manifest.associations

    # determine what models to load
    selected_models = set(manifest.models) if model_names is None else set(model_names)
    unknown = selected_models - set(manifest.models)
    if unknown:
        raise KeyError(
            f"Unknown model(s) {sorted(unknown)!r}. "
            f"Available: {sorted(manifest.models)!r}"
        )

    # Build the load set as the union of each selected model's associations.
    # When ``model_names is None`` this is equivalent to loading every
    # component listed in the manifest.
    models_to_load = [n for n in manifest.models if n in selected_models]
    if model_names is None:
        optimizers_to_load = list(manifest.optimizers)
        schedulers_to_load = list(manifest.schedulers)
    else:
        wanted_optimizers: set[str] = set()
        wanted_schedulers: set[str] = set()
        for n in selected_models:
            assoc = associations.get(n, {})
            wanted_optimizers.update(assoc.get("optimizers", []))
            wanted_schedulers.update(assoc.get("schedulers", []))
        optimizers_to_load = [n for n in manifest.optimizers if n in wanted_optimizers]
        schedulers_to_load = [n for n in manifest.schedulers if n in wanted_schedulers]

    # --- Models ---
    loaded_models: dict[str, tuple[nn.Module, BaseSpec]] = {}
    for name in models_to_load:
        spec = _load_spec(root / "models" / name / "spec.json")
        model = spec.build()
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                f"Model spec for {name!r} built {type(model)!r}, expected nn.Module."
            )
        # Move the freshly-built (uninitialized) module to the target device
        # before loading weights so that ``load_state_dict`` is a
        # device-local copy and we avoid a double transfer.
        if map_location is not None:
            model.to(map_location)
        weights = torch.load(
            root / "models" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        model.load_state_dict(weights)
        loaded_models[name] = (model, spec)

    # --- Optimizers ---
    loaded_optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] = {}
    for name in optimizers_to_load:
        spec = _load_spec(root / "optimizers" / name / "spec.json")
        params = _find_associated_model_params(name, associations, loaded_models)
        optimizer = spec.build(params)
        state = torch.load(
            root / "optimizers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        optimizer.load_state_dict(state)
        loaded_optimizers[name] = (optimizer, spec)

    # --- Schedulers ---
    loaded_schedulers: dict[
        str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]
    ] = {}
    for name in schedulers_to_load:
        spec = _load_spec(root / "schedulers" / name / "spec.json")
        assoc_optimizer = _find_associated_optimizer(
            name, associations, loaded_optimizers
        )
        scheduler = spec.build(assoc_optimizer)
        state = torch.load(
            root / "schedulers" / name / "checkpoints" / f"{checkpoint_index}.pt",
            weights_only=True,
            map_location=map_location,
        )
        scheduler.load_state_dict(state)
        loaded_schedulers[name] = (scheduler, spec)

    # Hydrate manifest with live objects
    manifest.models = loaded_models
    manifest.optimizers = loaded_optimizers
    manifest.schedulers = loaded_schedulers
    manifest.checkpoint_index = checkpoint_index
    return manifest


def _load_spec(spec_path: Path) -> BaseSpec:
    """Read and rehydrate a :class:`BaseSpec` from *spec_path*."""
    if not spec_path.exists():
        raise FileNotFoundError(f"Expected spec at {spec_path} but file not found.")
    return create_model_spec_from_json(json.loads(spec_path.read_text()))
