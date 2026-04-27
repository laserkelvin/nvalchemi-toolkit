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
"""Reproducible, no-pickle serialization of MLIP hyperparameters.

This module provides :class:`BaseSpec`, a Pydantic model that captures the
``__init__`` arguments of any target class --- typically an MLIP, an optimizer,
or a learning-rate scheduler --- and serializes them to plain JSON. Spec
reconstruction imports the target class by its dotted path and instantiates
it with the stored kwargs. This approach ensures that ``pickle`` is not needed
to recreate objects at runtime:

- Hyperparameters are stored as plain JSON (strings, numbers, lists, dicts).
- :class:`torch.Tensor` is serialized as ``{dtype, shape, data}`` — a data
  structure, not a bytecode payload.
- :class:`torch.dtype` is serialized as its string name and rehydrated with
  an :func:`isinstance` guard so that an attacker-controlled string cannot
  smuggle arbitrary ``torch.*`` attributes through :func:`getattr`.
- Model weights (stored separately) must be loaded with
  ``torch.load(..., weights_only=True)`` — the only pickle-free code path
  that PyTorch offers for weight bundles.

Custom (de)serializers for additional types are registered via
:func:`register_type_serializer`. The module pre-registers handlers for
:class:`torch.dtype`, :class:`torch.device`, and :class:`torch.Tensor`.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Annotated, Any

import torch
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializeAsAny,
    create_model,
)

_META_FIELDS: frozenset[str] = frozenset({"cls_path", "timestamp", "init_hash"})
"""Field names reserved by :class:`BaseSpec` itself; never forwarded to ``build``."""


# ---------------------------------------------------------------------------
# Custom type serializer registry (torch.dtype, torch.device, torch.Tensor, ...)
# ---------------------------------------------------------------------------

_TYPE_SERIALIZERS: dict[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {}
"""Registry mapping a type to its ``(serialize, deserialize)`` callable pair."""


def register_type_serializer(
    type_: type,
    serialize: Callable[[Any], Any],
    deserialize: Callable[[Any], Any],
) -> None:
    """Register JSON (de)serializers for a custom type.

    Registered types can appear as field values on a :class:`BaseSpec`
    subclass; :func:`create_model_spec` wraps them in a Pydantic
    :class:`~pydantic.BeforeValidator` / :class:`~pydantic.PlainSerializer`
    pair so that :meth:`~pydantic.BaseModel.model_dump_json` and
    :meth:`~pydantic.BaseModel.model_validate` round-trip values through the
    provided hooks.

    Parameters
    ----------
    type_
        The Python type to register, for example :class:`torch.dtype`.
    serialize
        Callable converting a ``type_`` instance to a JSON-safe value
        (usually a :class:`str` or a plain :class:`dict`).
    deserialize
        Callable converting the JSON-safe value back into a ``type_``
        instance. Must be tolerant of the case where it is handed an
        already-rehydrated ``type_`` instance (the wrapper short-circuits in
        that case, but custom implementations should not crash on it).

    Notes
    -----
    Re-registering an already-registered ``type_`` silently replaces the
    previous ``(serialize, deserialize)`` pair; no warning is emitted. This
    matches the prototype and allows downstream code to override built-in
    handlers for :class:`torch.dtype`, :class:`torch.device`, and
    :class:`torch.Tensor`. Callers that need to detect collisions can test
    ``type_ in _TYPE_SERIALIZERS`` against the module-private registry
    before registering.

    Examples
    --------
    >>> import torch
    >>> register_type_serializer(
    ...     torch.device,
    ...     serialize=str,
    ...     deserialize=torch.device,
    ... )
    """
    _TYPE_SERIALIZERS[type_] = (serialize, deserialize)


def _wrap_custom_type(t: type) -> Any:
    """Wrap a registered type in an ``Annotated[...]`` with Pydantic hooks."""
    ser, deser = _TYPE_SERIALIZERS[t]

    def _before(v: Any) -> Any:
        return v if isinstance(v, t) else deser(v)

    return Annotated[t, BeforeValidator(_before), PlainSerializer(ser)]


def _dtype_deserialize(s: Any) -> torch.dtype:
    """Rehydrate a :class:`torch.dtype` from its string form with a type guard."""
    if isinstance(s, torch.dtype):
        return s
    if not isinstance(s, str):
        raise TypeError(
            f"torch.dtype deserializer expected str, got {type(s).__name__}"
        )
    result = getattr(torch, s.removeprefix("torch."), None)
    if not isinstance(result, torch.dtype):
        raise ValueError(
            f"{s!r} does not resolve to a torch.dtype "
            "(defense-in-depth against attacker-controlled JSON smuggling "
            "non-dtype torch.* attributes)."
        )
    return result


def _tensor_serialize(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a :class:`torch.Tensor` as ``{data, dtype, shape}``."""
    return {
        "data": t.detach().cpu().tolist(),
        "dtype": str(t.dtype),
        "shape": list(t.shape),
    }


def _tensor_deserialize(v: Any) -> torch.Tensor:
    """Rehydrate a :class:`torch.Tensor` from its ``{data, dtype, shape}`` dict."""
    if isinstance(v, torch.Tensor):
        return v
    if not isinstance(v, dict):
        raise TypeError(f"Cannot deserialize torch.Tensor from {type(v).__name__}")
    dtype = _dtype_deserialize(v["dtype"])
    out = torch.tensor(v["data"], dtype=dtype)
    expected_shape = tuple(v["shape"])
    if tuple(out.shape) != expected_shape:
        out = out.reshape(expected_shape)
    return out


# register some serializers that will definitely be used
register_type_serializer(
    torch.dtype,
    serialize=str,
    deserialize=_dtype_deserialize,
)
register_type_serializer(
    torch.device,
    serialize=str,
    deserialize=lambda s: s if isinstance(s, torch.device) else torch.device(s),
)

register_type_serializer(torch.Tensor, _tensor_serialize, _tensor_deserialize)


# ---------------------------------------------------------------------------
# cls_path <-> class resolution
# ---------------------------------------------------------------------------


def _import_cls(cls_path: str) -> type:
    """Import the class identified by a dotted path.

    Resolves a dotted path such as ``"module.submodule.Class"`` or
    ``"module.Outer.Inner"`` into a class object. The module prefix is
    matched greedily: the longest importable prefix of ``cls_path`` is
    used as the module, and the remaining dotted components are resolved
    as attributes via :func:`getattr` (supporting nested classes).

    Parameters
    ----------
    cls_path : str
        Dotted path of the form ``"module.[submodule...].QualName"``.
        ``QualName`` may itself contain dots when the target is a nested
        class (e.g. ``"pkg.mod.Outer.Inner"``).

    Returns
    -------
    type
        The resolved class object.

    Raises
    ------
    ModuleNotFoundError
        No importable module prefix was found in ``cls_path``.
    AttributeError
        A component of the attribute chain after the module does not
        exist on its parent.
    TypeError
        The resolved object is not a class.
    """
    parts = cls_path.split(".")
    # Find the longest dotted prefix that is an importable module. Try
    # increasingly long prefixes left-to-right; stop at the first
    # ModuleNotFoundError and keep the last successful module. Only
    # catch ModuleNotFoundError so genuine import failures inside a
    # real module still propagate.
    module: Any = None
    module_depth = 0
    for i in range(1, len(parts)):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            break
        module_depth = i
    if module is None:
        raise ModuleNotFoundError(
            f"Could not import any module prefix of {cls_path!r}. "
            "Expected a dotted path like 'pkg.mod.Class' or 'pkg.mod.Outer.Inner'."
        )
    obj: Any = module
    for part in parts[module_depth:]:
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"{cls_path!r} resolved to non-class {obj!r}")
    return obj


def _cls_path_of(cls_: type) -> str:
    """Return the canonical dotted path (``module.QualName``) for ``cls_``."""
    return f"{cls_.__module__}.{cls_.__qualname__}"


def _ensure_importable(cls_path: str) -> str:
    """Pydantic validator: ensure the class path is importable without modifying the string.

    We cannot use `_import_cls` directly as a validator because it returns the
    class object, but the `cls_path` field must store the raw string.
    """
    _import_cls(cls_path)
    return cls_path


# ---------------------------------------------------------------------------
# Signature introspection + hashing
# ---------------------------------------------------------------------------


def _signature(cls_: type) -> inspect.Signature:
    """Return the (string-annotation-resolved) signature of ``cls_.__init__``."""
    return inspect.signature(cls_, eval_str=True)


def _hash_init_signature(cls_: type) -> str:
    """Compute a 16-hex-char SHA-256 digest of ``cls_``'s ``__init__`` signature.

    The hash incorporates each parameter's name, kind, annotation, and
    default value, so any visible change to the signature produces a
    different hash. Truncating to 16 hex characters (64 bits) keeps specs
    readable while retaining a negligible collision probability for
    realistic workloads.
    """
    sig = _signature(cls_)
    parts = [
        f"{name}|{p.kind}|{p.annotation!r}|{p.default!r}"
        for name, p in sig.parameters.items()
    ]
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _check_no_positional_only(cls_: type) -> None:
    """Raise :class:`TypeError` if ``cls_.__init__`` has positional-only params."""
    for name, p in _signature(cls_).parameters.items():
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{_cls_path_of(cls_)} has positional-only param {name!r}; "
                "create_model_spec only supports kwargs."
            )


# ---------------------------------------------------------------------------
# BaseSpec
# ---------------------------------------------------------------------------


class BaseSpec(BaseModel):
    """Base class for JSON-serializable, no-pickle hyperparameter specs.

    Concrete spec classes are built dynamically by :func:`create_model_spec`
    via :func:`pydantic.create_model`; each carries one field per
    ``__init__`` kwarg of its target class plus the three metadata fields
    defined here.

    Attributes
    ----------
    cls_path
        Dotted path (``"module.submodule.QualName"``) identifying the target
        class. Validated at assignment time by :func:`_import_cls`.
    timestamp
        ISO-8601 UTC timestamp recording when the spec was created.
    init_hash
        Truncated SHA-256 digest of the target class's ``__init__``
        signature at spec-creation time; see :func:`_hash_init_signature`.

    Notes
    -----
    ``revalidate_instances="never"`` is deliberate: specs are immutable
    records of past state, and revalidating on access would defeat the
    ``init_hash`` provenance guarantee.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        revalidate_instances="never",
    )

    cls_path: Annotated[
        str,
        AfterValidator(_ensure_importable),
        Field(description="Dotted import path of the target class."),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 UTC timestamp of spec creation."),
    ]
    init_hash: Annotated[
        str,
        Field(
            description=("Truncated SHA-256 of the target class's __init__ signature."),
        ),
    ]

    def build(self, *args: Any, strict: bool = False, **extra_kwargs: Any) -> object:
        """Instantiate the target class from the stored hyperparameters.

        Positional ``*args`` and ``**extra_kwargs`` inject runtime-only
        values that cannot be serialized into the spec --- for example,
        ``model.parameters()`` for an optimizer or an ``optimizer`` instance
        for a learning-rate scheduler.

        Before instantiating, the target class's current ``__init__``
        signature is re-hashed and compared to :attr:`init_hash`. By default
        (``strict=False``) a mismatch emits a :class:`UserWarning` but does
        not stop the build, and a subsequent :class:`TypeError` is re-raised
        with the stored/current hashes and the spec timestamp. When
        ``strict=True`` a mismatch raises :class:`ValueError` immediately
        and instantiation is not attempted.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the target class constructor
            (runtime-only, not stored in the spec).
        strict
            If ``True``, raise :class:`ValueError` when the current
            ``__init__`` signature hash does not match :attr:`init_hash`,
            before attempting instantiation. If ``False`` (default), emit
            a :class:`UserWarning` on mismatch and proceed.
        **extra_kwargs
            Extra keyword arguments forwarded to the target class
            constructor, overriding any spec-stored kwargs of the same name.

        Returns
        -------
        object
            A freshly constructed instance of the class at :attr:`cls_path`.

        Raises
        ------
        ValueError
            If ``strict=True`` and the current ``__init__`` signature hash
            does not match :attr:`init_hash`. The message contains both
            hashes, :attr:`cls_path`, and :attr:`timestamp`.
        TypeError
            If the target class cannot be instantiated with the resolved
            kwargs. If the ``init_hash`` does not match the current
            signature, the error message is augmented with the stored and
            current hash values and the spec timestamp.

        Warns
        -----
        UserWarning
            If ``strict=False`` (default) and :attr:`init_hash` does not
            match the hash of the current ``__init__`` signature.
        """
        cls_ = _import_cls(self.cls_path)
        current_hash = _hash_init_signature(cls_)
        hash_mismatch = current_hash != self.init_hash
        if hash_mismatch:
            mismatch_msg = (
                f"init_hash mismatch for {self.cls_path}: "
                f"stored={self.init_hash!r}, current={current_hash!r}. "
                f"The class's __init__ signature has changed since this "
                f"spec was saved (at {self.timestamp})."
            )
            if strict:
                raise ValueError(f"{mismatch_msg} Refusing to build under strict=True.")
            warnings.warn(
                f"{mismatch_msg} Proceeding anyway.",
                UserWarning,
                stacklevel=2,
            )
        sig = _signature(cls_)
        resolved: dict[str, Any] = {}
        for name in type(self).model_fields:
            if name in _META_FIELDS:
                continue
            v = getattr(self, name)
            # Nested spec: build unless target expects the spec itself.
            if isinstance(v, BaseSpec):
                param = sig.parameters.get(name)
                ann = param.annotation if param is not None else None
                wants_spec = isinstance(ann, type) and issubclass(ann, BaseSpec)
                resolved[name] = v if wants_spec else v.build()
            else:
                resolved[name] = v
        resolved.update(extra_kwargs)
        try:
            return cls_(*args, **resolved)
        except TypeError as e:
            if hash_mismatch:
                raise TypeError(
                    f"Failed to build {self.cls_path!r}. Signature hash "
                    f"mismatch: stored={self.init_hash!r}, "
                    f"current={current_hash!r}. The class's __init__ "
                    f"signature has changed since this spec was saved "
                    f"(at {self.timestamp}). Original error: {e}"
                ) from e
            raise


# ---------------------------------------------------------------------------
# Type annotation resolution
# ---------------------------------------------------------------------------


def _resolve_annotation(name: str, value: Any, sig: inspect.Signature) -> Any:
    """Pick the Pydantic field annotation for ``(name, value)`` in ``sig``.

    Order of precedence:

    1. ``value`` is a :class:`BaseSpec` → ``SerializeAsAny[BaseSpec]``
       (preserves the concrete dynamic schema under
       :meth:`~pydantic.BaseModel.model_dump_json`).
    2. The ``__init__`` signature annotates this parameter with a registered
       custom type → wrap via :func:`_wrap_custom_type`.
    3. The ``__init__`` signature has any non-``Any`` annotation → use it.
    4. Otherwise infer from ``type(value)``; if the inferred type is in the
       registry, wrap it; ``None`` values fall back to :class:`typing.Any`.
    """
    if isinstance(value, BaseSpec):
        return SerializeAsAny[BaseSpec]

    param = sig.parameters.get(name)
    sig_ann = param.annotation if param is not None else inspect.Parameter.empty
    has_sig_ann = sig_ann is not inspect.Parameter.empty and sig_ann is not Any

    if has_sig_ann and sig_ann in _TYPE_SERIALIZERS:
        return _wrap_custom_type(sig_ann)
    if has_sig_ann:
        return sig_ann

    vt = type(value)
    if vt in _TYPE_SERIALIZERS:
        return _wrap_custom_type(vt)
    return vt if value is not None else Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model_spec(cls_: type, **kwargs: Any) -> BaseSpec:
    """Build a :class:`BaseSpec` instance for ``cls_`` with the given kwargs.

    A new Pydantic model class is dynamically created via
    :func:`pydantic.create_model`, one field per kwarg, each annotated by
    :func:`_resolve_annotation`. The resulting spec is JSON-serializable
    with :meth:`~pydantic.BaseModel.model_dump_json` and reconstructible
    with :func:`create_model_spec_from_json`.

    Parameters
    ----------
    cls_
        The target class. Must accept all ``**kwargs`` as keyword arguments
        and must not declare any positional-only parameters.
    **kwargs
        Hyperparameters for ``cls_``. Registered types
        (:class:`torch.Tensor`, :class:`torch.dtype`, :class:`torch.device`,
        and any user-registered types) are handled via the type-serializer
        registry. Other values must themselves be JSON-serializable by
        Pydantic.

    Returns
    -------
    BaseSpec
        A dynamically subclassed :class:`BaseSpec` instance named
        ``"{cls_.__name__}Spec"`` with one field per kwarg plus the three
        metadata fields.

    Raises
    ------
    TypeError
        If ``cls_`` has positional-only parameters, or if ``**kwargs``
        contains names absent from the signature while the signature has no
        ``**kwargs`` parameter.

    Examples
    --------
    >>> import torch.nn as nn
    >>> spec = create_model_spec(nn.Linear, in_features=8, out_features=4)
    >>> module = spec.build()
    >>> (module.in_features, module.out_features)
    (8, 4)
    """
    _check_no_positional_only(cls_)
    sig = _signature(cls_)

    unknown = set(kwargs) - set(sig.parameters)
    if unknown:
        var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not var_kw:
            raise TypeError(
                f"Unknown kwargs for {_cls_path_of(cls_)}: {sorted(unknown)}"
            )

    fields: dict[str, tuple[Any, Any]] = {}
    for name, value in kwargs.items():
        annotation = _resolve_annotation(name, value, sig)
        fields[name] = (annotation, value)

    model_cls = create_model(
        f"{cls_.__name__}Spec",
        __base__=BaseSpec,
        **fields,
    )
    return model_cls(
        cls_path=_cls_path_of(cls_),
        timestamp=datetime.now(timezone.utc).isoformat(),
        init_hash=_hash_init_signature(cls_),
        **kwargs,
    )


def create_model_spec_from_json(spec: dict[str, Any]) -> BaseSpec:
    """Rebuild a :class:`BaseSpec` from its JSON-dict form.

    Recursively rehydrates nested specs (detected as values that are
    :class:`dict` and contain a ``"cls_path"`` key). Pydantic's
    :class:`~pydantic.BeforeValidator` hooks on registered types handle the
    str → :class:`torch.dtype` / :class:`torch.device` / dict →
    :class:`torch.Tensor` conversions transparently.

    The original ``timestamp`` and ``init_hash`` are preserved via
    :func:`object.__setattr__` rather than stamped fresh, so that a
    round-tripped spec remains byte-identical (up to JSON-whitespace)
    with its source.

    Parameters
    ----------
    spec
        A :class:`dict` as produced by
        :meth:`~pydantic.BaseModel.model_dump` or by
        :func:`json.loads` on the output of
        :meth:`~pydantic.BaseModel.model_dump_json`.

    Returns
    -------
    BaseSpec
        A spec instance equivalent to the source, with the original
        ``timestamp`` and ``init_hash`` preserved.

    Raises
    ------
    ValueError
        If ``spec`` is missing any of ``cls_path``, ``init_hash``, or
        ``timestamp``, or if ``cls_path`` cannot be imported / resolves to
        a non-class. The underlying exception is preserved as
        ``__cause__``.

    Examples
    --------
    >>> import json, torch.nn as nn
    >>> s = create_model_spec(nn.Linear, in_features=4, out_features=2)
    >>> dumped = json.loads(s.model_dump_json())
    >>> s2 = create_model_spec_from_json(dumped)
    >>> s2.init_hash == s.init_hash
    True
    """
    schema = dict(spec)
    try:
        cls_path = schema.pop("cls_path")
        stored_hash = schema.pop("init_hash")
        stored_timestamp = schema.pop("timestamp")
    except KeyError as e:
        raise ValueError(
            f"Spec JSON missing required field {e.args[0]!r}; "
            f"present keys: {sorted(spec)}"
        ) from e

    try:
        cls_ = _import_cls(cls_path)
    except Exception as e:
        raise ValueError(
            f"Could not resolve cls_path={cls_path!r} while rehydrating spec JSON: {e}"
        ) from e

    kwargs: dict[str, Any] = {}
    for name, value in schema.items():
        if isinstance(value, dict) and "cls_path" in value:
            kwargs[name] = create_model_spec_from_json(value)
        else:
            # Pydantic BeforeValidators on registered types (dtype, device,
            # Tensor) handle the raw value -> typed instance conversion.
            kwargs[name] = value

    rebuilt = create_model_spec(cls_, **kwargs)
    # Preserve original provenance rather than stamping a fresh hash/timestamp.
    object.__setattr__(rebuilt, "timestamp", stored_timestamp)
    object.__setattr__(rebuilt, "init_hash", stored_hash)
    return rebuilt
