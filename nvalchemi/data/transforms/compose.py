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
"""Left-to-right composition helper for user-supplied data transforms.

A single polymorphic :class:`Compose` class handles both per-sample
transforms (which take and return ``(AtomicData, metadata)``) and
per-batch transforms (which take and return :class:`Batch`). Arity is
inferred from what the transform returns: a tuple return is unpacked
into the next transform's positional arguments; any other value is
passed as a single argument.

Transforms must *return* their (possibly mutated) output; in-place
mutation without returning is not supported.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


class Compose:
    """Compose a sequence of transforms into a single callable.

    ``Compose`` is deliberately polymorphic: it threads whatever it is
    given through each transform, unpacking tuple returns between steps
    so a transform returning ``(data, metadata)`` feeds the next
    transform's ``(data, metadata)`` arguments. A transform returning a
    non-tuple value (such as a :class:`Batch`) is forwarded as a single
    positional argument to the next step.

    An empty sequence behaves as the identity.

    Parameters
    ----------
    transforms : Sequence[Callable[..., Any]]
        Transforms to compose in application order. Any iterable is
        accepted and eagerly materialized into an immutable
        :class:`tuple`. Each entry must be callable.

    Attributes
    ----------
    transforms : tuple[Callable[..., Any], ...]
        The composed transforms, stored in application order.

    Raises
    ------
    TypeError
        If any entry in ``transforms`` is not callable.
    RuntimeError
        (At call time) If any transform raises, the original exception
        is wrapped in a :class:`RuntimeError` that identifies the
        failing index. The original exception is preserved via
        ``__cause__``.

    Notes
    -----
    **Arity-preservation rule.** Every transform in a chain must return
    the same shape as its input: tuple-in → tuple-out, single-value-in →
    single-value-out. A per-sample transform receiving ``(data,
    metadata)`` must return a ``(data, metadata)`` tuple; a per-batch
    transform receiving ``batch`` must return a :class:`Batch`.

    **Return-value contract.** Transforms must *return* their output.
    Mutating ``metadata`` in place and returning ``None`` will fail the
    next step, since :class:`Compose` rebinds its running state from
    each return value.

    **Call-site semantics.** ``Compose(...)`` inspects each return
    value: a :class:`tuple` becomes the next step's ``*args``; anything
    else is wrapped in a 1-tuple. After the last step, a single running
    argument is returned raw (so per-batch callers get a :class:`Batch`
    back, not ``(batch,)``); multiple running arguments are returned as
    a tuple (so per-sample callers get ``(data, metadata)`` back).

    Examples
    --------
    Per-sample composition (2-arg input, tuple output):

    >>> def shift_positions(data, metadata):  # doctest: +SKIP
    ...     new_data = data.replace(positions=data.positions + 1.0)
    ...     return new_data, metadata
    >>> def tag_origin(data, metadata):  # doctest: +SKIP
    ...     metadata["origin"] = "shifted"
    ...     return data, metadata
    >>> compose = Compose([shift_positions, tag_origin])  # doctest: +SKIP
    >>> new_data, new_meta = compose(data, {})  # doctest: +SKIP

    Per-batch composition (1-arg input, batch output):

    >>> def scale_positions(batch):  # doctest: +SKIP
    ...     return batch.replace(positions=batch.positions * 2.0)
    >>> compose = Compose([scale_positions])  # doctest: +SKIP
    >>> new_batch = compose(batch)  # doctest: +SKIP
    """

    __slots__ = ("transforms",)

    def __init__(self, transforms: Sequence[Callable[..., Any]]) -> None:
        self.transforms: tuple[Callable[..., Any], ...] = tuple(transforms)
        for i, t in enumerate(self.transforms):
            if not callable(t):
                raise TypeError(
                    f"Compose: transform at index {i} is not callable: "
                    f"got {type(t).__name__}"
                )

    def __call__(self, *args: Any) -> Any:
        """Apply the composed transforms to ``*args`` left-to-right.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the first transform.
            Typically one of:

            - ``(data, metadata)`` for a per-sample chain, or
            - ``(batch,)`` for a per-batch chain.

        Returns
        -------
        Any
            The value returned by the final transform. A single running
            argument is returned raw; multiple running arguments are
            returned as a :class:`tuple`.

        Raises
        ------
        RuntimeError
            If any transform raises; the error message identifies the
            failing index and the original exception is attached via
            ``__cause__``.
        """
        for i, transform in enumerate(self.transforms):
            try:
                result = transform(*args)
            except Exception as e:
                raise RuntimeError(
                    f"Compose: transform[{i}] ({transform!r}) raised {type(e).__name__}"
                ) from e
            args = result if isinstance(result, tuple) else (result,)
        if len(args) == 1:
            return args[0]
        return args

    def __repr__(self) -> str:
        """Return a developer-readable representation listing the transforms."""
        return f"Compose({list(self.transforms)!r})"
