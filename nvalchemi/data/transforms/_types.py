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
"""Type aliases for the :mod:`nvalchemi.data.transforms` package.

These aliases describe the call signatures that user-supplied transforms
must satisfy when plugged into :class:`~nvalchemi.data.Dataset` and
:class:`~nvalchemi.data.DataLoader`.

Return-value contract
---------------------
Transforms must *return* their (possibly mutated) output. In-place
mutation without returning is not supported: the composition helper
(:class:`~nvalchemi.data.transforms.Compose`) rebinds its running
state from each transform's return value on every step. A transform
that mutates ``metadata`` in place and returns ``None`` will fail when
the next step tries to unpack the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeAlias

if TYPE_CHECKING:
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch


SampleTransform: TypeAlias = Callable[
    ["AtomicData", dict[str, Any]],
    tuple["AtomicData", dict[str, Any]],
]
"""Per-sample transform callable.

A ``SampleTransform`` receives a single :class:`AtomicData` together with
its metadata ``dict`` and returns a (possibly modified) pair of the same
types. Per-sample transforms are intended to be applied inside
:class:`~nvalchemi.data.Dataset` after device transfer.
"""


BatchTransform: TypeAlias = Callable[["Batch"], "Batch"]
"""Per-batch transform callable.

A ``BatchTransform`` receives a fully collated :class:`Batch` and returns
a (possibly modified) :class:`Batch`. Per-batch transforms are intended
to be applied inside :class:`~nvalchemi.data.DataLoader` after
:meth:`Batch.from_data_list` and before the batch is yielded.
"""
