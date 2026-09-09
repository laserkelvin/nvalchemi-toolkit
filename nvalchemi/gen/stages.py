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
"""Lifecycle stages of the generative pipeline at which hooks can fire."""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["GenerationStage"]


class GenerationStage(Enum):
    """Stages of the :class:`~nvalchemi.gen.generator.AtomGenerator` pipeline.

    One stage per distinct point of the fixed pipeline (condition →
    generate → materialize; the raw sample is exposed to hooks between
    generation and its materialization into a
    :class:`~nvalchemi.data.Batch`). Before/after pairs collapse into single
    stages because hooks mutate the
    :class:`~nvalchemi.hooks.GenerationContext` by replacing its fields,
    and the next step re-reads the context — so a "before generate" hook and
    an "after condition" hook are the same point.
    Attributes
    ----------
    BEFORE_CONDITION
        Fired before the conditioning batch is built; ``ctx.batch`` is
        ``None``. Edit or replace ``ctx.cond`` here. (Text encodings are
        expected to happen before the ``AtomGenerator`` is entered, so ``cond``
        already holds tensor data.)
    AFTER_CONDITION
        Fired after conditioning; ``ctx.batch`` holds the conditioning
        batch, tiled by ``num_samples_per_batch``. Attach conditioning
        metadata (e.g. text embeddings for classifier-free guidance) or
        replace the conditioning batch here.
    AFTER_SAMPLE
        Fired after the generating function returns and before the raw sample
        is materialized into a :class:`~nvalchemi.data.Batch`;
        ``ctx.sample`` holds the sample in whatever container the generating
        function produced (a :class:`~tensordict.TensorDict` for
        tensor-native families, otherwise any container the materialization
        callable understands), and ``ctx.batch`` still holds the conditioning
        batch. Filter or replace ``ctx.sample`` here — workflows with compact
        internal representations can drop rejected candidates before paying
        materialization cost. The driver re-reads ``ctx.sample`` after
        dispatch and hands it to the materialization callable.
    AFTER_GENERATE
        Fired after the raw sample has been materialized into the generated
        :class:`~nvalchemi.data.Batch`; ``ctx.batch`` holds it. Filter or
        mutate the generated batch here — filtering is graph-level
        subsetting (``ctx.batch = ctx.batch[keep]``). The materialized batch
        may already be zero-graph (a materialization callable may signal
        total rejection via :meth:`~nvalchemi.data.Batch.empty`), so filters
        should tolerate ``num_graphs == 0``. Zero-graph *selections* still
        raise ``IndexError`` — a hook signalling total rejection replaces
        ``ctx.batch`` with an explicitly built empty batch rather than
        subsetting to nothing.
    """

    BEFORE_CONDITION = auto()
    AFTER_CONDITION = auto()
    AFTER_SAMPLE = auto()
    AFTER_GENERATE = auto()
