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
"""Hook registry mixin for workflow engines."""

from __future__ import annotations

from enum import Enum

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks._protocol import Hook


class HookRegistryMixin:
    """Mixin providing flat-list hook storage and dispatch.

    The host class must provide a ``step_count`` attribute. Override
    ``_build_context`` to populate workflow-specific fields.

    Attributes
    ----------
    hooks : list[Hook]
        Flat list of registered hooks.
    step_count : int
        Current step number (must be provided by the engine).
    """

    hooks: list[Hook]
    step_count: int

    def _init_hooks(self, hooks: list[Hook] | None = None) -> None:
        """Initialize hook storage and register provided hooks.

        Parameters
        ----------
        hooks : list[Hook] | None
            Optional list of hooks to register.
        """
        self.hooks = []
        if hooks:
            for hook in hooks:
                self.register_hook(hook)

    def register_hook(self, hook: Hook) -> None:
        """Register a hook.

        Parameters
        ----------
        hook : Hook
            Hook to register.

        Raises
        ------
        ValueError
            If ``hook.frequency`` is not a positive integer.
        """
        if not isinstance(hook.frequency, int) or hook.frequency < 1:
            raise ValueError(
                f"Hook frequency must be a positive integer, got {hook.frequency}"
            )
        self.hooks.append(hook)

    def _build_context(self, batch: Batch) -> HookContext:
        """Build a HookContext for the current state.

        Override in subclasses to populate workflow-specific fields.

        Parameters
        ----------
        batch : Batch
            Current batch being processed.

        Returns
        -------
        HookContext
            Context object for hooks.
        """
        return HookContext(
            batch=batch,
            step_count=self.step_count,
            model=getattr(self, "model", None),
        )

    def _call_hooks(self, stage: Enum, batch: Batch) -> None:
        """Call hooks whose :meth:`~Hook._runs_on_stage` returns ``True``.

        Hooks that do not define ``_runs_on_stage`` are filtered by the
        default check ``stage == hook.stage``.

        Parameters
        ----------
        stage : Enum
            Current workflow stage.
        batch : Batch
            Current batch being processed.
        """
        ctx = self._build_context(batch)
        for hook in self.hooks:
            runs_on_stage = getattr(hook, "_runs_on_stage", None)
            if runs_on_stage is not None:
                if not runs_on_stage(stage):
                    continue
            elif stage != hook.stage:
                continue
            hook(ctx, stage)
