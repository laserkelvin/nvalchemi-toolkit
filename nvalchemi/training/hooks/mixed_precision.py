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
"""Mixed-precision training hook driving ``torch.amp.autocast`` and ``GradScaler``.

See :class:`MixedPrecisionHook` for the user-facing API. The hook claims
:attr:`TrainingStage.DO_BACKWARD` and :attr:`TrainingStage.DO_OPTIMIZER_STEP`
so that :class:`~nvalchemi.training.strategy.TrainingStrategy` remains free
of any AMP-specific code.
"""

from __future__ import annotations

from types import TracebackType
from typing import Annotated, Literal

import torch
from plum import Function, dispatch
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    PrivateAttr,
)

from nvalchemi.hooks._context import HookContext
from nvalchemi.training._spec import _dtype_deserialize
from nvalchemi.training._stages import TrainingStage

__all__ = ["MixedPrecisionHook"]


_SUPPORTED_PRECISIONS: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)
"""Autocast dtypes this hook understands (fp32 is a no-op, fp16 enables the scaler)."""


def _restrict_precision(value: torch.dtype) -> torch.dtype:
    """Reject dtypes outside :data:`_SUPPORTED_PRECISIONS`."""
    if value not in _SUPPORTED_PRECISIONS:
        supported = ", ".join(str(d) for d in _SUPPORTED_PRECISIONS)
        raise ValueError(
            f"MixedPrecisionHook.precision must be one of ({supported}); got {value!r}."
        )
    return value


Precision = Annotated[
    torch.dtype,
    BeforeValidator(_dtype_deserialize),
    AfterValidator(_restrict_precision),
    PlainSerializer(str),
]
"""``torch.dtype`` field accepting canonical names (``"float16"``) or dtype objects."""


class MixedPrecisionHook(BaseModel):
    """Automatic-mixed-precision hook driving autocast and ``GradScaler``.

    The hook claims :attr:`TrainingStage.DO_BACKWARD` and
    :attr:`TrainingStage.DO_OPTIMIZER_STEP` through
    :meth:`_runs_on_stage` so
    :class:`~nvalchemi.training.strategy.TrainingStrategy` delegates the
    backward pass and optimizer/scheduler stepping to this hook. The first
    :attr:`TrainingStage.BEFORE_FORWARD` lazily constructs the autocast
    region and :class:`torch.amp.GradScaler` on the workflow's primary
    device (``ctx.workflow.devices[0]``), so the hook need not know the
    device at construction time. The autocast region is released in
    :meth:`__exit__` which ``TrainingStrategy`` invokes after training.

    Precision modes:

    * :data:`torch.float32` — autocast is ``enabled=False`` and the scaler
      is disabled; the hook is a functional no-op aside from calling the
      default ``backward()``/``step()`` path on the live context.
    * :data:`torch.bfloat16` — autocast casts eligible ops to ``bfloat16``.
      No gradient scaling because bf16's exponent range matches fp32.
    * :data:`torch.float16` — autocast casts eligible ops to ``float16``
      and the scaler scales the loss before backward, unscales gradients
      before observers in ``AFTER_BACKWARD`` see them, and skips optimizer
      steps that would otherwise consume ``inf``/``nan`` gradients.

    Parameters
    ----------
    precision : torch.dtype, optional
        Autocast dtype and scaler policy. Accepts either a
        :class:`torch.dtype` (e.g. ``torch.float16``) or the canonical
        string name (``"float32"``, ``"bfloat16"``, ``"float16"``).
        Default :data:`torch.float32` (no-op).

    Attributes
    ----------
    precision : torch.dtype
        Active autocast dtype.
    frequency : int
        Hook Protocol attribute; fixed at ``1`` because mixed precision
        must act on every step or not at all.
    stage : TrainingStage | None
        Hook Protocol attribute; always ``None`` because dispatch is
        driven by :meth:`_runs_on_stage`.

    Raises
    ------
    pydantic.ValidationError
        If ``precision`` is not one of the supported dtypes.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.hooks import MixedPrecisionHook
    >>> MixedPrecisionHook(precision=torch.bfloat16).precision
    torch.bfloat16
    >>> MixedPrecisionHook(precision="float16").precision
    torch.float16

    Notes
    -----
    * When multiple optimizers are configured, every optimizer in
      ``ctx.optimizers`` is unscaled and stepped in list order, and each
      scheduler in ``ctx.lr_schedulers`` advances only when its paired
      optimizer step was not skipped by the scaler.
    * Skip detection uses ``GradScaler._found_inf_per_device`` (private
      but canonical) wrapped in a ``try``/``except``; a ``get_scale``
      pre/post comparison is used as a fallback if the API changes.
    * Under ``precision=torch.float16`` on CPU (where the scaler is
      effectively a no-op) no warning is emitted and no exception is
      raised — the hook still drives ``backward()`` and ``step()``
      through the disabled scaler.
    """

    precision: Precision

    # Hook Protocol attributes.
    frequency: int = 1
    stage: TrainingStage | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="allow",
        # ``plum.Function`` descriptors stored by @dispatch on the class are
        # neither fields nor ``ClassVar``; mark them as ignored so Pydantic
        # does not demand annotations for them.
        ignored_types=(Function,),
    )

    _autocast_ctx: torch.amp.autocast | None = PrivateAttr(default=None)
    _scaler: torch.amp.GradScaler | None = PrivateAttr(default=None)
    _active: bool = PrivateAttr(default=False)

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return ``True`` for the three stages this hook drives.

        Parameters
        ----------
        stage : TrainingStage
            Stage being dispatched by the registry.

        Returns
        -------
        bool
            ``True`` when ``stage`` is :attr:`~TrainingStage.BEFORE_FORWARD`,
            :attr:`~TrainingStage.DO_BACKWARD`, or
            :attr:`~TrainingStage.DO_OPTIMIZER_STEP`; ``False`` otherwise.
        """
        return stage in (
            TrainingStage.BEFORE_FORWARD,
            TrainingStage.DO_BACKWARD,
            TrainingStage.DO_OPTIMIZER_STEP,
        )

    def __enter__(self) -> MixedPrecisionHook:
        """Enter the hook's context; lazy-init is deferred to ``BEFORE_FORWARD``.

        Returns
        -------
        MixedPrecisionHook
            This hook instance, for ``with`` expressions.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the autocast region and reset internal state for reuse.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class raised inside the managed block, if any.
        exc : BaseException | None
            Exception instance raised inside the managed block, if any.
        tb : TracebackType | None
            Traceback associated with ``exc``, if any.
        """
        if self._active and self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc, tb)
        self._autocast_ctx = None
        self._scaler = None
        self._active = False

    # ------------------------------------------------------------------
    # Stage dispatch — ``@plum.dispatch`` overloads keyed on stage Literal
    # ------------------------------------------------------------------

    @dispatch
    def __call__(  # noqa: F811
        self, ctx: HookContext, stage: Literal[TrainingStage.BEFORE_FORWARD]
    ) -> None:
        """Lazily construct and enter the autocast region on first forward."""
        # TODO: lazy-init lives here because the hook has no workflow reference
        # at ``__enter__`` time. A future ``Hook.register(self, workflow)``
        # protocol addition would let autocast move into ``__enter__`` and
        # drop this guard.
        if self._autocast_ctx is not None:
            return
        device_type = ctx.workflow.devices[0].type
        enabled = self.precision != torch.float32
        self._autocast_ctx = torch.amp.autocast(
            device_type=device_type, dtype=self.precision, enabled=enabled
        )
        self._scaler = torch.amp.GradScaler(
            device=device_type, enabled=(self.precision == torch.float16)
        )
        self._autocast_ctx.__enter__()
        self._active = True

    @dispatch
    def __call__(  # noqa: F811
        self, ctx: HookContext, stage: Literal[TrainingStage.DO_BACKWARD]
    ) -> None:
        """Scale and run the backward pass; unscale grads for fp16 observers."""
        if self._scaler is None:
            raise RuntimeError(
                "MixedPrecisionHook: DO_BACKWARD fired before BEFORE_FORWARD; "
                "strategy invariant violated."
            )
        scaled_loss = self._scaler.scale(ctx.loss)
        scaled_loss.backward()
        if self.precision == torch.float16:
            # Unscale in-place so AFTER_BACKWARD observers see true gradients.
            for opt in ctx.optimizers:
                self._scaler.unscale_(opt)

    @dispatch
    def __call__(  # noqa: F811
        self, ctx: HookContext, stage: Literal[TrainingStage.DO_OPTIMIZER_STEP]
    ) -> None:
        """Step each optimizer (gated by the scaler) and its scheduler.

        Each optimizer is stepped through the scaler; when the scaler
        signals an ``inf``/``nan`` gradient it suppresses the step
        internally and the paired scheduler is not advanced. The scaler's
        ``update()`` is called once after all optimizers have stepped.
        """
        if self._scaler is None:
            raise RuntimeError(
                "MixedPrecisionHook: DO_OPTIMIZER_STEP fired before BEFORE_FORWARD; "
                "strategy invariant violated."
            )
        skipped_flags: list[bool | None] = []
        for opt in ctx.optimizers:
            self._scaler.step(opt)
            skipped_flags.append(self._opt_step_skipped(opt))
        # Fallback attribution is needed only when ``_found_inf_per_device``
        # was unavailable for at least one optimizer. Avoid two unconditional
        # ``get_scale()`` host reads (a CUDA sync point) on the common path
        # where every flag is a concrete ``bool``.
        need_fallback = self.precision == torch.float16 and any(
            flag is None for flag in skipped_flags
        )
        pre_scale = self._scaler.get_scale() if need_fallback else 0.0
        self._scaler.update()
        # A post-update scale drop means at least one optimizer saw inf/nan.
        # Without per-optimizer attribution we conservatively skip every
        # scheduler whose flag is ``None`` this step.
        fallback_skipped = need_fallback and self._scaler.get_scale() < pre_scale
        for sched, skipped in zip(ctx.lr_schedulers, skipped_flags, strict=True):
            if sched is None:
                continue
            if skipped or (fallback_skipped and skipped is None):
                continue
            sched.step()

    @dispatch
    def __call__(  # noqa: F811
        self,
        ctx: HookContext,  # noqa: ARG002
        stage: TrainingStage,  # noqa: ARG002
    ) -> None:
        """No-op fallback for stages this hook does not claim.

        Parameters
        ----------
        ctx : HookContext
            Per-batch context; unused here.
        stage : TrainingStage
            Any stage outside :meth:`_runs_on_stage`.
        """
        return

    def _opt_step_skipped(self, opt: torch.optim.Optimizer) -> bool | None:
        """Return whether the scaler suppressed ``opt``'s most recent step.

        Parameters
        ----------
        opt : torch.optim.Optimizer

        Returns
        -------
        bool | None
            ``True`` if the scaler found ``inf``/``nan`` gradients and
            suppressed the step; ``False`` if the step was applied;
            ``None`` when the private ``_found_inf_per_device`` API is
            unavailable (callers fall back to the post-update scale
            comparison). For ``precision != torch.float16`` the scaler is
            disabled so this method always returns ``False``.
        """
        if self._scaler is None:
            raise RuntimeError("MixedPrecisionHook: scaler not initialized.")
        if self.precision != torch.float16:
            return False
        try:
            found_inf = self._scaler._found_inf_per_device(opt)
        except Exception:
            return None
        return any(bool(v.item()) for v in found_inf.values())
