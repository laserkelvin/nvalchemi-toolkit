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
"""
Fine-tuning a MACE foundation model
=====================================

This example shows how to fine-tune a pre-trained MACE-MP foundation model
on a small dataset of reference energies and forces, using the full
``nvalchemi`` + PhysicsNeMo training stack.

Key concepts demonstrated
--------------------------
* Loading a :class:`~nvalchemi.models.mace.MACEWrapper` in training mode.
* Configuring :attr:`~nvalchemi.models.base.ModelConfig` to request energies
  and forces during the forward pass.
* Wiring a :class:`~nvalchemi.dynamics.hooks.NeighborListHook` manually for
  static inference outside of a dynamics engine.
* Energy + force-matching loss with a configurable force weight.
* Using PhysicsNeMo utilities for distributed setup, AMP, logging, and
  checkpointing (``nvidia-physicsnemo[datapipes-extras]`` extra required for
  distributed and logging features; core training works without it).
* Exporting the fine-tuned underlying MACE model via
  :meth:`~nvalchemi.models.mace.MACEWrapper.export_model`.

Prerequisites
-------------
Install the MACE and training optional dependencies::

    pip install 'nvalchemi-toolkit[mace,training]'

Then set ``MACE_MODEL_PATH`` to a named MACE-MP checkpoint or a local file::

    export MACE_MODEL_PATH=medium-0b2   # auto-downloads from MACE cache

The example generates synthetic training data internally, so no external
dataset file is required to run it.

Notes on CUDA graphs
--------------------
MACE operates on variable-size graphs (each molecule or snapshot has a
different atom / edge count), so ``use_graphs=True`` in
``StaticCaptureTraining`` is **not** compatible with a standard per-sample
training loop.  This example uses ``use_graphs=False``.  If you batch fixed-
size systems (e.g., replicated unit cells of identical composition), you can
re-enable graph capture for a significant throughput improvement.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PhysicsNeMo imports
# ---------------------------------------------------------------------------
# These are only required when running distributed or when using the advanced
# logging / checkpointing utilities.  Single-GPU training works without them.
try:
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.capture import StaticCaptureTraining
    from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
    from physicsnemo.utils.logging import (
        LaunchLogger,
        PythonLogger,
        RankZeroLoggingWrapper,
    )

    _PHYSICSNEMO_AVAILABLE = True
except ImportError:
    _PHYSICSNEMO_AVAILABLE = False
    logger.warning(
        "nvidia-physicsnemo not found — running without distributed support, "
        "PhysicsNeMo logging, or advanced checkpointing.  "
        "Install with: pip install 'nvalchemi-toolkit[training]'"
    )

# %%
# Distributed setup
# ------------------
# ``DistributedManager`` auto-detects the launch environment (torchrun /
# SLURM / OpenMPI) and initialises the default process group.  On a single
# GPU this is a no-op.

if _PHYSICSNEMO_AVAILABLE:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    is_rank_zero = dist.rank == 0
    LaunchLogger.initialize()
    _logger = RankZeroLoggingWrapper(PythonLogger("mace_finetune"), dist)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_rank_zero = True
    _logger = logger  # type: ignore[assignment]

# %%
# Load the MACE model
# --------------------
# ``MACEWrapper.from_checkpoint`` accepts a local ``.pt`` path or a named
# MACE-MP checkpoint string (e.g. ``"medium-0b2"``).  The model is loaded
# in **eval** mode by default; we switch to **train** mode after loading so
# that dropout and batch-norm layers (if any) behave correctly during
# fine-tuning.

from nvalchemi.models.mace import MACEWrapper  # noqa: E402 (after dist setup)

MACE_MODEL_PATH = os.environ.get("MACE_MODEL_PATH", "medium-0b2")

model = MACEWrapper.from_checkpoint(
    checkpoint_path=MACE_MODEL_PATH,
    device=device,
    dtype=torch.float32,  # float32 recommended for fine-tuning stability
    enable_cueq=False,  # set True for GPU speed-up (requires cuequivariance)
    compile_model=False,  # torch.compile disables gradients; skip for fine-tuning
)
model.train()

# Tell the wrapper to compute energies AND forces on every forward pass.
model.model_config.compute_energies = True
model.model_config.compute_forces = True
model.model_config.compute_stresses = False

# %%
# Wrap for DDP (multi-GPU)
# -------------------------
if _PHYSICSNEMO_AVAILABLE and dist.world_size > 1:
    from torch.nn.parallel import DistributedDataParallel

    ddp_stream = torch.cuda.Stream()
    with torch.cuda.stream(ddp_stream):
        model = DistributedDataParallel(  # type: ignore[assignment]
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

# %%
# Build the neighbor-list hook
# -----------------------------
# Outside of a dynamics engine the neighbor list must be applied manually
# before each forward pass.  We read the required configuration directly
# from the model card so this code generalises to any BaseModelMixin.

from nvalchemi.dynamics.hooks import NeighborListHook  # noqa: E402

_card = model.model_card if not hasattr(model, "module") else model.module.model_card
nl_hook = NeighborListHook(_card.neighbor_config)

# %%
# Generate synthetic training data
# ----------------------------------
# In a real workflow you would load reference DFT data from an extended XYZ
# file, an ASE database, or a Zarr store via AtomicDataZarrReader + Dataset.
# Here we generate random water clusters so the example is self-contained.

from nvalchemi.data import AtomicData, Batch  # noqa: E402


def _make_water_cluster(n_molecules: int, seed: int = 0) -> AtomicData:
    """Synthetic H₂O cluster with random reference energies and forces."""
    torch.manual_seed(seed)
    n_atoms = n_molecules * 3
    o_h_bond = 0.96
    half_angle = math.radians(104.5 / 2)
    positions_list = []
    atomic_numbers_list = []
    for i in range(n_molecules):
        ox = float(i) * 3.5
        o_pos = torch.tensor([ox, 0.0, 0.0])
        h1_pos = o_pos + o_h_bond * torch.tensor(
            [math.sin(half_angle), 0.0, math.cos(half_angle)]
        )
        h2_pos = o_pos + o_h_bond * torch.tensor(
            [-math.sin(half_angle), 0.0, math.cos(half_angle)]
        )
        positions_list.extend([o_pos, h1_pos, h2_pos])
        atomic_numbers_list.extend([8, 1, 1])
    positions = torch.stack(positions_list).float()
    positions += 0.05 * torch.randn_like(positions)
    # Synthetic DFT reference values (not physical — for demonstration only)
    ref_energy = torch.randn(1, 1) * 0.1 - 76.0 * n_molecules  # eV
    ref_forces = torch.randn(n_atoms, 3) * 0.05  # eV/Å
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.tensor(atomic_numbers_list, dtype=torch.long),
        forces=ref_forces,
        energies=ref_energy,
    )


N_TRAIN = 16
N_VAL = 4
torch.manual_seed(42)
train_data = [
    _make_water_cluster(n_molecules=i % 3 + 1, seed=i) for i in range(N_TRAIN)
]
val_data = [
    _make_water_cluster(n_molecules=i % 3 + 1, seed=100 + i) for i in range(N_VAL)
]

# %%
# Training configuration
# -----------------------

BATCH_SIZE = 4
MAX_EPOCHS = 5
LR = 1e-4
FORCE_WEIGHT = 10.0  # relative weight of force loss vs energy loss
CHECKPOINT_DIR = Path(tempfile.mkdtemp()) / "mace_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Optimizer and scheduler
# ------------------------
# We apply a lower learning rate to the MACE backbone and a higher one to
# any task-specific head layers you may add.  Here all parameters share the
# same rate for simplicity.

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# %%
# Restore from checkpoint if one exists
# ---------------------------------------
if _PHYSICSNEMO_AVAILABLE:
    # Returns the last saved epoch, or 0 if no checkpoint found.
    start_epoch = load_checkpoint(
        str(CHECKPOINT_DIR),
        models=[model],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
else:
    start_epoch = 0

# %%
# Define the training step
# -------------------------
# We wrap the forward + loss computation so it can optionally be decorated
# with StaticCaptureTraining.  Note ``use_graphs=False`` because MACE
# processes variable-size graphs (see module docstring).


def _compute_loss(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass + energy/force matching loss.

    Returns
    -------
    loss : scalar total loss
    e_loss : energy component (eV² per atom)
    f_loss : force component (eV²/Å² per atom)
    """
    # Build neighbor list in-place (populates batch.edge_index, batch.unit_shifts)
    nl_hook(batch, dynamics=None)  # type: ignore[arg-type]

    outputs = model(batch)

    pred_e = outputs["energies"]  # [B, 1]
    ref_e = batch.energies  # [B, 1]

    # Per-atom energy normalisation avoids bias toward larger systems.
    n_atoms = batch.num_nodes_per_graph.float().unsqueeze(-1)  # [B, 1]
    e_loss = ((pred_e - ref_e) / n_atoms).pow(2).mean()

    f_loss = torch.tensor(0.0, device=device)
    if "forces" in outputs and outputs["forces"] is not None:
        pred_f = outputs["forces"]  # [N, 3]
        ref_f = batch.forces  # [N, 3]
        f_loss = (pred_f - ref_f).pow(2).mean()

    loss = e_loss + FORCE_WEIGHT * f_loss
    return loss, e_loss, f_loss


if _PHYSICSNEMO_AVAILABLE:

    @StaticCaptureTraining(
        model,
        optim=optimizer,
        use_graphs=False,  # variable graph sizes — see docstring
        use_autocast=True,
        gradient_clip_norm=1.0,
    )
    def train_step(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _compute_loss(batch)
else:

    def train_step(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[misc]
        optimizer.zero_grad()
        loss, e_loss, f_loss = _compute_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss, e_loss, f_loss


# %%
# Training loop
# --------------


def _make_batches(data_list: list[AtomicData], batch_size: int) -> list[Batch]:
    """Collate a list of AtomicData into fixed-size Batch objects."""
    batches = []
    for i in range(0, len(data_list), batch_size):
        chunk = data_list[i : i + batch_size]
        batches.append(Batch.from_data_list(chunk, device=device))
    return batches


train_batches = _make_batches(train_data, BATCH_SIZE)
val_batches = _make_batches(val_data, BATCH_SIZE)

for epoch in range(start_epoch, MAX_EPOCHS):
    # ---- training ----
    model.train()
    if _PHYSICSNEMO_AVAILABLE:
        ctx = LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_batches))
    else:
        import contextlib

        ctx = contextlib.nullcontext()  # type: ignore[assignment]

    with ctx as log:
        for batch in train_batches:
            loss, e_loss, f_loss = train_step(batch)
            if _PHYSICSNEMO_AVAILABLE and log is not None:
                log.log_minibatch(
                    {
                        "loss": loss.item(),
                        "e_loss": e_loss.item(),
                        "f_loss": f_loss.item(),
                    }
                )
            else:
                logger.info(
                    f"epoch {epoch}  loss={loss.item():.4f}  "
                    f"e={e_loss.item():.4f}  f={f_loss.item():.4f}"
                )

    scheduler.step()

    # ---- validation ----
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_batches:
            nl_hook(batch, dynamics=None)  # type: ignore[arg-type]
            outputs = model(batch)
            pred_e = outputs["energies"]
            ref_e = batch.energies
            n_atoms = batch.num_nodes_per_graph.float().unsqueeze(-1)
            val_loss_total += ((pred_e - ref_e) / n_atoms).pow(2).mean().item()

    val_loss_mean = val_loss_total / len(val_batches)

    if _PHYSICSNEMO_AVAILABLE:
        with LaunchLogger("val", epoch=epoch) as log:
            log.log_minibatch({"val_e_loss": val_loss_mean})
    else:
        logger.info(f"epoch {epoch}  val_e_loss={val_loss_mean:.4f}")

    # ---- checkpoint (rank 0 only) ----
    if is_rank_zero:
        if _PHYSICSNEMO_AVAILABLE:
            save_checkpoint(
                str(CHECKPOINT_DIR),
                models=[model],
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metadata={"val_e_loss": val_loss_mean},
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                CHECKPOINT_DIR / f"checkpoint_{epoch:03d}.pt",
            )

# %%
# Export the fine-tuned model
# ----------------------------
# ``export_model`` saves the underlying MACE ``nn.Module`` without the
# nvalchemi wrapper, so it can be reloaded with the standard MACE / ASE
# interface.  To keep the wrapper use ``torch.save(model, path)`` instead.

if is_rank_zero:
    export_path = CHECKPOINT_DIR / "mace_finetuned.pt"
    # Unwrap DDP if present
    _model_to_export = model.module if hasattr(model, "module") else model
    _model_to_export.export_model(export_path, as_state_dict=False)
    logger.info(f"Fine-tuned MACE model exported to: {export_path}")

# %%
# Cleanup
# --------
if _PHYSICSNEMO_AVAILABLE:
    dist.cleanup(barrier=True)

logger.info("Done.")
