<!-- markdownlint-disable MD014 -->

(losses_guide)=

# Losses

Loss functions in ALCHEMI are tensor-first, composable
{py:class}`torch.nn.Module` objects. A **leaf loss** consumes a
prediction tensor and a target tensor and returns a scalar; a
{py:class}`~nvalchemi.training.ComposedLossFunction` *routes* keyed
mappings of predictions and targets into each leaf and returns a
`total_loss` plus per-component contributions.

This page covers:

- the built-in leaf losses and how to call them directly;
- {py:class}`~nvalchemi.training.ComposedLossFunction` for multi-task
  training;
- loss-weight scheduling via the
  {py:class}`~nvalchemi.training.LossWeightSchedule` protocol;
- how to write your own loss — first a pure tensor-to-tensor loss,
  then a metadata-aware one.

```{tip}
Loss terms never read from {py:class}`~nvalchemi.data.Batch`. They take
plain tensors and optional `**kwargs` for graph metadata. Assembling
predictions, targets, and metadata into a loss call is the job of the
training loop (or `TrainingStrategy`), not of the loss.
```

## Built-in losses

The three provided losses cover the standard MLIP training targets.
Each is a {py:class}`torch.nn.Module` with an MSE-style `_forward`,
configurable `target_key` / `prediction_key` attributes used by
composition, and an opt-in `ignore_nan` flag for batches with
missing labels.

| Class | Target | Key defaults | Extra knobs |
|-------|--------|--------------|-------------|
| {py:class}`~nvalchemi.training.EnergyLoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | `per_atom` normalization, `ignore_nan` |
| {py:class}`~nvalchemi.training.ForceLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | `normalize_by_atom_count`, `ignore_nan` |
| {py:class}`~nvalchemi.training.StressLoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | `ignore_nan` |

### Calling a leaf loss directly

A leaf loss is a plain `nn.Module`: call it with `(pred, target)` and
it returns a scalar. Schedule-aware behavior is the same — if `weight`
is `None`, the returned tensor equals the unweighted `_forward` output:

```python
import torch
from nvalchemi.training import EnergyLoss

loss_fn = EnergyLoss()
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

loss = loss_fn(pred, target)         # scalar Tensor
loss.backward()
```

Concrete losses may require graph metadata as keyword arguments. For
example, `ForceLoss` with the default graph-balanced normalization
needs `batch_idx` and `num_graphs` for dense `(V, 3)` forces:

```python
from nvalchemi.training import ForceLoss

force_fn = ForceLoss()                         # normalize_by_atom_count=True

pred = torch.randn(10, 3, requires_grad=True)
target = torch.randn(10, 3)
batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

loss = force_fn(pred, target, batch_idx=batch_idx, num_graphs=3)
```

The same loss accepts a padded `(B, V_max, 3)` layout with per-graph
counts instead:

```python
pred_padded = torch.randn(3, 4, 3, requires_grad=True)
target_padded = torch.randn(3, 4, 3)
counts = torch.tensor([3, 4, 3])

loss = force_fn(pred_padded, target_padded, num_nodes_per_graph=counts)
```

Any leaf loss accepts `step=` and `epoch=` keyword arguments; they
matter only when a weight schedule is attached (see
[Scheduling weights](scheduling_weights)).

### Ignoring missing labels with `ignore_nan`

Every built-in loss has an `ignore_nan=False` flag. When `True`, target
entries equal to `NaN` contribute zero to both the loss value and the
gradient — a "nanmean"-style reduction implemented with branch-free
tensor ops so it stays `torch.compile`-safe:

```python
energy_loss = EnergyLoss(ignore_nan=True)

target = torch.tensor([[1.0], [float("nan")], [3.0]])
pred = torch.zeros_like(target, requires_grad=True)

loss = energy_loss(pred, target)
loss.backward()

assert torch.isfinite(loss)
assert pred.grad[1].item() == 0.0   # masked row has zero gradient
```

`NaN` targets contribute zero loss and zero gradient; a graph whose
target is entirely `NaN` contributes exactly `0.0` because the numerator
and denominator both go to zero and the denominator is clamp-min'd to
`1`. The default (`ignore_nan=False`) lets `NaN` propagate, which is
usually what you want during development when a label *shouldn't* be
missing.

```{warning}
Only target `NaN`s are treated as missing labels. Prediction `NaN`s still
propagate whenever the corresponding target is finite; if the target is
`NaN`, that position contributes zero loss and zero gradient. Do not
rely on `ignore_nan` to hide model explosions.
```

## Composition

Real training objectives typically combine several targets. The idiomatic way is
to add leaves together and use the resulting
{py:class}`~nvalchemi.training.ComposedLossFunction`:

```python
from nvalchemi.training import EnergyLoss, ForceLoss, StressLoss

loss_fn = EnergyLoss() + ForceLoss() + StressLoss()
```

`loss_fn` is an `nn.Module` whose components sit in an
`nn.ModuleList`, so `.to(device)`, `.state_dict()`, `.modules()`, and
the nested `__repr__` work the way you'd expect. Adding a
`ComposedLossFunction` to another loss flattens transparently:

```python
loss_fn_a = EnergyLoss() + ForceLoss()
loss_fn_b = loss_fn_a + StressLoss()   # still 3 flat components
```

### The call signature

A composed loss takes **keyed mappings**, not tensors:

```python
def loss_fn(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    step: int = 0,
    epoch: int | None = None,
    **kwargs,
) -> ComposedLossOutput: ...
```

Each component reads its own `prediction_key` and `target_key`
attributes to pull tensors out of the two mappings. Any extra `**kwargs`
(graph metadata, for example) are forwarded unchanged to every leaf;
each leaf consumes the kwargs it needs and ignores the rest.

```python
predictions = {
    "predicted_energy": model_outputs["energy"],
    "predicted_forces": model_outputs["forces"],
    "predicted_stress": model_outputs["stress"],
}
targets = {
    "energy": batch.energy,
    "forces": batch.forces,
    "stress": batch.stress,
}

out = loss_fn(
    predictions, targets,
    step=global_step, epoch=epoch,
    batch_idx=batch.batch_idx,
    num_graphs=batch.num_graphs,
    num_nodes_per_graph=batch.num_nodes_per_graph,
)

out["total_loss"].backward()
```

### The return type

`ComposedLossFunction.forward` returns a
{py:class}`~nvalchemi.training.ComposedLossOutput` — a dict with
`total_loss` plus each component's weighted loss keyed by class name
(duplicate class names get numeric suffixes: `EnergyLoss`,
`EnergyLoss_0`, `EnergyLoss_1`, …).

```python
out = loss_fn(predictions, targets)
print(out)
# {"EnergyLoss": tensor(0.123), "ForceLoss": tensor(0.456), "total_loss": tensor(0.579)}
```

Each component's contribution is already weighted by its own schedule
before being summed. The composition itself carries no schedule of its
own: coefficients and schedules belong on leaf `weight` fields.

When logging per-component values, remember that each `.item()` call
triggers a GPU→CPU synchronization; guard the call with
`if global_step % log_every == 0:` inside a training loop.

### Routing errors

`ComposedLossFunction` validates its inputs eagerly and fails with a
focused error when a contract is broken:

- A missing `prediction_key` or `target_key` in the input mappings
  raises `KeyError`.
- A mapping entry that is not a `torch.Tensor` raises `TypeError`.
- A shape mismatch between `pred` and `target` raises `ValueError`.
- A component class without `prediction_key` / `target_key`
  attributes (e.g. a bespoke loss you forgot to configure) raises
  `AttributeError`.

## Scheduling weights

(scheduling_weights)=

Per-loss coefficients are set with the `weight` keyword on a leaf.
Passing a plain float is not supported — use
{py:class}`~nvalchemi.training.ConstantWeight` to get a static
coefficient that round-trips through specs:

```python
from nvalchemi.training import ConstantWeight, EnergyLoss, ForceLoss

loss_fn = (
    EnergyLoss(weight=ConstantWeight(value=1.0))
    + ForceLoss(weight=ConstantWeight(value=10.0))
)
```

For a curriculum-style ramp, use {py:class}`~nvalchemi.training.LinearWeight`
or {py:class}`~nvalchemi.training.CosineWeight`. For discrete phase
transitions, use {py:class}`~nvalchemi.training.PiecewiseWeight`.

| Schedule | Shape | Typical use |
|----------|-------|-------------|
| {py:class}`~nvalchemi.training.ConstantWeight` | Flat | Static task weight |
| {py:class}`~nvalchemi.training.LinearWeight` | `start` → `end` over `num_steps`, clamped | Curriculum warm-up |
| {py:class}`~nvalchemi.training.CosineWeight` | Half-cosine `start` → `end`, clamped | Smooth curriculum |
| {py:class}`~nvalchemi.training.PiecewiseWeight` | Step function over boundaries | Phase changes |

### Step vs. epoch

Every schedule has a `per_epoch: bool` field. When `False` (the default)
the schedule advances by the `step` argument passed to the loss; when
`True`, it advances by `epoch`. Mixing the two lets most schedules
advance per batch while keeping others, such as a stress-weight
curriculum, aligned with learning-rate epochs.

```python
from nvalchemi.training import (
    ConstantWeight,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    PiecewiseWeight,
    StressLoss,
)

# per-batch schedule (default): index by step
batch_sched = LinearWeight(start=0.0, end=1.0, num_steps=1000)

# per-epoch schedule: index by epoch
epoch_sched = PiecewiseWeight(
    boundaries=(0, 10, 20),
    values=(0.0, 0.5, 1.0, 1.0),
    per_epoch=True,
)

loss_fn = (
    EnergyLoss(weight=ConstantWeight(value=1.0))
    + ForceLoss(weight=batch_sched)
    + StressLoss(weight=epoch_sched)
)

# step 500, epoch 7 → force weight is 0.5 (linear midpoint);
# stress weight is 0.5 (piecewise interval [0, 10)).
print(loss_fn.weight_factors(step=500, epoch=7))
```

A `per_epoch=True` schedule called with `epoch=None` raises
`ValueError` — passing `epoch` is required whenever the attached
schedule opts in.

### Inspecting current weights

`BaseLossFunction.current_weight(step, epoch)` returns the scalar
coefficient that would be applied at a given `(step, epoch)` without
running the model:

```python
energy = EnergyLoss(weight=LinearWeight(start=0.0, end=1.0, num_steps=100))
energy.current_weight(step=50)        # 0.5
energy.current_weight(step=200)       # 1.0 (clamped)
```

`ComposedLossFunction.weight_factors(step, epoch)` reports the current
coefficient of every component in one call, using the same
collision-suffix naming as the output dict:

```python
loss_fn = EnergyLoss() + ForceLoss(weight=ConstantWeight(value=10.0))
loss_fn.weight_factors(step=0)
# {"EnergyLoss": 1.0, "ForceLoss": 10.0}
```

Useful for logging and for sanity-checking that a curriculum is doing
what you expect before kicking off a multi-hour run.

### Bring your own schedule

{py:class}`~nvalchemi.training.LossWeightSchedule` is a
`runtime_checkable` {py:class}`typing.Protocol`: any object with a
`per_epoch` attribute and a `__call__(step: int, epoch: int) -> float`
method qualifies. You don't need to subclass anything to attach a
custom schedule to a loss; it just has to quack like one.

```python
class CappedInverse:
    """Return min(1.0, 1.0 / max(step, 1)) — reciprocal step decay."""

    per_epoch = False

    def __call__(self, step: int, epoch: int) -> float:
        return min(1.0, 1.0 / max(step, 1))

loss = ForceLoss(weight=CappedInverse())
loss.current_weight(step=0)    # 1.0
loss.current_weight(step=10)   # 0.1
```

Subclass the internal `_BaseWeightSchedule` (from
`nvalchemi.training.losses.base`) instead when you want Pydantic
validation and `create_model_spec` round-tripping for checkpoints.

## Writing your own loss

Writing a custom loss is a matter of subclassing
{py:class}`~nvalchemi.training.BaseLossFunction` and implementing
`_forward(pred, target, **kwargs) -> torch.Tensor`. The base class
takes care of shape validation and weight scheduling; your job is the
math.

```{tip}
**Never override `forward`.** The base class's `forward` applies the
weight schedule before calling `_forward`. Overriding `forward` bypasses
scheduling and breaks `ComposedLossFunction` composition.
```

Three conventions worth knowing:

1. **Accept `**kwargs`.** `ComposedLossFunction` forwards every kwarg
   to every component. Swallowing the ones you don't use keeps your
   loss composable with any other loss in the mix.
2. **Define `target_key` and `prediction_key`.** These attributes tell
   `ComposedLossFunction` which slots in the prediction/target mappings
   to wire into your `_forward`. Without them, your loss works
   standalone but cannot participate in a composition.
3. **Keep `_forward` tensor-first.** Don't reach into a `Batch` or
   `HookContext`. Graph metadata — `batch_idx`, `num_nodes_per_graph`,
   custom masks — should arrive as kwargs.

### Example 1: a Huber energy loss

A simple drop-in replacement for MSE. Pure tensor-to-tensor, no graph
metadata, works standalone or inside a composition out of the box.

```python
from typing import Any

import torch
import torch.nn.functional as F

from nvalchemi.training import BaseLossFunction, LossWeightSchedule


class HuberEnergyLoss(BaseLossFunction):
    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        delta: float = 1.0,
        weight: LossWeightSchedule | None = None,
    ) -> None:
        super().__init__(weight=weight)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.delta = delta

    def _forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,   # accept & ignore composition kwargs
    ) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)
```

Override `extra_repr()` if you want `print(loss_fn)` to show
`delta=...` alongside the default fields.

Use it like any other leaf:

```python
from nvalchemi.training import ConstantWeight, ForceLoss

loss_fn = (
    HuberEnergyLoss(delta=0.5, weight=ConstantWeight(value=1.0))
    + ForceLoss(weight=ConstantWeight(value=10.0))
)

out = loss_fn(predictions, targets, step=step, epoch=epoch,
              batch_idx=batch.batch_idx, num_graphs=batch.num_graphs)
```

The `**kwargs` sink is what makes this loss composable: `ForceLoss`
needs `batch_idx` and `num_graphs`, `HuberEnergyLoss` doesn't — and
`ComposedLossFunction` hands both to both components. The Huber loss
simply ignores the metadata it doesn't need.

Because the `weight` kwarg is forwarded to `BaseLossFunction.__init__`,
any {py:class}`~nvalchemi.training.LossWeightSchedule` — including
`CosineWeight`, `LinearWeight`, or a custom object — works without any
extra wiring.

### Example 2: a metadata-aware masked-energy loss

When your loss depends on graph structure, pull the pieces you need out
of `**kwargs`. The established pattern is:

1. Declare typed keyword arguments with defaults of `None`.
2. Validate presence with a focused error — raise `ValueError` with a
   clear message naming the required metadata.
3. Use the reduction helpers in `nvalchemi.training.losses.reductions`
   for scatter-based per-graph reductions.

The example below is a per-atom-count-normalized energy loss: both
`pred` and `target` are per-graph `(B, 1)`, so it passes the
composition shape check and drops into any `ComposedLossFunction`.

```python
from typing import Any

import torch

from nvalchemi.training import BaseLossFunction


class MaskedEnergyLoss(BaseLossFunction):
    """Energy MSE that uses node-count metadata to normalize per atom."""

    target_key = "energy"
    prediction_key = "predicted_energy"

    def _forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        num_nodes_per_graph: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if num_nodes_per_graph is None:
            raise ValueError(
                "MaskedEnergyLoss requires num_nodes_per_graph=... metadata."
            )
        counts = num_nodes_per_graph.to(pred.dtype).clamp_min(1.0)
        return torch.mean(((pred - target) / counts.unsqueeze(-1)) ** 2)
```

`target_key` and `prediction_key` are resolved by composition via
`getattr`, so class-level defaults are enough when a loss has no other
constructor state; the inherited `BaseLossFunction.__init__` still
accepts `weight=...`, so `MaskedEnergyLoss(weight=LinearWeight(...))`
works out of the box. If you want callers to override routing keys or
configure additional fields, expose those via `__init__` the way
`HuberEnergyLoss` does above.

`num_nodes_per_graph` flows through any `ComposedLossFunction.forward`
call like any other keyword — the callsite passes it alongside
`predictions`, `targets`, `step`, and `epoch`.

### Testing a custom loss

Two checks usually suffice:

1. The unweighted `_forward` returns a scalar of the expected dtype
   and gradient flows back to `pred`.
2. If `ignore_nan` matters for your loss, assert that a `NaN`-filled
   target row contributes zero to `pred.grad`.

```python
import torch

loss_fn = HuberEnergyLoss(delta=1.0)
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

value = loss_fn(pred, target)
assert value.ndim == 0
value.backward()
assert pred.grad is not None
```

For composed losses, assert `total_loss == sum(weighted component losses)`
on a tiny batch.

## See also

- **API**: {ref}`losses-api` for the full class and schedule reference.
- **Reductions**: the `nvalchemi.training.losses.reductions` module for
  scatter-based per-graph helpers usable in custom losses.
- **Models**: the [models guide](models) covers the model-side of the
  contract (how `predictions` mappings are produced).
- **Hooks**: the [hooks guide](hooks_guide) covers the
  {py:class}`~nvalchemi.hooks.HookContext` fields a training loop
  makes available, including `ctx.loss`.
