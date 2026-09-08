# Internal Python 3.15 development environment (experimental)

Status: **internal only** — not release support. Python 3.15.0rc1 works for the
core stack as of 2026-09, validated on an L4 (CUDA 12.6 wheels, driver 12.8+):
`import nvalchemi`, ~3,554 tests (models, training, data, dynamics, hooks), and
a GPU smoke (LJ forward + Velocity-Verlet) all pass. The CUDA extras
(`cu12`/`cu13`), `mace`, and `uma` remain blocked on upstream cp315 wheels.

Out-of-the-box `uv sync` targets Python 3.11–3.14; this recipe is a bespoke
environment, deliberately kept out of the release metadata so the default
experience stays intact.

## What blocks what (linux x86_64, as of 2026-09)

- **tensordict**: no cp315 wheel or sdist on any index (PyPI, PyTorch, nightly
  channels). Only path: build from source (below). This is the single hard
  blocker for the core stack.
- **torch**: 2.14.0+cu126 cp315 wheels exist on the standard
  `download.pytorch.org/whl/cu126` index. The wheel self-gates `triton` and
  `cuda-bindings` behind `python_version < "3.15"`, so they never conflict.
- **torch.compile**: NOT supported on 3.15 (torch raises at runtime). Since
  `nvalchemiops.torch.neighbors.neighbor_utils` applies a bare `@torch.compile`
  at import time, `import nvalchemi.hooks` fails without the shim below.
- **nvalchemi-toolkit-ops / nvidia-physicsnemo**: both declare
  `requires-python <3.15` but are pure wheels — install with
  `--ignore-requires-python`.
- **cu12/cu13 extras**: dead on 3.15 — cupy-cuda1x, numba, numba-cuda,
  cuda-bindings/core, ray, nvidia-dali, cuequivariance-ops have no cp315
  binaries and mostly no sdists. Revisit when RAPIDS/cupy ship cp315.
- **beartype**: 0.22.x imports a symbol removed in 3.15; needs
  `beartype>=0.23.0rc0` (pre-release).
- **plotext**: use `<6` (6.x removed `clf()` used by hooks/reporting and
  training/cli — the release spec now caps this too).
- **h5py**: no stable cp315; nightly wheels at
  `https://pypi.anaconda.org/scientific-python-nightly-wheels/simple`.

## Recipe

```bash
# 1. venv (uv downloads cpython-3.15.0rc1; --seed gives pip fallback)
uv venv /tmp/py315 --python 3.15 --seed
PY=/tmp/py315/bin/python

# 2. torch from the cu126 index (cp315 wheel; pick cu130 for CUDA 13 stacks)
uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.14.0" "torchvision==0.29.0"

# 3. tensordict from source (needs cmake >= 3.26 + pybind11 >= 3.1;
#    ~10 min, keep MAX_JOBS low on small machines)
uv pip install --python $PY "pybind11>=3.1.0" cmake ninja
MAX_JOBS=2 uv pip install --python $PY \
    "tensordict @ git+https://github.com/pytorch/tensordict"

# 4. the rest of the core stack; pin the two known-problem packages
uv pip install --python $PY \
    "numpy>=2.5" "beartype>=0.23.0rc0" "plotext<6" \
    jaxtyping pydantic click loguru plum-dispatch "zarr>=3" \
    periodictable==2.0.2 rich dm-tree
uv pip install --python $PY --prerelease \
    --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    h5py

# 5. metadata-capped pure wheels (ignore requires-python)
$PY -m pip install --ignore-requires-python nvalchemi-toolkit-ops
$PY -m pip install --ignore-requires-python --no-deps nvidia-physicsnemo
#    then let uv resolve physicsnemo's remaining deps manually if needed

# 6. eager-compile shim: put this in a sitecustomize.py on PYTHONPATH
#    (mp.spawn children inherit it via env) until nvalchemiops makes
#    its @torch.compile lazy:
#        import torch
#        torch.compile = lambda fn=None, **kw: (fn if callable(fn) else (lambda f: f))

# 7. the repo itself
uv pip install --python $PY --no-deps \
    -e /path/to/nvalchemi-toolkit
uv pip install --python $PY pytest pytest-asyncio pytest-timeout \
    pytest-dependency hypothesis hypothesis-torch pyyaml

# 8. run
cd /path/to/nvalchemi-toolkit && $PY -m pytest test/models test/data
```

## Upstream asks (to make this unnecessary)

1. `nvalchemi-toolkit-ops`: relax `requires-python` to admit 3.15, and make
   the `@torch.compile` in `neighbor_utils` lazy/conditional so importing
   `nvalchemi.hooks` does not require a working torch.compile.
2. tensordict: any cp315 wheel or sdist release would remove the last source
   build from this recipe.
3. RAPIDS/cupy/cuequivariance: cp315 wheels would unlock the CUDA extras.
