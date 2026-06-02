# Equivalence Verifier

Proves that an MLIR transform schedule preserves the semantics of a kernel by
checking **array-dataflow equivalence** at the stable-memref boundary. Given an
`original` and a `transformed` affine function, it verifies that every original
read-after-write, write-after-read, and write-after-write dependence over
function-argument memrefs is still preserved after transformation. Local
allocations introduced by the transform are ignored — only accesses that
ultimately alias a function `BlockArgument` are considered. The check is a
Presburger emptiness query over the composed access relations and the
lexicographic schedule derived from the surrounding `affine.for` loops.

## Components

The build produces three self-registering pass plugins (`.so`):

| Plugin | Pass name | Role |
| --- | --- | --- |
| `libEquivalenceVerifier.so` | `check-array-dataflow-equivalence` | The verifier. Compares `original` vs `transformed` and fails (emits an error to stderr) when a dependence is not preserved. |
| `libTagLinalgOps.so` | `tag-linalg-ops-for-equivalence` | Tags each `linalg` op with a stable `eq_id_<n>` NameLoc so the verifier can match an original access to its transformed counterpart. Run on the still-`linalg` form of **both** functions before lowering. |
| `libRaiseSCFToAffine.so` | `raise-scf-to-affine` | Raises `scf.for` loops to `affine.for` so the verifier can analyze the memory accesses w.r.t induction variables. Pair with `scf-forall-to-for` for parallel tiling. |

## Installation

The plugins link against the MLIR/LLVM shipped in the project conda env, so
activate it (or pass `PREFIX`) first, then run `make`:

```bash
# from this directory, with the conda env active
make                       # uses $CONDA_PREFIX

# or point PREFIX at any MLIR install prefix explicitly
make PREFIX=/path/to/env

# or derive the prefix from the active Python
make PREFIX=$(python -c 'import sys; print(sys.prefix)')

make clean                 # remove build artifacts
```

Outputs land in `build/lib/`. `make` runs `mlir-tblgen` to emit
`build/include/Passes.h.inc` (pass declarations from `Passes.td`), then builds
each `.so`. Each plugin carries its own `mlirGetPassPluginInfo` entry point and
self-registers its pass on `dlopen`, so its pass name resolves in both
`mlir-opt` and the Python `PassManager`.

## Usage

The verifier expects a single module containing two functions named `original`
and `transformed`, both lowered to the **affine + memref** form. The pass
options are:

- `original-func=<name>` — name of the original function (default `original`)
- `transformed-func=<name>` — name of the transformed function (default `transformed`)
- `verbose` — emit a step-by-step trace of the check to stdout

A non-empty stderr / non-zero exit means a dependence was **not** preserved.

### 1. From the command line (`mlir-opt`)

Load the plugins and run the pass on an already-lowered module:

```bash
mlir-opt prepared.mlir \
  --load-pass-plugin=build/lib/libEquivalenceVerifier.so \
  --pass-pipeline='builtin.module(check-array-dataflow-equivalence{original-func=original transformed-func=transformed verbose})'
```

To go from a `linalg` kernel to the form the verifier consumes, tag and lower
first (load all three plugins). The canonical lowering pipeline is:

```text
builtin.module(
  func.func(scf-forall-to-for, raise-scf-to-affine),
  convert-linalg-to-affine-loops,
  func.func(fold-memref-alias-ops, affine-raise-from-memref))
```

Tag both functions with `builtin.module(func.func(tag-linalg-ops-for-equivalence))`
on the still-`linalg` form *before* applying this lowering.

### 2. From Python (ctypes + MLIR bindings)

`dlopen` each plugin with `RTLD_GLOBAL` so its pass self-registers, then drive
the pipelines through the Python bindings:

```python
import ctypes
from pathlib import Path

LIB = Path("build/lib")
for so in ("libTagLinalgOps.so", "libRaiseSCFToAffine.so", "libEquivalenceVerifier.so"):
    ctypes.CDLL(str(LIB / so), mode=ctypes.RTLD_GLOBAL)

# Now `tag-linalg-ops-for-equivalence`, `raise-scf-to-affine`, and
# `check-array-dataflow-equivalence` resolve in PassManager.parse / the
# apply_pipeline_to_module helpers in llm_transform.utils.transformation.
```

Sketch of the full flow (see `run_equivalence_verifier` in
`test_mlir_validation.py` for a complete, working example):

1. Parse the kernel and tag its linalg ops (`tag-linalg-ops-for-equivalence`).
2. Clone it: keep one copy as `original`, apply the transform schedule to the other to get `transformed`.
3. Bufferize and lower both with the affine + memref pipeline above.
4. Place both functions in one module, renamed `original` and `transformed`.
5. Run `check-array-dataflow-equivalence`; non-empty stderr means a violation.

### 3. Via the validation harness

`test_mlir_validation.py` (repo root) wires all of this up as one of two
detectors (`mlir`, `equivalence`). It loads the plugins, splits
each `tests/validation/*.mlir` into kernel + schedule, runs both variants, and
compares the equivalence verdict against the ground-truth executed outputs.
Restrict to this detector with `--tool equivalence`.

## Notes

- The two functions must have matching signatures; the verifier walks linalg ops in pre-order, so both must be tagged in the same order for `eq_id_<n>` tags to line up.
- Only function-argument memrefs are checked. A transform that is correct purely through scratch buffers it allocates and frees is treated as a no-op on the stable boundary.
- If a plugin `.so` is missing, rebuild with `make` (see Installation); the harness prints a build hint and reports the column as `n/a`.
