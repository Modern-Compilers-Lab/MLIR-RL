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

Activate the conda env and run `make`:

```bash
conda activate main
make
```

This produces three plugins in `build/lib/`:

- `build/lib/libEquivalenceVerifier.so`
- `build/lib/libTagLinalgOps.so`
- `build/lib/libRaiseSCFToAffine.so`

## Usage

Use `verify_equivalence.py`. It takes one MLIR file containing a kernel plus its
transform schedule — exactly the shape of the files in
[`tests/validation/`](../../../../tests/validation): one `func.func` followed by a
`module attributes {transform.with_named_sequence}`:

```mlir
// The kernel (taken as the "original").
func.func @kernel(%arg0: memref<...>, ...) {
  // ... linalg ops, tagged with {tag = "..."} so the schedule can match them ...
  return
}

// The transform schedule, applied to a copy to produce the "transformed".
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // ... transform.structured.* ops matching the tagged kernel ops ...
    transform.yield
  }
}
```

The script applies the schedule, prepares the IR, runs the verifier, and prints
the verdict:

```bash
conda activate main
python verify_equivalence.py path/to/kernel_and_schedule.mlir
```

It prints one of:

- `valid` — the transform preserves the original array dataflow (exit 0)
- `invalid` — a dependence is not preserved; the transform is illegal (exit 1)
- `error` — the transform schedule itself failed to apply or lower (exit 2)

Add `-v` / `--verbose` to also print the verifier's trace and diagnostics:

```bash
python verify_equivalence.py path/to/kernel_and_schedule.mlir --verbose
```

## Running the pass without the script

If you want to drive the passes yourself, the script does the following in a
single Python process:

1. **Load the plugins.** `dlopen` each `.so` with `RTLD_GLOBAL` so its pass
   self-registers and resolves by name in `apply_pipeline_to_module`.
2. **Tag, before transforming.** Run `tag-linalg-ops-for-equivalence` on the
   kernel that has linalg operations, then clone it to create the transform
   module (don't use textual cloning since name loc information will be lost).
3. **Transform.** Apply the schedule to the cloned copy.
4. **Lower both** to affine + memref: bufferize, then
   `builtin.module(func.func(scf-forall-to-for,raise-scf-to-affine),convert-linalg-to-affine-loops,func.func(fold-memref-alias-ops,affine-raise-from-memref))`.
5. **Combine** the two functions into one module, renamed `original` and
   `transformed`.
6. **Check.** Run `check-array-dataflow-equivalence{original-func=original
   transformed-func=transformed}` (add `verbose` for a trace). A non-empty
   stderr / pass failure means a dependence was not preserved.

See [`verify_equivalence.py`](verify_equivalence.py) for the exact pipeline
strings and helper calls.

## Notes

- The two functions must have matching signatures.
- Only function-argument memrefs, their aliases, and any buffers copied to/from them are checked.
