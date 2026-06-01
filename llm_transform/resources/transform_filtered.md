# MLIR Transform Dialect — Selected Operations

Documentation extracted from `Transform.md`.

## `transform.sequence` (transform::SequenceOp)

_Contains a sequence of other transform ops to apply_

Syntax:

```
operation ::= `transform.sequence` custom<SequenceOpOperands>($root, type($root), $extra_bindings, type($extra_bindings)) (`->` type($results)^)? `failures` `(` $failure_propagation_mode `)` attr-dict-with-keyword regions
```

The transformations indicated by the sequence are applied in order of their
appearance. Each value produced by a transformation within the sequence
corresponds to a group of operations or values in the payload IR, or to a
group of parameters, depending on the type of the value. The behavior of the
operation when a nested transformation produces a silenceable error is
controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the caller to ensure the following transformations are
robust enough when errors are suppressed. Definite errors reported by nested
transformations abort the sequence regardless of the propagation mode. The
set of modes may be extended in the future, e.g., to collect silenceable
errors and report them after attempting all transformations in the sequence.

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.

The type of the block argument must match the type of the operand. If the
sequence is a top-level transform (without an operand), it can be used for
matching operations if the specified type within the top-level container
payload IR (including the container op itself). E.g.:

```mlir
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // %arg1 is mapped to the top-level container of the payload IR, which is
  // typically a module
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.op<"func.func>"):
  // %arg1 is mapped to all "func.func" ops within and including the
  // top-level container of the payload IR. Nested operations that have the
  // specified op type are not included.
}
```

The body of the sequence terminates with an implicit or explicit
`transform.yield` op. The operands of the terminator are returned as the
results of the sequence op.

Traits: `AttrSizedOperandSegments`, `PossibleTopLevelTransformOpTrait`, `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `OpAsmOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>failure_propagation_mode</code></td><td>::mlir::transform::FailurePropagationModeAttr</td><td>Silenceable error propagation policy</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |
| `extra_bindings` | variadic of any transform handle or parameter |

### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of TransformHandleTypeInterface instance |

## `transform.alternatives` (transform::AlternativesOp)

_Attempts sequences of transforms until one succeeds_

Syntax:

```
operation ::= `transform.alternatives` ($scope^ `:` type($scope))? (`->` type($results)^)? attr-dict-with-keyword regions
```

This op may have an arbitrary number of regions, each of which represents a
sequence of transform operations to be applied to the same payload IR. The
regions are visited in order of appearance, and transforms in them are
applied in their respective order of appearance. If one of these transforms
fails to apply, the remaining ops in the same region are skipped an the next
region is attempted. If all transformations in a region succeed, the
remaining regions are skipped and the entire "alternatives" transformation
succeeds. If all regions contained a failing transformation, the entire
"alternatives" transformation fails.

It is up to the nested operations to define which errors are "recoverable"
(or "silenceable") and allow another alternatives to be attempted, and which
errors should be propagated without attempting the other alternatives.

The single operand of this operation is the scope in which the alternative
transformation sequences are attempted, that is, an operation in the payload
IR that contains all the other operations that may be modified by the
transformations. The scope operation must be isolated from above. There is
no check that the transforms are indeed scoped as their "apply" methods can
be arbitrarily complex. Therefore it is the responsibility of the user to
ensure that the transforms are scoped correctly, or to produce an
irrecoverable error and thus abort the execution without attempting the
remaining alternatives. Note that the payload IR outside of the given scope
is not necessarily in the valid state, or even accessible to the
transformation.

The changes to the IR within the scope performed by transforms in the failed
alternative region are reverted before attempting the next region.
Practically, this is achieved by cloning the scope. Therefore it is advised
to limit the scope as much as possible and place the most likely
alternatives early in the region list. The operation is also isolated from
above and requires rediscovering the operations within the given scope to
avoid additional handle invalidation. The latter restriction may be lifted
in the future.

Each of the regions may yield transform IR handles. The handles of the first
successful alternative region are returned as the results of the
"alternatives" op. Therefore, each alternative region must yield the same
number of results, which should also match the number and the types of the
"alternatives" op results.

Remark: this op allows one to implement a simple "try" construct as follows:

```mlir
%result = transform.alternatives %scope {
^bb0(%arg0: !transform.any_op):
  // Try a fallible transformation.
  %0 = transform.fallible %arg0 // ...
  // If succeeded, yield the the result of the transformation.
  transform.yield %0 : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // Otherwise, the second alternative is tried and it always succeeds by
  // returning the original handle.
  transform.yield %arg0 : !transform.any_op
}
```

Traits: `IsolatedFromAbove`, `PossibleTopLevelTransformOpTrait`, `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `scope` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of TransformHandleTypeInterface instance |

## `transform.yield` (transform::YieldOp)

_Yields operation handles from a transform IR region_

Syntax:

```
operation ::= `transform.yield` operands attr-dict (`:` type($operands)^)?
```

This terminator operation yields operation handles from regions of the
transform IR ops back to the containing op. It is not itself associated with
any transformation on the payload IR and is used for flow purposes only.

Traits: `Terminator`

Interfaces: `MemoryEffectOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any transform handle or parameter |

## `transform.structured.match` (transform::MatchOp)

Syntax:

```
operation ::= `transform.structured.match` (`ops` `{` $ops^ `}`)?
              (`interface` `{` $interface^ `}`)?
              (`attributes` $op_attrs^)?
              (`filter_result_type` `=` $filter_result_type^)?
              (`filter_operand_types` `=` $filter_operand_types^)?
              `in` $target attr-dict
              `:` functional-type($target, results)
```

Match op with the specified constraints, within the target op.

The following constraints are supported:
  - interface: an optional MatchInterfaceEnum specifying an enum
    representation for an interface to target.
  - ops: an optional StrArrayAttr specifying the concrete name of an op.
    Multiple names can be specified. Matched ops must have one of specified
    names.
  - attribute: the matched op must have all specified attributes (with their
    specified values).
  - filter_result_type: the matched op must return exactly this one type.
  - filter_operand_types: all the operands of the matched op must must be of
    this type. If more than a type is specified, then the length of the list
    must be equal to the number of operands in the matched op, and the match
    will succeed only if the operand types match all the types in the list
    in the order in which they are specified.

Note: Only ops that satisfy all specified constraints are matched.

TODO: Extend with regions to allow a limited form of constraints.

### Return modes

This op traverses the ops nested under `target` and returns the handles to
all the operations that match the requirements.

This op fails if the target is not a handle to exactly one operation.
Otherwise it succeeds.

This operation does not consume the target handle and produces new handles:
it is a navigation op.

Traits: `NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>ops</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>interface</code></td><td>mlir::transform::MatchInterfaceEnumAttr</td><td>An interface to match</td></tr>
<tr><td><code>op_attrs</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
<tr><td><code>filter_result_type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
<tr><td><code>filter_operand_types</code></td><td>::mlir::ArrayAttr</td><td>type array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `results` | TransformHandleTypeInterface instance |

## `transform.get_producer_of_operand` (transform::GetProducerOfOperand)

_Get handle to the producer of this operation's operand number_

Syntax:

```
operation ::= `transform.get_producer_of_operand` $target `[` $operand_number `]` attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to operation that
produces the SSA value defined by the `target` and `operand_number`
arguments. If the origin of the SSA value is not an operations (i.e. it is
a block argument), the transform produces a silenceable failure.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>operand_number</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `producer` | TransformHandleTypeInterface instance |

## `transform.get_consumers_of_result` (transform::GetConsumersOfResult)

_Get handle to the consumers of this operation's result number_

Syntax:

```
operation ::= `transform.get_consumers_of_result` $target `[` $result_number `]` attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to all operations that
consume the SSA value defined by the `target` and `result_number`
arguments.
This operation applies to a single payload operation, otherwise it produces
a definite failure.
The return handle points to the consuming operations operations, which can
be empty.

Traits: `NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>result_number</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `consumers` | TransformHandleTypeInterface instance |

## `transform.get_parent_op` (transform::GetParentOp)

_Gets handles to the closest parent ops_

Syntax:

```
operation ::= `transform.get_parent_op` $target attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to the parents of the
targeted payload ops (in the same order).

Requirements that parent ops must fulfill can be optionally specified. In
that case for each target op, the closest parent op that fulfills all
requirements, is returned.
- `isolated_from_above`: the parent op must be isolated from above
- `allow_empty_results`: get_parent_op is allowed to return an empty list
  and still succeeds. In such a case, if `get_parent_op` fails for any
  operation in the list, the entire transform returns an empty handle.
- `op_name`: the parent op must have the specified name
- `nth_parent`: get the n-th parent of that satisfies the above requirements

If `deduplicate` is set, the result handle does not contain any duplicate
ops. For example, given the list
"(childof(A), childof(B), childof(B), childof(A), childof(B))", the
resulting list will be just "(A, B)". Note that no other semantic ordering
is applied, e.g., "B" may itself be a parent of "A". This may have an impact
on the further transformation applied to the handle produced here.

If any of the given Payload IR ops has no such suitable parent, then:
  - if `allow_empty_results` is set, the result handle is empty
  - otherwise, the transformation produces a silenceable failure.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>isolated_from_above</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>allow_empty_results</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>op_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>deduplicate</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>nth_parent</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `parent` | TransformHandleTypeInterface instance |

## `transform.structured.tile_using_for` (transform::TileUsingForOp)

Syntax:

```
operation ::= `transform.structured.tile_using_for` $target
              `tile_sizes` custom<DynamicIndexList>(
              $dynamic_sizes,
              $static_sizes,
              $scalable_sizes)
              (`interchange` `=` $interchange^)?
              attr-dict
              `:` functional-type(operands, results)
```

Indicates that the given `target` op should be tiled with the given sizes.
This transform generates a loop nest with a smaller ("tiled") target
operation in its body. Currently limited to LinalgOps.

Tile sizes may be known at transformation time, in which case they are
expected to be provided in the `static_size` attribute, or not, in which
case the tile value must be computed by the payload IR and the handle to the
operation computing it must be provided through `dynamic_sizes`. When the
sizes are not known statically, the corresponding entry in the
`static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
be as many handles as `ShapedType::kDynamic` values in the
`static_sizes` attribute. A static size of `0` indicates that the dimension
should not be tiled. No loop will be generated for such dimensions. If all
tile sizes are `0`, this transform is effectively a no-op.

This op returns handles to the tiled op (in the generated loop nest) and the
generated loops. The number of loops is the number of tile sizes that are
statically known to be non-zero.

### Return modes

On success, the resulting handles are associated with co-indexed lists of
tiled operations and loops around them.

This operation only supports Linalg ops and produces a silenceable failure
if the input contains any non-Linalg ops. The ops preceding it in the list
associated with the `target` handle will have been tiled.

This operation produces a silenceable failure if the `dynamic_sizes` handles
are associated with lists of payload operations of a size different than
that of the list associated with the `target` handle.

If the internal implementation of tiling for any of the operations fails,
produces a definite failure.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>interchange</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>scalable_sizes</code></td><td>::mlir::DenseBoolArrayAttr</td><td>i1 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_sizes` | variadic of transform any param type or any handle type |

### Results:

| Result | Description |
| :----: | ----------- |
| `tiled_linalg_op` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |

## `transform.structured.tile_using_forall` (transform::TileUsingForallOp)

Syntax:

```
operation ::= `transform.structured.tile_using_forall` $target oilist(
              `num_threads` custom<PackedOrDynamicIndexList>($packed_num_threads,
              $num_threads,
              $static_num_threads) |
              `tile_sizes` custom<PackedOrDynamicIndexList>($packed_tile_sizes,
              $tile_sizes,
              $static_tile_sizes))
              (`(` `mapping` `=` $mapping^ `)`)? attr-dict
              `:` functional-type(operands, results)
```

Tile a TilingInterface op to a tiled `scf.forall`.

Tiling is applied by either specifying `num_threads` or `tile_size`. If
`num_threads` is specified, then the tile size for each dimension `i` is
calculated dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
`num_threads` and `tile_size` can be either static index attributes or
operation handles (or a mix thereof). Operation handles must be mapped to
exactly one op that has exactly one result of index type.

Static zero tile sizes indicate that the dimension is not tiled and can be
thought of as tiling by the full size of data.

It is the user's responsibility to ensure that `num_threads/tile_sizes` is
a valid tiling specification (i.e. that only tiles parallel dimensions,
e.g. in the Linalg case). If the dimension is not parallelizable, a warning
is issued to notify the user that the generated code is not safe to
parallelize.

If non-empty, the `mapping` is added as an attribute to the
resulting `scf.forall`.

Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
threads can be an index attribute or a transform handle that is mapped to
exactly one payload op with exactly one index result.

### Return modes

This operation ignores ops that do not implement the TilingInterface and
drops them in the return.

If all the operations referred to by the `target` handle tile
successfully, the transform succeeds.
Otherwise the transform produces a silenceable failure.

The two returned handles point to only the subset of successfully produced
tiled operations, which can all be empty.

These two returned handles point to:
  - the tiled op that implements TilingInterface,
  - the new scf.forall op.

### Example using `num_threads`

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%3:2 = transform.structured.tile_using_forall %0 num_threads [10, 20]
   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### Example using `tile_sizes`

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%sz = transform.structured.match ...
%3:2 = transform.structured.tile_using_forall %0 tile_sizes [0, %sz, 20]
   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
```

Traits: `AttrSizedOperandSegments`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_num_threads</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>static_tile_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>mapping</code></td><td>::mlir::ArrayAttr</td><td>Device Mapping array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `num_threads` | variadic of transform any param type or any handle type |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `packed_num_threads` | transform any param type or any handle type |
| `packed_tile_sizes` | transform any param type or any handle type |

### Results:

| Result | Description |
| :----: | ----------- |
| `tiled_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |

## `transform.structured.tile_reduction_using_for` (transform::TileReductionUsingForOp)

Syntax:

```
operation ::= `transform.structured.tile_reduction_using_for` $target
              (`reduction_dims` `=` $reduction_dims^)?
              `by` `tile_sizes` `=` $tile_sizes
              attr-dict
              `:` functional-type(operands, results)
```

Indicates that the given `target` op should be transformed with the
`tileReduction` transformation with the tile size provided as attribute.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates nested
loops with a parallel version of `target` op inside. The parallel op
dimensions are less or equal to the tile size passed by user.
After the loop a merge operation is created to do a final reduction with the
partial reductions.
The initial tensor always uses the tile size dimension. This may overallocate
if the tile size is greater than the reduction dimension.

### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `for` op.

The `reduction_dims` can be used to specify the subset of reduction dimensions
of the operation to tile. If left unspecified, all reduction dimensions are
tiled.

### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.for %arg2 = %c0 to %dim_0 step %c5 iter_args(%arg3 = %1) -> (tensor<?x5xf32>) {
    %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, 5] [1, 1] : tensor<?x5xf32> to tensor<?x5xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg2] [%dim, 5] [1, 1] : tensor<?x?xf32> to tensor<?x5xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%extracted_slice_2 : tensor<?x5xf32>)
    outs(%extracted_slice : tensor<?x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<?x5xf32>
    %dim_3 = tensor.dim %1, %c0 : tensor<?x5xf32>
    %inserted_slice = tensor.insert_slice %4 into %arg3[0, 0] [%dim_3, 5] [1, 1] : tensor<?x5xf32> into tensor<?x5xf32>
    scf.yield %inserted_slice : tensor<?x5xf32>
  }
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%2 : tensor<?x5xf32>)
  outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction_dims</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `for_op` | TransformHandleTypeInterface instance |

## `transform.structured.tile_reduction_using_forall` (transform::TileReductionUsingForallOp)

Syntax:

```
operation ::= `transform.structured.tile_reduction_using_forall` $target
              (`reduction_dims` `=` $reduction_dims^)?
              `by`
              (`num_threads` `=` $num_threads^)?
              (`tile_sizes` `=` $tile_sizes^)?
              (`mapping` `=` $mapping^)?
              attr-dict
              `:` functional-type(operands, results)
```

Tile a PartialReductionOpInterface op to a tiled `scf.forall` doing
partial reduction.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates a
`scf.forall` loops with the number threads given by `num_threads`.
The op is tiled op with a size equal to `floordiv(size, num_threads)`.
All the partial reduction value is are parallel inserted to create a new
tensor. After the loop a merge operation is created to do a final reduction
with the partial reductions tensor.
If an extra `tile_sizes` parameter is passed the tiles are cyclically
distributed on the threads of the `scf.foralls` loop.

### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `forall` op.

### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.forall (%arg2) in (%c5) shared_outs(%arg3 = %1) -> (tensor<?x5xf32>) {
    %4 = affine.min #map(%arg2)[%dim_0]
    %5 = affine.max #map1(%4)
    %extracted_slice = tensor.extract_slice %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
    %6 = affine.apply #map2(%arg2)[%dim_0]
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %6] [%dim, %5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice[0] [%dim] [1] : tensor<?xf32> to tensor<?xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_2 : tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
    }
  } {mapping = []}
  %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<?x5xf32>) outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction_dims</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>num_threads</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>mapping</code></td><td>::mlir::ArrayAttr</td><td>Device Mapping array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |

## `transform.structured.fuse` (transform::FuseOp)

Syntax:

```
operation ::= `transform.structured.fuse` $target ($tile_sizes^)? (`interchange` $tile_interchange^)?
              (`apply_cleanup` `=` $apply_cleanup^)? attr-dict
              `:` functional-type(operands, results)
```

Tiles the operations pointed to by the target handle and fuses their
producers greedily using the options provided as attributes.

If `apply_cleanup` is true then slice canonicalization is applied between
fusion steps.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>tile_interchange</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>apply_cleanup</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |

## `transform.structured.fuse_into_containing_op` (transform::FuseIntoContainingOp)

_Fuse a producer into a containing operation._

Syntax:

```
operation ::= `transform.structured.fuse_into_containing_op` $producer_op `into` $containing_op attr-dict  `:` functional-type(operands, results)
```

Fuses the `producer_op` into the `containing_op`.
Returns a handle to the fused ops and the `new_containing_op`.

The producer is typically a slice of a tileable op (i.e., implements
TilingInterface). In that case, this transform computes the accessed
producer slice inside of the containing op ("tile and fuse") and if required,
creates a new containing op with outputs from the fused producer. Otherwise,
the entire producer is cloned inside the containing op ("clone and fuse").

The containing op handle must be associated with exactly one payload op. The
producer op handle may be associated with multiple payload ops. This
transform fuses producers one-by-one, always picking an unspecified producer
that has at least one use inside the containing op among the
producers. A producer can be listed multiple times in the handle.

Note: If a producer has multiple uses inside the containing op, it is
currently tiled and/or cloned multiple times into the containing op.
TODO: Reuse already fused OpResults instead of tiling/cloning a second time
when possible. Fuse producers according to a topological sorting to achieve
the largest amount of reuse.

### Return modes

If at least one producer could not be fused, this operation produces a
silenceable failure.  This is the case when tiling fails or when no
producer op could be found among the remaining producers that has at least
one use within the containing op. I.e., "producers" that are not consumed
within the containing op are rejected by this operation.

This operation consumes the producer handle.
This operation only reads the containing op handle.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `producer_op` | TransformHandleTypeInterface instance |
| `containing_op` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `fused_op` | TransformHandleTypeInterface instance |
| `new_containing_op` | TransformHandleTypeInterface instance |

## `transform.structured.interchange` (transform::InterchangeOp)

Syntax:

```
operation ::= `transform.structured.interchange` $target
              (`iterator_interchange` `=` $iterator_interchange^)? attr-dict
              `:` custom<SemiFunctionType>(type($target), type($transformed), "false")
```

Interchanges the iterators of the operations pointed to by the target handle
using the iterator interchange attribute.

### Return modes

This operation ignores non-linalg::Generic ops and drops them in the return.
This operation fails if the interchange attribute is invalid.
If all the operations referred to by the `target` handle interchange
properly, the transform succeeds.
If any interchange fails, the transform produces a definite failure.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>iterator_interchange</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute whose value is non-negative</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.structured.split` (transform::SplitOp)

Splits the given `target` op into two or more complementary
parts, which combined cover the entire iteration domain of the original op.
The split is performed along the iteration space dimension provided as
chunk size attribute specifying the size of the lower part; the remaining
range in the iteration space is assigned as the upper part. In case of
dimension overflow, the transformation fails. The split is performed at the
dimension iterator value specified as either the static chunk size
attribute when it is known at transform IR construction time or
as the handle to an operation producing a single index-typed value
when it is computed by payload IR. In the latter case, the chunk size
point must be set to `ShapedType::kDynamic` and the dynamic size handle
must point to as many value-producing operations as there are structured
operations pointed to by the target handle.

The operation consumes the target handle, but preserves the chunk size
handle if provided. Without the `multiway` attribute, it produces a
new handle that is a list of the two parts of the structured op after
splitting, whose lower index part corresponding to the part with lower
iteration space indices.

Multiway split mode is enabled by specifying the `multiway` attribute.
In this mode a single `target` op is split into multiple parts covering
the iteration space of the specified dimension. `static_chunk_sizes` and
`dynamic_chunk_sizes` in this case is a list of chunk sizes that the given
dimension should be split into. With `multiway` it also produces a handle;
The result handle is a list of the multiple parts of the structured op
after splitting, where the target dimensions for each linalg op in the
list corresponds to the chunk sizes specfied in the input split list.
If the chunk sizes do not cover the entire iteration space, the leftover
chunk is the last payload in the result handle.

As the result handle is most of time a list, an `transform.split_handle`
is needed to access individual handle.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>static_chunk_sizes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>multiway</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_chunk_sizes` | transform any param type or any handle type |

### Results:

| Result | Description |
| :----: | ----------- |
| `split_list` | TransformHandleTypeInterface instance |

## `transform.structured.split_reduction` (transform::SplitReductionOp)

Syntax:

```
operation ::= `transform.structured.split_reduction` $target attr-dict `:`functional-type(operands, results)
```

Indicates that the given `target` op should be transformed with the
`splitReduction` transformation and split factor provided as attribute.

The `splitReduction` transformation splits the first single linalg op
reduction into a parallel and reduction dimension.
A new `linalg.generic` op is created to perform the rest of the reduction.

The transformation supports different configurations attributes:
  - split_factor: the factor by which to split (i.e. the size of the
    remaining reduction after splitting).
  - insert_split_dimension: the dimension in the temporary tensor into
    which the new parallel dimension is inserted.
  - inner_parallel: specifies whether the parallel dimension is before or
    after the reduction dimension in the splitting op.
  - use_scaling_algorithm: whether to use a scaling based formulation that
    does not create an ExpandShapeOp (default: do not use scaling)
  - use_alloc: whether to use an alloc op to allocate the temporary
    tensor (default: do not use alloc op)

### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation produces a definite failure if the splitting fails for any
reason.

If all the operations referred to by the `target` handle split
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.  The 4 returned handles points to only the subset of
successfully produced computational operations, which can all be empty.
This 4 returned handles point to:
  - the init op (or tensor_alloc op if use_alloc = true),
  - the fill op used to initialize the neutral element,
  - the split op and
  - the result-combining op.

### Example (default: `use_scaling_algorithm = false, use_alloc = false`):

```
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
  ins(%in : tensor<32xf32>)
  outs(%out : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %y = arith.addf %arg1, %arg2 : f32
    linalg.yield %y : f32
  } -> tensor<f32>
```

is split into:

```
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
    ^bb0(%arg3: f32, %arg5: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>
```

### Example (`use_scaling_algorithm = true, use_alloc = true`):

Instead of introducing an ExpandShapeOp, this scaling-based implementation
rewrites a reduction dimension `k` into `k * split_factor + kk`.
The dimension `kk` is added as an extra parallel dimension to the
intermediate output tensor at position `insert_split_dimension`.

Consider a minimal example where `k` is reduced:
    O(i, j) += I(i, j, k)
Assume i=3, j=5, k=128, split_factor=16 and insert_split_dimension=0.
The compute is rewritten as:
  a. O_i(kk, i, j) += I(i, j, 16 * k + kk)
  b. O(i, j) += O_i(kk, i, j)
The intermediate tensor O_i is of shape (128/16)x3x5 == 8x3x5.

### Example:

```
 %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
   outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
```

Is transformed to:

```
 #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
 #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
 #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
 #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
 #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
 #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
 %0 = tensor.empty() : tensor<16x32x64xf32>
 %cst = arith.constant 0.000000e+00 : f32
 %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
    tensor<16x32x64xf32>
 %2 = tensor.empty() : tensor<64x4xi1>

 %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
   ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
   outs(%1 : tensor<16x32x64xf32>) {
     ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
       %5 = arith.mulf %arg3, %arg4 : f32
       %6 = arith.addf %arg6, %5 : f32
       linalg.yield %6 : f32
 } -> tensor<16x32x64xf32>

 %4 = linalg.generic {indexing_maps = [#map4, #map5],
   iterator_types = ["parallel", "parallel", "reduction"]}
   ins(%3 : tensor<16x32x64xf32>)
   outs(%C : tensor<16x32xf32>) {
     ^bb0(%arg3: f32, %arg4: f32):
       %5 = arith.addf %arg3, %arg4 : f32
       linalg.yield %5 : f32
 } -> tensor<16x32xf32>

 return %4 : tensor<16x32xf32>
```

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>split_factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>insert_split_dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>inner_parallel</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_scaling_algorithm</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_alloc</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `init_or_alloc_op` | TransformHandleTypeInterface instance |
| `fill_op` | TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |

## `transform.structured.multitile_sizes` (transform::MultiTileSizesOp)

Syntax:

```
operation ::= `transform.structured.multitile_sizes` $target attr-dict `:` custom<MultitileSizesTypes>(type($target), type($low_size), type($high_size), type($split_point))
```

Emits the IR computing the tile sizes `s1` and `s2` such that:

  - there exists a combination of `n` tiles of size `s1` and `m` tiles of
    size `s2` that covers the entirety of the iteration space `dimension` of
    the target structured op;
  - `s1`, `s2` is less than or equal to `target_size`;
  - `s1` and `s2` are divisible by `divisor.

For example, for a dimension of size 54 with target size 12 and divisor 2,
this can emit the IR computing the tile size 10, used for 3 tiles, and 12,
used for 2 tiles, totally 10*3 + 12*2 = 54. Note that when the divisor does
not divide the original dimension size, it is impossible to compute such
tile sizes. An assertion is emitted to guard against this in the dynamic
case.

Expects the target size and the divisor to be strictly positive. Folds the
IR as much as possible, normally obtaining constant sizes and numbers of
tiles for a statically known dimension.

This does *not* consume the target handle and produces three handles each
pointing to single-result index-typed operations (which may be arithmetic
constant operations) defining the two respective tile sizes and the product
of the first tile size with the number of tiles of that size (useful for
splitting the iteration space).

This operation composes with the regular tiling when applied per-dimension:

```mlir
%sz1, %sz2, %split = structured.multitile_sizes %target
                     { target_size = 10, dimension = 1 }
                   : !transform.any_op, !transform.param<i64>,
                     !transform.param<i64>, !transform.param<i64>
%handles = structured.split %target after %split { dimension = 1 }
            : !transform.any_op, !transform.param<i64>
%low, %high = transform.split_handle %handles : (!transform.any_op)
                  -> (!transform.any_op, !transform.any_op)
%tiled_low, %loop1 = structured.tile_using_for %low [0, %sz1]
                   : (!transform.any_op, !transform.param<i64>)
                  -> (!transform.any_op, !transform.any_op)
%tiled_high, %loop2 = structured.tile_using_for %high [0, %sz2]
                    : (!transform.any_op, !transform.param<i64>)
                   -> (!transform.any_op, !transform.any_op)
%common = merge_handles %tiled_low, %tiled_high : !transform.any_op

%sz3, %sz4, %split = structured.multitile_size %target
                     { target_size = 42, dimension = 0 }
                   : !transform.any_op, !transform.any_op,
                     !transform.any_op, !transform.any_op
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
         : !transform.any_op, !transform.any_op, !transform.any_op
structured.split %common after %splitr { dimension = 0 }
         : !transform.any_op, !transform.any_op
// ...
```

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>target_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>divisor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `low_size` | transform any param type or any handle type |
| `high_size` | transform any param type or any handle type |
| `split_point` | transform any param type or any handle type |

## `transform.structured.pack` (transform::PackOp)

Syntax:

```
operation ::= `transform.structured.pack` $target
              `packed_sizes` `=` custom<DynamicIndexList>($packed_sizes,
              $static_packed_sizes)
              attr-dict
              `:` functional-type(operands, results)
```

Pack a LinalgOp by applying a data tiling transformation on the op and
packing the operands according to the `packed_sizes` specification.

Iterator dimensions are tiled in their canonical order in the op spec.
Operands are packed according to the same canonical order of the op iterator
dimensions.

Specifying a packed size of 0 for an iterator removes it from consideration
for packing.

`linalg.pack` (resp. `linalg.unpack`) operations are inserted for the operands
(resp. results) that need to be packed (resp. unpacked) according to the
`packed_sizes` specification.

### Example

Consider a `linalg.matmul` with indexing maps:
```
  //              M   N   K       M   K
  // affine_map<(d0, d1, d2) -> (d0, d2)>
  //                              K   N
  // affine_map<(d0, d1, d2) -> (d2, d1)>
  //                              M   N
  // affine_map<(d0, d1, d2) -> (d0, d1)>
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(    %C: tensor<?x?xf32>)
```

Specifying packed_sizes [2, 3, 4] results in tiling the iterator dimensions
M, N and K, in this order, in both the op and its operands.
```
  //              M   N   K   m   n   k       M   K   m   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
  //                                          K   N   n   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
  //                                          M   N   m   n
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
  %0 = linalg.generic_representing_some_higher_d_matmul
        ins(%A, %B: tensor<?x?x2x4xf32>, tensor<?x?x4x3xf32>)
       outs(    %C: tensor<?x?x2x3xf32>)
```
In particular, note that the second operand `B` has shape `KxNxnxk` (and not
`KxNxkxn` as one could expect by looking **only** at the operand).

Other layouts can be obtained unsurprisingly from this canonical
transformation by composing the resulting operation with a
`transform.structured.pack_transpose` op.
This composition allows separating concerns and composes better compared
to adding additional permutation attributes to this transform op.

### Return modes

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definite failure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_packed_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `packed_sizes` | variadic of TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |

## `transform.structured.pack_greedily` (transform::PackGreedilyOp)

Syntax:

```
operation ::= `transform.structured.pack_greedily` $target
              oilist(
              `matmul_packed_sizes` `=` custom<DynamicIndexList>($matmul_packed_sizes,
              $static_matmul_packed_sizes)
              (`matmul_padded_sizes_next_multiple_of` `=`
              $matmul_padded_sizes_next_multiple_of^)?
              `matmul_inner_dims_order` `=` $matmul_inner_dims_order
              )
              attr-dict
              `:` functional-type(operands, results)
```

Target a Linalg op and rewrite it into packed LinalgOp form by trying to
infer whether a known suboperation is embedded

Different packing strategies are applied in order, when one applies
successfully, the transform returns:
  1. Matmul packing: Try to infer a matmul operation embedded in the target op.
     Specifically, this looks for 2 parallel dimensions that participate in
     an outer-product and 1 reduction dimension.
     These dimensions are referred as (m, n, k) to match canonical matmul
     terminology.

     The packed sizes for (m, n, k) are specified by `matmul_packed_sizes`
     and the optional `matmul_padded_sizes_next_multiple_of`.
     When an entry `matmul_packed_sizes[i]` is non-0, the corresponding
     dimension is packed by `matmul_packed_sizes[i]`.
     Otherwise, the dimension is merely padded to the next multiple of
     `matmul_padded_sizes_next_multiple_of[i]`.

     `matmul_padded_sizes_next_multiple_of` is optional and is expected to
     either be empty or of size `3`, matching the size of `matmul_packed_sizes`.
     For each individual element of `matmul_packed_sizes` and
     `matmul_padded_sizes_next_multiple_of`, only one of them is allowed to
     be non-zero.

     The ordering of the packed dimensions (mm, nn, kk) is specified by the
     `matmul_inner_dims_order` attribute.

Packing occurs as follows:
  1. Find the dimensions to pack according to the strategy.
  2. The target is converted to linalg.generic form.
  3. An interchange transform is applied to isolate the dimensions to pack as
     the most minor indexing dimensions of the linalg.generic. The most minor
     dimensions are themselves ordered according to `inner_dims_order`.
  4. An elementwise traversal of `matmul_packed_sizes` and
     `matmul_padded_sizes_next_multiple_of` is performed and for each
     dimension `d`, either pack to `matmul_packed_sizes[d]` or pad to the
     `matmul_padded_sizes_next_multiple_of[d]`.
  5. Packing/padding is performed by the amounts determined in step 4. and
     following `inner_dims_order`.

By normalizing the most minor dimensions to `inner_dims_order`, the transform
guarantees that packing immediately generates inner dimensions in a desirable
layout.

Outer dimension layout permutations are not controlled by this transform op
at the moment and can be obtained by composing with the pack_transpose
transformation.

### Return modes

This operation ignores non-Linalg ops and drops them in the return.
It returns the list of packed Linalg ops or the original op when all available
packing strategies failed to apply.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_matmul_packed_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with exactly 3 elements</td></tr>
<tr><td><code>matmul_padded_sizes_next_multiple_of</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with 0 or 3 elements</td></tr>
<tr><td><code>matmul_inner_dims_order</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with exactly 3 elements</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `matmul_packed_sizes` | variadic of TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |

## `transform.structured.pack_transpose` (transform::PackTransposeOp)

Syntax:

```
operation ::= `transform.structured.pack_transpose` $target_pack_or_un_pack_op
              `with_compute_op` `(` $target_linalg_op `)`
              (`outer_perm` `=` $outer_perm^ )?
              (`inner_perm` `=` $inner_perm^ )?
              attr-dict
              `:` functional-type(operands, results)
```

Apply a transposition to a single `linalg.pack` (resp. `linalg.unpack`) and
update the `linalg.generic` op that consumes (resp. produces) the operation.

This transform allows composing a simple `structured.pack` with additional
transpositions to e.g. match the data format required by a specific library
call or ISA instruction.

The transpose spec must specify at least one of `outer_perm` or `inner_perm`
attributes, which will act upon the `outer_dims_perm` or `inner_dims_pos` of
the specified `linalg.pack` or `linalg.unpack` op.

If the `target` of this op is a `linalg.pack` then a new `tensor.empty` will
be created along with transposed versions of the `linalg.pack` and the
consuming `linalg.generic`, which is expected to be the sole consumer.

If the `target` of this op is a `linalg.unpack` then the whole pack / compute
/ unpack chain will be transposed and transposed clones of `linalg.pack`,
the consuming `linalg.generic` and the tail `linalg.pack` will be created.

### Return modes

This operation targets a single `linalg.pack` / `linalg.unpack` op and a
single matching `linalg.generic` that consumes / produces the op. Otherwise,
it produces a silenceableFailure.

This operation may produce a silenceableFailure if the transpose spec is
ill-formed (i.e. `outer_perm` or `inner_perm` are not permutations of the
proper rank) or if the transposition of all involved operations fails for any
reason.

This operation returns 3 handles, one to the transformed LinalgOp, one to
the transformed `linalg.pack` and one to the transformed `linalg.unpack`.
The last handle for `linalg.unpack` is empty if `target_pack_or_unpack_op`
was not itself a `linalg.unpack`.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>outer_perm</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>inner_perm</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target_pack_or_un_pack_op` | TransformHandleTypeInterface instance |
| `target_linalg_op` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |
| `pack_op` | TransformHandleTypeInterface instance |
| `un_pack_op` | TransformHandleTypeInterface instance |

## `transform.structured.pad` (transform::PadOp)

Syntax:

```
operation ::= `transform.structured.pad` $target
              (`pad_to_multiple_of` custom<DynamicIndexList>($pad_to_multiple_of, $static_pad_to_multiple_of)^)?
              (`use_prescribed_tensor_shapes` $use_prescribed_tensor_shapes^)?
              attr-dict
              `:` functional-type(operands, results)
```

Pads the operations pointed to by the target handle using the options
provides as operation attributes. The operation returns a handle to the
padded operation and to the padding operation ("tensor.pad").

To preserve tensor SSA use-def chains, the unpadded result is copied back to
the original destination tensor of the targeted op. The op that copies back
the result can be customized with `copy_back_op`:

* "bufferization.materialize_in_destination" (default)
* "linalg.copy"
* "none" (no copy back)

### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation may produce a definite failure if the padding fails for any
reason.

If all the operations referred to by the `target` handle pad
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>padding_values</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
<tr><td><code>padding_dimensions</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>static_pad_to_multiple_of</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>nofold_flags</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>transpose_paddings</code></td><td>::mlir::ArrayAttr</td><td>array of arrays of i64</td></tr>
<tr><td><code>copy_back_op</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>use_prescribed_tensor_shapes</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `pad_to_multiple_of` | variadic of transform any param type or any handle type |

### Results:

| Result | Description |
| :----: | ----------- |
| `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |
| `copy` | TransformHandleTypeInterface instance |

## `transform.structured.hoist_pad` (transform::HoistPadOp)

Syntax:

```
operation ::= `transform.structured.hoist_pad` $target
              `by` $num_loops `loops`
              (`,` `transpose` `by` $transpose^)?
              attr-dict
              `:` functional-type(operands, results)
```

Hoist the tensor.pad target operation by at most the given number of loops.
Optionally apply the transpose attribute to the inner dimensions.

TODO: In the future, we should consider rewriting as a linalg.pack after
hoisting since this abstraction is now available.
TODO: Maybe also return the linalg.generic transpose created at some point.

### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

If all the operations referred to by the `target` handle padproperly, the
transform succeeds. Otherwise the transform produces a silenceable failure.

The return handle points to only the subset of successfully hoisted
tensor.pad operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_loops</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.structured.hoist_pad.build_packing_loop_nest` (transform::HoistPadBuildPackingLoopNestOp)

Syntax:

```
operation ::= `transform.structured.hoist_pad.build_packing_loop_nest` $target
              `above` $loop
              (`,` `transpose` `by` $transpose^)?
              attr-dict
              `:` functional-type(operands, results)
```

Helper transform used to hoist a tensor.pad target operation. This operation
creates the packing loop nest required by the hoist_pad operation and makes
that functionality available independently.

TODO: In the future, we should consider rewriting as a linalg.pack after
hoisting since this abstraction is now available.

### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

The return handle points to only the subset of successfully created packing
loop nests, which can be empty.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `loop` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `packing_loop` | TransformHandleTypeInterface instance |

## `transform.loop.unroll` (transform::LoopUnrollOp)

_Unrolls the given loop with the given unroll factor_

Syntax:

```
operation ::= `transform.loop.unroll` $target attr-dict `:` type($target)
```

Unrolls each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

### Return modes

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.loop.unroll_and_jam` (transform::LoopUnrollAndJamOp)

_Unrolls and jam the given loop with the given unroll factor_

Syntax:

```
operation ::= `transform.loop.unroll_and_jam` $target attr-dict `:` type($target)
```

Unrolls & jams each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

### Return modes

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.loop.pipeline` (transform::LoopPipelineOp)

_Applies software pipelining to the loop_

Syntax:

```
operation ::= `transform.loop.pipeline` $target attr-dict `:` functional-type(operands, results)
```

Transforms the given loops one by one to achieve software pipelining for
each of them. That is, performs some amount of reads from memory before the
loop rather than inside the loop, the same amount of writes into memory
after the loop, and updates each iteration to read the data for a following
iteration rather than the current one.

The amount is specified by the attributes.

The values read and about to be stored are transferred as loop iteration
arguments. Currently supports memref and vector transfer operations as
memory reads/writes.

### Return modes

This operation ignores non-scf::For ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation pipeline
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.  The return handle points to only the subset of
successfully produced pipelined loops, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>iteration_interval</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>read_latency</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to scf.for operations |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.loop.peel` (transform::LoopPeelOp)

_Peels the first or last iteration of the loop_

Syntax:

```
operation ::= `transform.loop.peel` $target attr-dict `:` functional-type(operands, results)
```

Rewrite the given loop with a main loop and a partial (first or last) loop.
When the `peelFront` option is set to true, the first iteration is peeled off.
Otherwise, updates the given loop so that its step evenly divides its range and puts
the remaining iteration into a separate loop or a conditional.

In the absence of sufficient static information, this op may peel a loop,
even if the step always divides the range evenly at runtime.

### Return modes

This operation ignores non-scf::ForOp ops and drops them in the return.
The op returns two loops, the peeled loop which has trip count divisible
by the step, and the remainder loop.

When `peelFront` is true, the first result (remainder loop) executes all
but the first iteration of the target loop. The second result (peeled
loop) corresponds to the first iteration of the loop which can be
canonicalized away in the following optimizations.

When `peelFront` is false, the first result (peeled loop) is the portion
of the target loop with the highest upper bound that is divisible by the
step. The second result (remainder loop) contains the remaining iterations. 

Note that even though the Payload IR modification may be performed
in-place, this operation consumes the operand handle and produces a new one.

### Return Modes

Produces a definite failure if peeling fails.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>peel_front</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>fail_if_already_divisible</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to scf.for operations |

### Results:

| Result | Description |
| :----: | ----------- |
| `peeled_loop` | TransformHandleTypeInterface instance |
| `remainder_loop` | TransformHandleTypeInterface instance |

## `transform.loop.coalesce` (transform::LoopCoalesceOp)

_Coalesces the perfect loop nest enclosed by a given loop_

Syntax:

```
operation ::= `transform.loop.coalesce` $target attr-dict `:` functional-type($target, $transformed)
```

Given a perfect loop nest identified by the outermost loop,
perform loop coalescing in a bottom-up one-by-one manner.

### Return modes

The return handle points to the coalesced loop if coalescing happens, or
the given input loop if coalescing does not happen.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.loop.fuse_sibling` (transform::LoopFuseSiblingOp)

_Fuse a loop into another loop, assuming the fusion is legal._

Syntax:

```
operation ::= `transform.loop.fuse_sibling` $target `into` $source attr-dict  `:` functional-type(operands, results)
```

Fuses the `target` loop into the `source` loop assuming they are
independent of each other. In the fused loop, the arguments, body and
results of `target` are placed _before_ those of `source`.

For fusion of two `scf.for` loops, the bounds and step size must match. For
fusion of two `scf.forall` loops, the bounds and the mapping must match.
Otherwise a silencable failure is produced.

The `target` and `source` handles must refer to exactly one operation,
otherwise a definite failure is produced. It is the responsibility of the
user to ensure that the `target` and `source` loops are independent of each
other -- this op will only perform rudimentary legality checks.

### Return modes

This operation consumes the `target` and `source` handles and produces the
`fused_loop` handle, which points to the fused loop.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `source` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `fused_loop` | TransformHandleTypeInterface instance |

## `transform.loop.hoist_loop_invariant_subsets` (transform::HoistLoopInvariantSubsetsOp)

_Hoist loop invariant subset ops_

Syntax:

```
operation ::= `transform.loop.hoist_loop_invariant_subsets` $target attr-dict `:` type($target)
```

This transform hoists loop-invariant subset ops out of the targeted
loop-like op. It looks for matching subset extraction/insertion op pairs and
hoists them. The loop body operates on a newly introduced region iter_arg.

Subset ops are hoisted only from the targeted op. If subset ops should be
hoisted from an entire loop nest, this transformation must be applied to
each loop-like op of the loop nest, starting with the innermost loop and
ending with the outermost loop.

Example:
```
%r = scf.for ... iter_args(%t = %a) -> (tensor<?xf32>) {
  %0 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
  %1 = "test.foo"(%0) : (tensor<5xf32>) -> (tensor<5xf32>)
  %2 = tensor.insert_slice %1 into %t[0][5][1]
      : tensor<5xf32> into tensor<?xf32>
  scf.yield %2 : tensor<?xf32>
}
```
Is transformed to:
```
%0 = tensor.extract_slice %a[0][5][1] : tensor<?xf32> to tensor<5xf32>
%new_loop:2 = scf.for ... iter_args(%t = %a, %h = %0) -> (tensor<?xf32>) {
  %1 = "test.foo"(%h) : (tensor<5xf32>) -> (tensor<5xf32>)
  scf.yield %t, %2 : tensor<?xf32>, tensor<5xf32>
}
%r = tensor.insert_slice %new_loop#1 into %new_loop#0
    : tensor<5xf32> into tensor<?xf32>
```

Subset ops are hoisted only if there are no conflicting subset ops. E.g.,
if there were a second overlapping extraction in the above example, no ops
could be hoisted safely.

This transform reads the target handle and modifies the payload. This
transform does not invalidate any handles, but loop-like ops are replaced
with new loop-like ops when a subset op is hoisted. The transform rewriter
updates all handles accordingly.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.structured.vectorize` (transform::VectorizeOp)

Syntax:

```
operation ::= `transform.structured.vectorize` $target oilist(
              `vector_sizes` custom<DynamicIndexList>(
              $vector_sizes,
              $static_vector_sizes,
              $scalable_sizes))
              attr-dict
              `:` type($target)(`,`type($vector_sizes)^)?
```

Vectorize the target ops, which must be Linalg ops.

Use the optional vector sizes to specify exactly what configuration the
vectorizer should use. It will then use masked vectors of the specified
size to enforce this configuration ("masked vectorization"). If no vector
sizes are specified, the vectorizer will infer the shapes to use from the
target Linalg ops ("regular vectorization"). More specifically:

```mlir
transform.structured.vectorize %target vector_sizes [1, 4] : !transform.any_op

## `transform.structured.vectorize_children_and_apply_patterns` (transform::VectorizeChildrenAndApplyPatternsOp)

Syntax:

```
operation ::= `transform.structured.vectorize_children_and_apply_patterns` $target attr-dict `:`functional-type(operands, results)
```

Vectorizes all children contained in the given `target` using the
configuration specified by the attributes of this op. This only vectorizes
structured ops that operate on shaped types and does not vectorize loops or
straight-line. Internally, it applies a set of rewrite patterns, some of
which enable vectorization and some of which clean up the results.
Therefore, it can only be applied to an op with the "isolated from above"
property. This transformation only fails if the entire pattern rewriting
failed, i.e., it does **not** fail when no ops were vectorized.

Finer granularity can be achieved either with the `VectorizeOp` for
individual ops or by outlining the target part of the payload IR into, e.g.,
a function, performing this transformation, and inlining it back.

Note that this transformation invalidates the handles to any payload IR
operation that is contained inside the vectorization target.

This transformation supports the following attributes:
- `vectorize_padding`: a `UnitAttr` to activate the vectorization of
  `tensor.pad` ops. Different pipelines may prefer to lower such ops to
  loops.
- `disable_multi_reduction_to_contract_patterns`: a `UnitAttr` to deactivate
  the rewrite of `vector.multi_reduction` to `vector.contract`. This is
  intended to be used in tests only.
- `disable_transfer_permutation_map_lowering_patterns`: a `UnitAttr` to
  deactivate the rewrite of `vector.transfer` with permutation maps into
  explicit `vector.transpose` operations. This is intended to be used in
  tests only but may be promoted to a first class attribute in the future.

### Return modes:

This operation produces a definite failure if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>vectorize_padding</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>vectorize_nd_extract</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>flatten_1d_depthwise_conv</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>disable_multi_reduction_to_contract_patterns</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>disable_transfer_permutation_map_lowering_patterns</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.structured.hoist_redundant_vector_transfers` (transform::HoistRedundantVectorTransfersOp)

Syntax:

```
operation ::= `transform.structured.hoist_redundant_vector_transfers` $target attr-dict `:` functional-type(operands, results)
```

Hoist vector.transfer_read / vector.transfer_write pairs out of immediately
enclosing scf::ForOp iteratively, if the following conditions are true:
   1. The 2 ops access the same memref with the same indices.
   2. All operands are invariant under the enclosing scf::ForOp.
   3. No uses of the memref either dominate the transfer_read or are
   dominated by the transfer_write (i.e. no aliasing between the write and
   the read across the loop)

WARNING: This hoisting does not model parallelism and is generally incorrect
when used on distributed loops with memref semantics!
TODO: obsolete and should be retired.

### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>verify_non_zero_trip</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.structured.hoist_redundant_vector_broadcasts` (transform::HoistRedundantVectorBroadcastsOp)

Syntax:

```
operation ::= `transform.structured.hoist_redundant_vector_broadcasts` $target attr-dict `:` functional-type(operands, results)
```

Hoist vector.extract / vector.broadcasts pairs out of immediately
enclosing scf::ForOp iteratively.

### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.bufferization.buffer_loop_hoisting` (transform::BufferLoopHoistingOp)

Syntax:

```
operation ::= `transform.bufferization.buffer_loop_hoisting` $target attr-dict `:` type($target)
```

Hoist buffer allocations ("memref.alloc" and "memref.alloca") from loops
within the targeted op. This transform assumes that there are no buffer
deallocation ops in the IR.

This transform reads the `target` handle and modifies the payload.

Traits: `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.memref.multibuffer` (transform::MemRefMultiBufferOp)

_Multibuffers an allocation_

Syntax:

```
operation ::= `transform.memref.multibuffer` $target attr-dict `:` functional-type(operands, results)
```

Transformation to do multi-buffering/array expansion to remove
dependencies on the temporary allocation between consecutive loop
iterations. This transform expands the size of an allocation by
a given multiplicative factor and fixes up any users of the
multibuffered allocation.
If skip analysis is not set the transformation will only apply
if it can prove that there is no data being carried across loop
iterations.

### Return modes

This operation returns the new allocation if multi-buffering
succeeds, and failure otherwise.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
<tr><td><code>skip_analysis</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to memref.alloc operations |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.memref.erase_dead_alloc_and_stores` (transform::MemRefEraseDeadAllocAndStoresOp)

Syntax:

```
operation ::= `transform.memref.erase_dead_alloc_and_stores` $target attr-dict `:` functional-type($target, results)
```

This applies memory optimization on memref. In particular it does store to
load forwarding, dead store elimination and dead alloc/alloca elimination.

### Return modes

This operation applies a set of memory optimization on the whole region of
the operand.

The transformation does not consume the target handle. It modifies the
payload. Dead allocations, loads and stores are silently dropped from all
mappings.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.memref.make_loop_independent` (transform::MemRefMakeLoopIndependentOp)

Syntax:

```
operation ::= `transform.memref.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed)
```

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`memref.subview` inside the loop.

Currently supported operations are:
- memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
  followed by a memref.subview.

### Return modes

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:
- memref.alloca: The returned handle points to the memref.subview op.

This transform op consumes the target handle and produces a result handle.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_loops</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |

## `transform.apply_patterns` (transform::ApplyPatternsOp)

_Greedily applies patterns to the body of the targeted op_

Syntax:

```
operation ::= `transform.apply_patterns` `to` $target $patterns attr-dict `:` type($target)
```

This transform greedily applies the specified patterns to the body of the
targeted op until a fixpoint was reached. Patterns are not applied to the
targeted op itself.

The patterns that should be applied are specified in the graph region of
this op. They must implement the `PatternDescriptorOpInterface`. The order
in which patterns are applied is unspecified; i.e., the ordering of ops in
the region of this op is irrelevant.

If `apple_cse` is set, the greedy pattern rewrite is interleaved with
common subexpression elimination (CSE): both are repeated until a fixpoint
is reached.

This transform only reads the target handle and modifies the payload. If a
pattern erases or replaces a tracked op, the mapping is updated accordingly.

Only replacements via `RewriterBase::replaceOp` or `replaceOpWithNewOp` are
considered "payload op replacements". Furthermore, only if the replacement
values are defined by the same op and that op has the same type as the
original op, the mapping is updated. Otherwise, this transform produces a
silenceable failure. More details can be found at the documentation site of
`TrackingListener`.

This transform also produces a silenceable failure if the pattern
application did not converge within the default number of
iterations/rewrites of the greedy pattern rewrite driver.

Traits: `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>apply_cse</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>max_iterations</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>max_num_rewrites</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.apply_cse` (transform::ApplyCommonSubexpressionEliminationOp)

_Eliminate common subexpressions in the body of the target op_

Syntax:

```
operation ::= `transform.apply_cse` `to` $target attr-dict `:` type($target)
```

This transform applies common subexpression elimination (CSE) to the body
of the targeted op.

This transform reads the target handle and modifies the payload. Existing
handles to operations inside of the targeted op are retained and updated if
necessary. Note that this can lead to situations where a handle, that was
previously mapped to multiple distinct (but equivalent) operations, is now
mapped to the same operation multiple times.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.apply_dce` (transform::ApplyDeadCodeEliminationOp)

_Eliminate dead operations in the body of the target op_

Syntax:

```
operation ::= `transform.apply_dce` `to` $target attr-dict `:` type($target)
```

This transform applies dead code elimination (DCE) to the body of the
targeted op.

Note: "transform.apply_patterns" with an empty region can also be used to
remove dead ops. However, that op applies additional simplifications such as
op folding and region simplification.

This transform reads the target handle and modifies the payload. Note that
this transform may silently remove payload ops from handles.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.apply_licm` (transform::ApplyLoopInvariantCodeMotionOp)

_Move loop-invariant code out of a loop-like op_

Syntax:

```
operation ::= `transform.apply_licm` `to` $target attr-dict `:` type($target)
```

This transform moves side-effect free, loop invariant code out of the
targeted loop-like op. The targeted op must implement the
`LoopLikeOpInterface`.

Note: To move invariant ops from a loop nest, this transform must be applied
to each loop of the loop nest, starting with the inner-most loop.

This transform reads the target handle and modifies the payload.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.affine.simplify_bounded_affine_ops` (transform::SimplifyBoundedAffineOpsOp)

Syntax:

```
operation ::= `transform.affine.simplify_bounded_affine_ops` $target `with` `[` ($bounded_values^ `:` type($bounded_values))? `]`
              `within` $lower_bounds `and` $upper_bounds attr-dict
              `:` type($target)
```

Simplify the targeted affine.min / affine.max ops given the supplied
lower and upper bounds for values that may be used as target op operands.

Example:
```
%0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1
%1 = transform.structured.match ops{["gpu.lane_id"]} in %arg1
transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [32]

// Multiple bounds can be specified.
transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 5] and [32, 50]
```

Bounded op handles (`%1` and `%2) must be mapped to ops that have a single
result of index type. The sets of target ops and bounded ops must not
overlap.

### Return modes

Target ops must be affine.min or affine.max ops. This transform consumes the
target handle and does not produce any handle. It reads the bounded op
handles.

TODO: Support affine.apply targets.
TODO: Allow mixed PDL_Operation/int64_t for lower_bounds and upper_bounds.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lower_bounds</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>upper_bounds</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `bounded_values` | variadic of TransformHandleTypeInterface instance |

## `transform.affine.simplify_min_max_affine_ops` (transform::SimplifyMinMaxAffineOpsOp)

Syntax:

```
operation ::= `transform.affine.simplify_min_max_affine_ops` $target attr-dict `:` type($target)
```

Simplify the targeted `affine.min` / `affine.max` ops using the
`mlir::affine::simplifyAffineMinMaxOps` transform.

Example:
```
%0 = transform.structured.match ops{["affine.max"]} in %arg1
transform.affine.simplify_min_max_affine_ops %0 : !transform.any_op
```

### Return modes

This transform consumes the target handle and does not produce any results.
This transforms definitely fails if any of the targeted operations is not an
`affine.min` or `affine.max` operation, or if the canonicalization patterns
failed to converge.
This transform silently fails if none of the operations were simplified.
Otherwise, it succeeds.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

## `transform.apply_patterns.linalg.tiling_canonicalization` (transform::ApplyTilingCanonicalizationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.tiling_canonicalization` attr-dict
```

Collects canonicalization patterns relevant to apply after tiling patterns.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` (transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` attr-dict
```

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via rank-reducing slices.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` (transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` attr-dict
```

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via reassociative reshape ops.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.scf.for_loop_canonicalization` (transform::ApplyForLoopCanonicalizationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.scf.for_loop_canonicalization` attr-dict
```

Collects patterns for canonicalizing operations inside SCF loop bodies.
At the moment, only affine.min/max computations with iteration variables,
loop bounds and loop steps are canonicalized.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.vector.reduction_to_contract` (transform::ApplyVectorReductionToContractPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.reduction_to_contract` attr-dict
```

Apply opt-in patterns that convert reductions to contract:
  - MultiReduceToContract
  - CombineContractBroadcast
  - CombineContractABTranspose
  - CombineContractResultTranspose
  - ReorderElementwiseOpsOnTranspose
  - ReorderElementwiseOpsOnBroadcast
  - ReorderCastOpsOnBroadcast

These patterns have the effect of rewriting a vector.multi_reduce into a
vector.contract.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.vector.transfer_permutation_patterns` (transform::ApplyTransferPermutationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.transfer_permutation_patterns` attr-dict
```

Apply opt-in vector transfer permutation patterns that include:
  - TransferReadPermutationLowering
  - TransferWritePermutationLowering
  - TransferOpReduceRank
  - TransferWriteNonPermutationLowering

These patterns have the effect of rewriting a vector.transfer with an
arbitrary permutation_map to a vector.transfer with a permutation_map that
is a minor identity followed by a vector.transpose.

In other words, this makes the vector.transfer contiguous on the most minor
dimensions and materializes the permutation_map as a vector.transpose.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.vector.lower_contraction` (transform::ApplyLowerContractionPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_contraction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
```

Indicates that vector contraction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowering_strategy</code></td><td>::mlir::vector::VectorContractLoweringAttr</td><td>control the lowering of `vector.contract` operations.</td></tr>
</table>

## `transform.apply_patterns.vector.lower_outerproduct` (transform::ApplyLowerOuterProductPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_outerproduct` attr-dict
```

Indicates that the vector outerproduct operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.vector.lower_transfer` (transform::ApplyLowerTransferPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_transfer` (`max_transfer_rank` `=` $max_transfer_rank^)? attr-dict
```

Indicates that vector transfer operations should be lowered to finer-grained
vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>max_transfer_rank</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

## `transform.apply_patterns.vector.lower_transpose` (transform::ApplyLowerTransposePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_transpose` oilist (
              `lowering_strategy` `=` $lowering_strategy
              | `avx2_lowering_strategy` `=` $avx2_lowering_strategy
              )
              attr-dict
```

Indicates that vector transpose-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowering_strategy</code></td><td>::mlir::vector::VectorTransposeLoweringAttr</td><td>control the lowering of `vector.transpose` operations.</td></tr>
<tr><td><code>avx2_lowering_strategy</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

## `transform.apply_patterns.vector.lower_shape_cast` (transform::ApplyLowerShapeCastPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_shape_cast` attr-dict
```

Indicates that vector shape_cast operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

## `transform.apply_patterns.vector.sink_ops` (transform::ApplySinkVectorPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.sink_ops` attr-dict
```

Patterns that remove redundant Vector Ops by re-ordering them with
e.g. elementwise Ops.

Example:
```
%at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%r = arith.addf %at, %bt : vector<2x4xf32>
```
gets converted to:
```
%0 = arith.addf %a, %b : vector<4x2xf32>
%r = vector.transpose %0, [1, 0] : vector<2x4xf32>
```
At the moment, these patterns are limited to vector.broadcast,
vector.transpose and vector.extract.

Interfaces: `PatternDescriptorOpInterface`
