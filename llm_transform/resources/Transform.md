# Transform Dialect

Fine-grain transformation control dialect.

## Overview

This dialect provides operations that can be used to control transformation
of the IR using a different portion of the IR. It refers to the IR being
transformed as payload IR, and to the IR guiding the transformation as
transform IR.

The main use case for this dialect is orchestrating fine-grain transformations
on individual IR objects (operations or values) or sets thereof. For example, it
may involve finding loop-like operations with specific properties (e.g., large
size) in the payload IR, applying loop tiling to those and only those
operations, and then applying loop unrolling to the inner loops produced by the
previous transformations. As such, it is not intended as a replacement for the
pass infrastructure, nor for the pattern rewriting infrastructure. In the most
common case, the transform IR will be processed and applied to the payload IR by
a pass. Transformations expressed by the Transform dialect may be implemented
using the pattern infrastructure or any other relevant MLIR component.

The following IR gives a rough idea of what the operations in this dialect
may look like without using actually existing operations:

```mlir
%0 = transform.loop.find { size > 42 } : !transform.interface<tileable>
%1 = transform.compute_trailing_tile_size %0 : !transform.param<index>
%2:2 = transform.loop.tile %0 tile_sizes(1, 4, %1)
      : (!transform.interface<tileable>)
     -> (!transform.op<loop>, !transform.op<loop>)
%3 = transform.get_op_result [0] %2#0 : !transform.any_value
transform.assign_to_fast_memory %3
transform.loop.unroll %1#1 : !transform.op<loop>
```

The values used in the Transform dialect may correspond to:

  * sets of operations in the payload IR;

  * sets of values in the payload IR;

  * sets of parameters (attributes) known at the execution time of the
    transform dialect.

The former two kinds of values are also referred to as operation and value
*handles*, respectively. In the example above, `%0` corresponds to the set of
loops found in the payload IR that satisfy the condition, and `%2` correspond to
groups of outer and inner loops, respectively, produced by the tiling
transformation. `%3` corresponds to a set of values that are produced by the
outer loops after tiling. `%1` corresponds to a list of tile sizes selected for
each of the operations that `%0` corresponds to.

An operation handle such as `%0` may be associated with multiple payload
operations. This is conceptually a set of operations and no assumptions should
be made about the order of ops unless specified otherwise by the operation.
Similarly, a value handle such as `%3` may be associated with a set of payload
IR values. Transform dialect operations may take as operands and produce an
arbitrary combination of values representing handles and parameters. Most
Transform IR ops support operand values that are mapped to multiple payload
objects. They usually apply the respective transformation for every mapped
object ("batched execution"). Deviations from this convention are described in
the documentation of Transform IR ops.

Parameters, such as `%1` in the above example, have two logical roles in
transform IR. In parameter based control, they carry the values needed to
execute the explicit control defined by the transforms, for example:

```mlir
%0 = transform.match.structured.rank %linalg_op_handle : !transform.param<index>
%1 = transform.param.constant 3 : i32 -> !transform.param<index>
transform.execute_if_cmpi eq %0, %1 : !transform.param<index>, !transform.param<index>
// Some nested body of transform ops
```

Alternatively, parameters can associate with the payload IR where the specific
value at execution time has no bearing on the execution of the transform IR. In
other words, parameters can either associate with the transform IR or the
payload IR.  Note that it is generally discouraged to use parameters containing
arbitrary attributes within transform control. Parameter based control should
try to be explicitly typed when possible.

The transform IR values have transform IR types, which should implement exactly one of:

  * [TransformHandleTypeInterface](#transformhandletypeinterface-transformhandletypeinterface),

  * [TransformValueHandleTypeInterface](#transformvaluehandletypeinterface-transformvaluehandletypeinterface),

  * [TransformParamTypeInterface](#transformparamtypeinterface-transformparamtypeinterface).

The goal of these type interfaces, beyond providing a common base for accepted
types, is to verify the properties of the associated objects. For example, a
handle type interface implementation may check whether all associated payload IR
operations implement the "TileableOp" interface or have a specific "loop" kind.
Similarly, a value handle type interface implementation may check if the
associated payload IR values are block arguments or have a specific type, or a
parameter type interface may check whether the associated attributes contain
non-negative integer values. These properties are used to statically indicate
 pre- and post-conditions of a transformation connected to a Transform dialect
operation. The conditions are verified when payload objects operations are first
associated with a transform handle. By convention, Transform dialect operations
are expected to indicate narrow preconditions for their operands by enforcing
operand type constraints in the their definitions and verifiers. On the
contrary, operations are expected to have few constraints on their results.
Specific instances of a transform operation can then be created with a more
restricted result type than the constraint in the operation (e.g., the "find"
operation only constrains the result type to be a transform IR type while its
concrete instance can have a type with stricter constraints such as implementing
the "tilable" interface). The verification will then happen at transform
execution time. This approach allows one to capture payload IR operation
properties in the transform IR without resorting to excessive use of type casts
or coupling dialect extensions between themselves. It is a trade-off between
verbosity/complexity and static hardening, which can be revised in the future.

Overall, Transform IR ops are expected to be contained in a single top-level
op. Such top-level ops specify how to apply the transformations described
by the operations they contain, e.g., `transform.sequence` executes
transformations one by one and fails if any of them fails. Such ops are
expected to have the `PossibleTopLevelTransformOpTrait` and may be used
without arguments.

A program transformation expressed using the Transform dialect can be
programmatically triggered by calling:

```c++
LogicalResult transform::applyTransforms(
    Operation *payloadRoot,
    const RaggedArray<transform::MappedValue> &extraMappings,
    TransformOpInterface transform,
    const TransformOptions &options);
```

that applies the transformations specified by the top-level `transform` to
payload IR contained in `payloadRoot`. The payload root operation will be
associated with the first argument of the entry block of the top-level transform
op. This block may have additional arguments, handles or parameters. They will
be associated with values provided as `extraMappings`. The call will report an
error and return if the wrong number of mappings is provided.

## Dialect Extension Mechanism

This dialect is designed to be extensible, that is, clients of this dialect
are allowed to inject additional operations into this dialect using the
`TransformDialectExtension` mechanism. This allows the dialect to avoid a
dependency on the implementation of the transformation as well as to avoid
introducing dialect-specific transform dialects. In the example above,
the operations may have been injected by a notional `loop` dialect rather
than defined in this dialect, hence the common prefix.

It is recommended to prefix injected operations with one or several
dot-separated words that indicate which extension adds them. For
dialect-specific transformations, the prefix is naturally the name of the
dialect, e.g., `transform.affine.reschedule`. For dialect-agnostic
transformations (typically implemented using interfaces), the prefix may
be derived from the interface name or from a common concept, e.g.,
`transform.loop.tile` may apply to any loop-like operation that implements
`TileableOpInterface`. The C++ classes for the dialect extension should
include the prefix in their name, e.g., `AffineTransformDialectExtension` or
`LoopTransformDialectExtension` in the cases above. Unprefixed operation
names are reserved for ops defined directly in the Transform dialect.

Operations injected into the dialect must:

  * Implement the `TransformOpInterface` to execute the corresponding
    transformation on the payload IR.

  * Implement the `MemoryEffectsOpInterface` to annotate the effects of
    the transform IR operation on the payload IR as well as on the mapping
    between transform IR values and payload IR operations. See below for
    the description of available effects.

The presence of interface implementations is checked at runtime when the
dialect is loaded to allow for those implementations to be supplied by
separate dialect extensions if desired.

Similarly to operations, additional types can be injected into the dialect using
the same extension mechanism. The types must:

  * Implement exactly one of `TransformHandleTypeInterface`,
    `TransformValueHandleTypeInterface`, `TransformParamTypeInterface`.

## Side Effects

The Transform dialect relies on MLIR side effect modelling to enable
optimization of the transform IR. More specifically, it provides several
side effect resource objects and expects operations to describe their
effects on these resources.

  * `TransformMappingResource` - side effect resource corresponding to the
    mapping between transform IR values and payload IR operations.

    - An `Allocate` effect from this resource means creating a new mapping
      entry, it is always accompanied by a `Write` effect.

    - A `Read` effect from this resource means accessing the mapping.

    - A `Free` effect on this resource indicates the removal of the mapping
      entry, typically after a transformation that modifies the payload IR
      operations associated with one of the transform IR operation's
      operands. It is always accompanied by a `Read` effect.

  * `PayloadIRResource` - side effect resource corresponding to the payload
    IR itself.

    - A `Read` effect from this resource means accessing the payload IR.

    - A `Write` effect on this resource means mutating the payload IR. It is
      almost always accompanied by a `Read`.

The typical flow of values in the transform IR is as follows. Most
operations produce new transform IR values and immediately associate them
with a list of payload IR operations. This corresponds to `Allocate` and
`Write` effects on the `TransformMappingResource`, and often requires at
least a `Read` effect on the `PayloadIRResource`. Transform operations that
only inspect the payload IR to produce new handles are usually limited to
these effects on their operands. Transform operations that mutate the
payload IR are thought to _consume_ the handles provided as operands, that
is have the `Read` and `Free` effects on them. As with the usual memory
effects, using a value after it was freed is incorrect. In case of the
transform IR, this value is likely associated with payload IR operations
that were modified or even removed by the transformation, so it is
meaningless to refer to them. When further transformations are desired, the
transform operations can return _new_ handles that can be read or consumed
by subsequent operations.

## Execution Model

The transformation starts at the user-specified top-level transform IR
operation and applies to some user-specified payload IR scope, identified by
the payload IR op that contains the IR to transform. It is the
responsibility of the user to properly select the scope and/or to avoid the
transformations to modify the IR outside of the given scope. The top-level
transform IR operation may contain further transform operations and execute
them in the desired order.

Transformation application functions produce a tri-state status:

- success;
- recoverable (silenceable) failure;
- irrecoverable failure.

Transformation container operations may intercept recoverable failures and
perform the required recovery steps thus succeeding themselves. On
the other hand, they must propagate irrecoverable failures. For such
failures, the diagnostics are emitted immediately whereas their emission is
postponed for recoverable failures. Transformation container operations may
also fail to recover from a theoretically recoverable failure, in which case
they can either propagate it to their parent or emit the diagnostic and turn
the failure into an irrecoverable one. A recoverable failure produced by
applying the top-level transform IR operation is considered irrecoverable.

Transformation container operations are allowed to "step over" some nested
operations if the application of some previous operation produced a failure.
This can be conceptually thought of as having a global "recoverable error
register" that is read/write accessed by each transform operation as a side
effect. The transformation is skipped if the register already contains an
error description, and the control flow proceeds to the following operation.

Note that a silenceable failure, if emitted, is a compiler _error_ rather
than a warning. Transformations are expected to produce silenceable failures
if they haven't yet modified the payload IR, i.e. when reporting a
precondition failure, and an irrecoverable failure when they modified the IR
in a way that is contrary to the semantics of the transform operation or
would fail a postcondition. Some "navigation" operations that identify
payload IR targets for the following transformation may have a conceptual
"failure to match" that is considered a successful execution in the
execution model but results in handles associated with empty payload IR
operation lists.

## Handle Invalidation

The execution model of the Transform dialect allows a payload IR operation to be
associated with _multiple_ handles as well as nested payload IR operations to be
associated with different handles. Similarly, a payload IR value may be
associated with multiple transform IR value handles. When a transform IR
operation consumes a handle, it usually indicates that the corresponding payload
IR object was destroyed and should no longer be referenced. Transform IR handles
that _may_ be pointing to an erased payload IR object are _invalidated_. The
mere presence of an invalidated handle in the transform IR is not a problem, but
_using_ it results in undefined behavior. Invalidated handles can be thought of
as dangling pointers. Note that the _entire_ handle is invalidated, even if some
of the payload IR objects associated with it remain live.

The following handle invalidation rules apply.

  * When an operation handle is consumed, are invalidated:

    - operation handles associated with one of the payload operations that the
      consumed handle is associated with;

    - operation handles associated with one of the operations _nested_ in the
      payload operations described above;

    - value handles associated with any result of any operation described above;

    - value handles associated with any argument of a block contained in a
      region attached to any operation described above.

  * When a value handle is consumed, are invalidated:

    - operation handles associated with payload operations that produce as
      result any value associated with the consumed handle (when the associated
      is an operation result);

    - operation handles associated with payload operations _nested_ in the
      payload operations described above;

    - operation handles associated with payload operations (recursively)
      _contained_ in the block that defines as argument any value associated
      with the consumed handle (when the associated value is a block argument);
      note that the adjacent blocks are not affected;

    - value handles associated with any result of any operation described above,
      including all results of the operation defining as result the value
      associated with the consumed handle;

    - value handles associated with any argument of a block contained in a
      region attached to any operation described above.

More intuitively, consuming a handle invalidates any handle that may be pointing
to an object defined or contained in the payload IR subtree rooted at the
closest operation or block.

The Transform dialect infrastructure has the capability of checking whether
the transform IR op operand is invalidated before applying the
transformation. However, such a check is computationally expensive and
must be enabled explicitly through `TransformOptions`. Additionally, the
`transform-dialect-check-uses` pass emits warnings when a handle may be used
after it has been consumed, but does so abstractly, without processing the
payload IR.

Values associated with parameters (non-handles) cannot be invalidated.

## Intended Use and Integrations

The transformation control infrastructure provided by this dialect is
positioned roughly between rewrite patterns and passes. A transformation
that is executed by a transform operation is likely to be sufficiently
complex to require at least a set of patterns to be implemented. It is also
expected to be more focused than a pass: a pass typically applies identical
transformations everywhere in the IR, a transform dialect-controlled
transformation would apply to a small subset of operations selected, e.g.,
by a pattern-matching operation or generated by a previous transformation.
It is discouraged, although technically possible, to run a pass pipeline as
part of the transform op implementation.

One of the main scenarios for using this dialect is fine-grain chaining of
transformations. For example, a loop-like operation may see its iteration
domain split into two parts, implemented as separate loops (transformation
known as index-set splitting), each of which is then transformed differently
(e.g., the first loop is tiled and the second unrolled) with the necessary
enabling and cleanup patterns around the main transformation:

```mlir
// <generate %loop, e.g., by pattern-matching>
// ...
%parts:2 = transform.loop.split %loop { upper_bound_divisible_by = 8 }
transform.loop.tile %parts#0 { tile_sizes = [8] }
transform.loop.unroll %parts#1 { full }
```

This composition would have been difficult to implement as separate passes
since the hypothetical "tiling" and "unrolling" pass would need to somehow
differentiate between the parts of the loop produced by the previous pass
(both are the same operation, and it is likely undesirable to pollute the
operation with pass-specific information). Implementing passes that run the
combined transformation would have run into the combinatorial explosion
issue due to multiple possible transform compositions or into the need for
deep pass parameterization, the ultimate form of which is an ad-hoc dialect
to specify which transformations the pass should run. The transform dialect
provides a uniform, extensible mechanism for controlling transformations in
such cases.

The Transform dialect is supposed to be consumed by an "interpreter" pass
that drives the application of transformations. To ensure extensibility and
composability, this pass is not expected to actually perform the
transformations specified by the ops. Instead, the transformations are
implemented by the transform ops themselves via `TransformOpInterface`. The
pass serves as the entry point, handles the flow of transform operations and
takes care of bookkeeping. As such, the Transform dialect does not provide
the interpreter pass. Instead, it provides a set of utilities that can be
used by clients to define their own interpreter passes or as part of a more
complex pass. For example, the mapping between values in the transform IR
and operations in the payload IR, or the function that applies the
transformations specified by ops in the given block sequentially. Note that
a transform op may have regions with further transform ops in them, with
the op itself guiding how to dispatch the transformation control flow to
those regions. This approach allows clients to decide on the relative
location of the transform IR in their input (e.g., nested modules, separate
modules, optional regions to certain operations, etc.), register additional
transform operations and perform client-specific bookkeeping.

## Effects on the Infrastructure

Although scoped to a single dialect, this functionality conceptually belongs
to the MLIR infrastructure. It aims to be minimally intrusive and opt-in.

Some infrastructural components may grow extra functionality to support the
transform dialect. In particular, the pattern infrastructure may add extra
hooks to identify the "main results" of a transformation or to notify
external observers about changes made to certain operations. These are not
expected to affect the existing uses of the infrastructure.

For the sake of reusability, transformations should be implemented as
utility functions that are called from the interface methods of transform
ops rather than having the methods directly act on the payload IR.

## Type Definitions

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

### AffineMapParamType

Syntax: `!transform.affine_map`

Transform IR parameter value that can be associated with a list of affine
map attributes.


### AnyOpType

Syntax: `!transform.any_op`

Transform IR handle that can be associated with a list of arbitrary
Payload IR operations.


### AnyParamType

Syntax: `!transform.any_param`

Transform IR value that can be associated with a list of parameters
of any type.


### AnyValueType

Syntax: `!transform.any_value`

Transform IR value that can be associated with a list of Payload IR values.


### OperationType

Syntax:

```
!transform.op<
  ::llvm::StringRef   # operation_name
>
```

Transform IR handle that can be associated with a list of Payload IR
operations with the specified operation name.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| operation_name | `::llvm::StringRef` | Name of the allowed payload operation |

### ParamType

Syntax:

```
!transform.param<
  ::mlir::Type   # type
>
```

Transform IR value that can be associated with the list of parameters
of the given type. Types are currently limited to integers, but may be
extended in the future to other types values of which can be contained
in attributes.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| type | `::mlir::Type` | Underlying type of the parameter |

### TypeParamType

Syntax: `!transform.type`

Transform IR parameter value that can be associated with a list of type
attributes.


## Core Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Transform/IR/TransformOps.td)

### `transform.alternatives` (transform::AlternativesOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `scope` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of TransformHandleTypeInterface instance |


### `transform.annotate` (transform::AnnotateOp)

_Annotates the target operation with an attribute by name_

Syntax:

```
operation ::= `transform.annotate` $target $name attr-dict (`=` $param^)?`:` type($target) (`,` type($param)^)?
```

Adds an attribute with the given `name` to the `target` operation. An
optional `param` handle can be provided to give the attribute a specific
value, else a UnitAttr is added. A single attribute will be broadcasted to
all target operations, otherwise the attributes will be mapped 1:1 based on
the order within the handles.

Produces a silenceable failure if the length of the parameter payload does
not match the length of the target payload. Does not consume the provided
handles.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `param` | TransformParamTypeInterface instance |


### `transform.apply_patterns.canonicalization` (transform::ApplyCanonicalizationPatternsOp)

_Populates canonicalization patterns_

Syntax:

```
operation ::= `transform.apply_patterns.canonicalization` attr-dict
```

This op populates all canonicalization patterns of all loaded dialects in
an `apply_patterns` transform.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_cse` (transform::ApplyCommonSubexpressionEliminationOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.apply_conversion_patterns` (transform::ApplyConversionPatternsOp)

_Applies conversion patterns to the body of the targeted op_

Syntax:

```
operation ::= `transform.apply_conversion_patterns` `to` $target $patterns
              (`with` `type_converter` $default_type_converter_region^)?
              attr-dict `:` type($target)
```

This transform applies the specified conversion patterns to the targeted op
and all nested ops. By default, this transform applies a "full" dialect
conversion. If the `partial_conversion` unit attribute is present, this
transform applies a partial dialect conversion.

The patterns that should be applied are specified in the first graph region
of this op. They must implement the
`ConversionPatternDescriptorOpInterface`. The order in which patterns are
applied is unspecified; i.e., the ordering of ops in the region of this op
is irrelevant.

The second, optional graph region contains exactly one op that specifies
default type converter that should be used with this dialect conversion. If
provided, this op must implement the `TypeConverterBuilderOpInterface`.
Type converters are a property of conversion patterns: each conversion
pattern stores the type converter that should be used in its C++ class. Each
conversion pattern descriptor can optionally specify a type converter in its
`getTypeConverter` interface method. If no type converter is specified in
this method, the default type converter of the dialect conversion is used.
Default type converters are useful if the same type converter should be used
for multiple sets of conversion patterns. (Patterns that should not use this
default type converter specify their own type converter.)

The `legal_ops`, `illegal_ops`, `legal_dialects`, `illegal_dialects`
attributes specify the conversion target.

This transform modifies the payload. By default, it consumes the `target`
handle. It does not produce any handles.

If the `preserve_handles` attribute is set, this transform does not consume
the `target` handle and instead updates handles based on notifications from
a tracking listener that is attached to the dialect conversion, similar to
`transform.apply_patterns`. Only replacements via `RewriterBase::replaceOp`
or `replaceOpWithNewOp` are considered "payload op replacements". In
contrast to `transform.apply_patterns`, we allow replacement ops even if the
op name has changed. This is because conversion patterns are expected to
lower ops to different ops (from a different dialect). More details can be
found at the documentation site of `TrackingListener`.

This transform produces a silenceable failure if the dialect conversion was
unsuccessful or the tracking listener failed to find a replacement op.

Traits: `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>legal_ops</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>illegal_ops</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>legal_dialects</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>illegal_dialects</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>partial_conversion</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>preserve_handles</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.apply_dce` (transform::ApplyDeadCodeEliminationOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.apply_licm` (transform::ApplyLoopInvariantCodeMotionOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.apply_patterns` (transform::ApplyPatternsOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>apply_cse</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>max_iterations</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>max_num_rewrites</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.apply_registered_pass` (transform::ApplyRegisteredPassOp)

_Applies the specified registered pass or pass pipeline_

Syntax:

```
operation ::= `transform.apply_registered_pass` $pass_name (`with` `options` `=`
              custom<ApplyRegisteredPassOptions>($options, $dynamic_options)^)?
              `to` $target attr-dict `:` functional-type(operands, results)
```

This transform applies the specified pass or pass pipeline to the targeted
ops. The name of the pass/pipeline is specified as a string attribute, as
set during pass/pipeline registration.

Optionally, pass options may be specified via a DictionaryAttr. This
dictionary is converted to a string -- formatted `key=value ...` -- which
is expected to be in the exact format used by the pass on the commandline.
Values are either attributes or (SSA-values of) Transform Dialect params.
For example:

```mlir
transform.apply_registered_pass "canonicalize"
    with options = { "top-down" = false,
                     "max-iterations" = %max_iter,
                     "test-convergence" = true,
                     "max-num-rewrites" = %max_rewrites }
    to %module
: (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
```

Options' values which are `ArrayAttr`s are converted to comma-separated
lists of options. Likewise for params which associate multiple values.

This op first looks for a pass pipeline with the specified name. If no such
pipeline exists, it looks for a pass with the specified name. If no such
pass exists either, this op fails definitely.

This transform consumes the target handle and produces a new handle that is
mapped to the same op. Passes are not allowed to remove/modify the operation
that they operate on, so the target op is guaranteed to still exist. The
target handle is invalidated because a pass may arbitrarily modify the body
of targeted ops.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pass_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>options</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_options` | variadic of TransformParamTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.apply_conversion_patterns.dialect_to_llvm` (transform::ApplyToLLVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.dialect_to_llvm` $dialect_name attr-dict
```

Collects patterns that convert ops from the specified dialect to LLVM
dialect ops. These patterns require an "LLVMTypeConverter".

Note: Only dialects that implement the `ConvertToLLVMPatternInterface` are
supported. Any conversion target modifications by interface implementations
are currently ignored. The conversion target is fully specified by the
enclosing "apply_conversion_patterns" op.

Interfaces: `ConversionPatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dialect_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>


### `transform.cast` (transform::CastOp)

Syntax:

```
operation ::= `transform.cast` $input attr-dict `:` type($input) `to` type($output)
```

Traits: `TransformEachOpTrait`

Interfaces: `CastOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | TransformHandleTypeInterface instance |


### `transform.collect_matching` (transform::CollectMatchingOp)

_Collects all payload ops that match the given named matcher_

Syntax:

```
operation ::= `transform.collect_matching` $matcher `in` $root attr-dict `:` functional-type($root, $results)
```

Collects operations or other payload IR objects nested under `root`
(inclusive) that match the given matcher expressed as a named sequence. The
matcher sequence must accept exactly one argument that it is not allowed to
modify. It must yield as many values as this op has results. Each of the
yielded values must be associated with exactly one payload object. If any
operation in the matcher sequence produces a silenceable failure, the
matcher advances to the next payload operation in the walk order without
finishing the sequence.

The i-th result of this operation is constructed by concatenating the i-th
yielded payload IR objects of all successful matcher sequence applications.
All results are guaranteed to be mapped to the same number of payload IR
objects.

The operation succeeds unless the matcher sequence produced a definite
failure for any invocation.

Interfaces: `MemoryEffectOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>matcher</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of any transform handle or parameter |


### `transform.foreach_match` (transform::ForeachMatchOp)

_Applies named sequences when a named matcher succeeds_

Syntax:

```
operation ::= `transform.foreach_match` oilist( `restrict_root` $restrict_root
              | `flatten_results` $flatten_results
              )
              `in`
              $root (`,` $forwarded_inputs^)?
              custom<ForeachMatchSymbols>($matchers, $actions)
              attr-dict
              `:` functional-type(operands, results)
```

Given a pair of co-indexed lists of transform dialect symbols (such as
`transform.named_sequence`), walks the payload IR associated with the root
handle and interprets the symbols as matcher/action pairs by applying the
body of the corresponding symbol definition. The symbol from the first list
is the matcher part: if it results in a silenceable error, the error is
silenced and the next matcher is attempted. Definite failures from any
matcher stop the application immediately and are propagated unconditionally.
If none of the matchers succeeds, the next payload operation in walk order
(post-order at the moment of writing, double check `Operation::walk`) is
matched. If a matcher succeeds, the co-indexed action symbol is applied and
the following matchers are not applied to the same payload operation. If the
action succeeds, the next payload operation in walk order is matched. If it
fails, both silenceable and definite errors are propagated as the result of
this op; propagation of silenceable errors is postponed until the end of the
walk.

The matcher symbol must take at least one operand of a type that implements
the same transform dialect interface as the `root` operand (a check is
performed at application time to see if the associated payload satisfies the
constraints of the actual type), and may take additional operands with a
similar type requirement. It must not consume operands as multiple matchers
may be applied. The matcher may produce any number of results. The action
symbol paired with the matcher must take the same number of arguments as the
matcher has results, and these arguments must implement the same transform
dialect interfaces, but not necessarily have the exact same type (again, a
check is performed at application time to see if the associated payload
satisfies the constraints of actual types on both sides).

The action symbol may have results that are accumulated from all actions and
returned from the `foreach_match` operation on success. Unless the
`flatten_results` attribute is present, each action result must be
associated with exactly one payload entity. The actions are expected to only
modify payload operations nested in the `root` payload operations associated
with the operand of this transform operation. Furthermore, the actions may
not modify operations outside of the currently matched payload operation,
e.g., they may not modify sibling or parent operations. If such behavior is
desired, the parent must be matched first and the nested operations obtained
by traversing the IR from the parent. This is due to the matching being
performed as a post-order IR walk.

This operation consumes the operand and produces a new handle associated
with the same payload. This is necessary to trigger invalidation of handles
to any of the payload operations nested in the payload operations associated
with the operand, as those are likely to be modified by actions.

By default, the root payload operation associated with the operand is not
matched. This is to support the conservative case where applied actions may
invalidate the root payload operation. If the optional `restrict_root`
attribute is set, the root operand is guaranteed to not be invalidated by any
of the applied actions. In such cases, the root payload operation is also
matched. This is useful because matching the root payload operation is a
common idiom, when e.g. matching a func.func directly and operations nested
under it.

The operation succeeds if none of the matchers produced a definite failure
during application and if all of the applied actions produced success. Note
that it also succeeds if all the matchers failed on all payload operations,
i.e. failure to apply is not an error. The operation produces a silenceable
failure if any applied action produced a silenceable failure. In this case,
the resulting handle is associated with an empty payload. The operation
produces a definite failure if any of the applied matchers or actions
produced a definite failure.

Interfaces: `MemoryEffectOpInterface`, `OpAsmOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>restrict_root</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>flatten_results</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>matchers</code></td><td>::mlir::ArrayAttr</td><td>symbol ref array attribute</td></tr>
<tr><td><code>actions</code></td><td>::mlir::ArrayAttr</td><td>symbol ref array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |
| `forwarded_inputs` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `updated` | TransformHandleTypeInterface instance |
| `forwarded_outputs` | variadic of any transform handle or parameter |


### `transform.foreach` (transform::ForeachOp)

_Executes the body for each element of the payload_

Syntax:

```
operation ::= `transform.foreach` $targets oilist(`with_zip_shortest` $with_zip_shortest) `:` type($targets) (`->` type($results)^)? $body attr-dict
```

Execute the op's body - its single region block - exactly once per
element of the payload associated to a target handle. The body's
transformations are applied in order of appearance until reaching the
(implicit) YieldOp terminator.

Each iteration gets executed by co-indexing the payloads of the arguments
and mapping the body's arguments to these tuples, as though iterating over
the zipped together `targets`. As such, in each iteration, the size of the
payload of each of the body's block arguments is exactly one. The attribute
`zip_shortest` can be used if the targets vary in their number of payloads;
this will limit the iterations to only the number of payloads found in the
shortest target.

This op always reads the target handles. Furthermore, it consumes a handle
if there is a transform op in the body that consumes the corresponding
block argument. Handles can point to ops, values, or parameters.

#### Return Modes

This op produces as many result handles as the body's terminating YieldOp
has operands. For each result, the payloads of the corresponding YieldOp
operand are merged and mapped to the same resulting handle.

If the target handles do not associate payloads of the same size, a
silencable failure will be generated.

During application, if any transformation in the sequence fails, the entire
sequence fails immediately with the same failure, leaving the payload IR in
a potentially invalid state, i.e., this operation offers no transformation
rollback capabilities.

Traits: `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>with_zip_shortest</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `targets` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of any transform handle or parameter |


### `transform.get_consumers_of_result` (transform::GetConsumersOfResult)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>result_number</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `consumers` | TransformHandleTypeInterface instance |


### `transform.get_defining_op` (transform::GetDefiningOp)

_Get handle to the defining op of a value_

Syntax:

```
operation ::= `transform.get_defining_op` $target attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to the defining op of
the targeted value.

This transform produces a silenceable failure if the targeted value is a
block argument.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformValueHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.get_operand` (transform::GetOperandOp)

_Get a handle to the operand(s) of the targeted op_

Syntax:

```
operation ::= `transform.get_operand` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to the operands of the
given `target` operation specified by the given set of positions. There are
three possible modes:

 - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
   operands at the specified positions.
 - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
   all operands except those at the given positions.
 - All, i.e. `%target[all]`. This will return all operands of the operation.

This transform produces a silenceable failure if any of the operand indices
exceeds the number of operands in the target. It reads the target handle and
produces the result handle.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>raw_position_list</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>is_inverted</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>is_all</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformValueHandleTypeInterface instance |


### `transform.get_parent_op` (transform::GetParentOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>isolated_from_above</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>allow_empty_results</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>op_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>deduplicate</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>nth_parent</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `parent` | TransformHandleTypeInterface instance |


### `transform.get_producer_of_operand` (transform::GetProducerOfOperand)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>operand_number</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `producer` | TransformHandleTypeInterface instance |


### `transform.get_result` (transform::GetResultOp)

_Get a handle to the result(s) of the targeted op_

Syntax:

```
operation ::= `transform.get_result` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op correspond to the OpResults of the
given `target` operation. Optionally `result_number` can be specified to
select a specific result.

This transform fails silently if the targeted operation does not have enough
results. It reads the target handle and produces the result handle.

The handle defined by this Transform op corresponds to the results of the
given `target` operation specified by the given set of positions. There are
three possible modes:

 - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
   results at the specified positions.
 - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
   all results except those at the given positions.
 - All, i.e. `%target[all]`. This will return all results of the operation.

This transform produces a silenceable failure if any of the result indices
exceeds the number of results returned by the target. It reads the target
handle and produces the result handle.

Traits: `NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>raw_position_list</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>is_inverted</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>is_all</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformValueHandleTypeInterface instance |


### `transform.get_type` (transform::GetTypeOp)

_Get a parameter containing the type of the given value_

Syntax:

```
operation ::= `transform.get_type` (`elemental` $elemental^)? $value attr-dict `:`functional-type(operands, results)
```

This operation creates a new Transform parameter containing the
type(s) of the value(s) associated with the operand handle.

This transform never fails.

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>elemental</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `value` | TransformValueHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `type_param` | TransformParamTypeInterface instance |


### `transform.include` (transform::IncludeOp)

_Includes a named transform sequence_

Syntax:

```
operation ::= `transform.include` $target `failures` `(` $failure_propagation_mode `)``(` $operands `)` attr-dict `:` functional-type($operands, $results)
```

The application of this transform operation is equivalent to applying the
operations contained in the named transform sequence with operands being
remapped to block arguments. The behavior of the operation when a
transformation in the included named sequence produces a silenceable error
is controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the user to ensure the following transformations are
robust enough when errors are suppressed. Definite errors are propagated
immediately regardless of the mode. The objects associated with the results
of this operation are the same as those associated with the operands of the
`transform.yield` in the referenced named sequence.

Interfaces: `CallOpInterface`, `MatchOpInterface`, `MemoryEffectOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>target</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
<tr><td><code>failure_propagation_mode</code></td><td>::mlir::transform::FailurePropagationModeAttr</td><td>Silenceable error propagation policy</td></tr>
<tr><td><code>arg_attrs</code></td><td>::mlir::ArrayAttr</td><td>Array of dictionary attributes</td></tr>
<tr><td><code>res_attrs</code></td><td>::mlir::ArrayAttr</td><td>Array of dictionary attributes</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of any transform handle or parameter |


### `transform.match.operation_empty` (transform::MatchOperationEmptyOp)

_Matches if the handle is not associated to any op_

Syntax:

```
operation ::= `transform.match.operation_empty` $operand_handle attr-dict `:` type($operand_handle)
```

Succeeds if the handle is not associated to any op.

Traits: `AtMostOneOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |


### `transform.match.operation_name` (transform::MatchOperationNameOp)

_Matches a single operation of one of the given kinds_

Syntax:

```
operation ::= `transform.match.operation_name` $operand_handle $op_names attr-dict `:` type($operand_handle)
```

Succeeds if the operation associated with the operand handle has one of the
given operation names. Produces a silenceable failure otherwise.

If more than one payload operation is associated with the operand handle,
produces a definite failure.

Traits: `SingleOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>op_names</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |


### `transform.match.param.cmpi` (transform::MatchParamCmpIOp)

_Matches if two parameter lists are associated with the same value_

Syntax:

```
operation ::= `transform.match.param.cmpi` $predicate $param `,` $reference attr-dict `:` type($param)
```

Succeeds if all of the co-indexed values associated with the given
parameters relate as specified by the predicate (greater than, less than,
equal to, or their combinations). Comparison treats all values as signed.
Produces a silenceable failure otherwise.

Traits: `SameTypeOperands`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>predicate</code></td><td>::mlir::transform::MatchCmpIPredicateAttr</td><td>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `param` | TransformParamTypeInterface instance |
| `reference` | TransformParamTypeInterface instance |


### `transform.merge_handles` (transform::MergeHandlesOp)

_Merges handles into one pointing to the union of payload ops_

Syntax:

```
operation ::= `transform.merge_handles` (`deduplicate` $deduplicate^)? $handles attr-dict `:` type($result)
```

Creates a new Transform IR handle value that points to the same Payload IR
operations/values/parameters as the operand handles. The Payload IR elements
are listed in the same order as they are in the operand handles, grouped by
operand handle, e.g., all Payload IR associated with the first handle comes
first, then all Payload IR associated with the second handle and so on. If
`deduplicate` is set, do not add the given Payload IR operation, value, or
parameter more than once to the final list regardless of it coming from the
same or different handles. Consumes the operands and produces a new handle.

Traits: `SameOperandsAndResultType`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>deduplicate</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handles` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any transform handle or parameter |


### `transform.named_sequence` (transform::NamedSequenceOp)

_Named transform sequence that can be included elsewhere_

Defines a named (callable, function-like) sequence of other Transform
dialect operations that can be included using `transform.include` as part of
another Transform dialect construct. This sequence is not processed
immediately but rather dispatched to when the inclusion is processed. The
arguments and results can be used to communicate a subset of mapping into
the named sequence. The sequence must consist of a single block and end with
a `transform.yield` terminator. The operands of the terminator become the
results of the `transform.include`.

When dispatched to, the operations in the named sequence are executed one by
one, similarly to the regular unnamed sequence. The failure propagation mode
is specified on the `transform.include`. Different inclusions may use
different failure propagation modes. This transform operation always
succeeds by itself, but the inclusion may fail if any of the operations
fail.

Named sequences can only appear at the top-level of the Transform dialect
nesting structure. That is, they cannot be nested in other Transform dialect
operations. Furthermore, one of the ancestors must have the `SymbolTable`
trait and have the `transform.with_named_sequence` attribute attached.

Named sequences may include other named sequences via `transform.include`,
but recursion is *not* allowed.

Traits: `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `MemoryEffectOpInterface`, `Symbol`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>function_type</code></td><td>::mlir::TypeAttr</td><td>function type attribute</td></tr>
<tr><td><code>sym_visibility</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>arg_attrs</code></td><td>::mlir::ArrayAttr</td><td>Array of dictionary attributes</td></tr>
<tr><td><code>res_attrs</code></td><td>::mlir::ArrayAttr</td><td>Array of dictionary attributes</td></tr>
</table>


### `transform.num_associations` (transform::NumAssociationsOp)

_Returns the number of payload objects associated with the argument_

Syntax:

```
operation ::= `transform.num_associations` $handle attr-dict `:` functional-type(operands, results)
```

Given an argument, handle or parameter, returns a new parameter associated
with a single 64-bit number that corresponds to the number of payload
objects (operations or values for a handle, attributes for a parameter)
associated with the argument.

Always succeeds.

Traits: `ParamProducerTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handle` | any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `num` | TransformParamTypeInterface instance |


### `transform.param.constant` (transform::ParamConstantOp)

_Produces a new transform dialect parameter value associated with the given attribute_

Syntax:

```
operation ::= `transform.param.constant` $value attr-dict `->` type($param)
```

Produces a new transform dialect parameter associated with the singleton
list containing the given attribute. The operation itself always succeeds,
but the general association check may fail if the parameter type does not
accept the given kind of attribute as valid.

Traits: `ParamProducerTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `param` | TransformParamTypeInterface instance |


### `transform.print` (transform::PrintOp)

_Dump each payload op_

Syntax:

```
operation ::= `transform.print` $target attr-dict (`:` type($target)^)?
```

Prints each payload op that is associated with the `target` operand to
`stdout`. It also prints the `name` string attribute. If no target is
specified, the top-level op is dumped.

This op is useful for printf-style debugging.

Supported printing flag attributes:
* `assume_verified` -- skips verification when the unit attribute is
  specified. This improves performace but may lead to crashes and
  unexpected behavior when the printed payload op is invalid.
* `use_local_scope` -- prints in local scope when the unit attribute is
  specified. This improves performance but may not be identical to
  printing within the full module.
* `skip_regions` -- does not print regions of operations when the unit
  attribute is specified.

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>assume_verified</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_local_scope</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>skip_regions</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.replicate` (transform::ReplicateOp)

_Lists payload ops multiple times in the new handle_

Syntax:

```
operation ::= `transform.replicate` `num` `(` $pattern `)` $handles attr-dict `:` type($pattern) `,` type($handles)
```

Produces a new handle associated with a list of payload IR ops that is
computed by repeating the list of payload IR ops associated with the
operand handle as many times as the "pattern" handle has associated
operations. For example, if pattern is associated with [op1, op2] and the
operand handle is associated with [op3, op4, op5], the resulting handle
will be associated with [op3, op4, op5, op3, op4, op5].

This transformation is useful to "align" the sizes of payload IR lists
before a transformation that expects, e.g., identically-sized lists. For
example, a transformation may be parameterized by same notional per-target
size computed at runtime and supplied as another handle, the replication
allows this size to be computed only once and used for every target instead
of replicating the computation itself.

Note that it is undesirable to pass a handle with duplicate operations to
an operation that consumes the handle. Handle consumption often indicates
that the associated payload IR ops are destroyed, so having the same op
listed more than once will lead to double-free. Single-operand
MergeHandlesOp may be used to deduplicate the associated list of payload IR
ops when necessary. Furthermore, a combination of ReplicateOp and
MergeHandlesOp can be used to construct arbitrary lists with repetitions.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `pattern` | TransformHandleTypeInterface instance |
| `handles` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `replicated` | variadic of any transform handle or parameter |


### `transform.select` (transform::SelectOp)

_Select payload ops by name_

Syntax:

```
operation ::= `transform.select` $op_name `in` $target attr-dict `:` functional-type(operands, results)
```

The handle defined by this Transform op corresponds to all operations among
`target` that have the specified properties. Currently the following
properties are supported:

- `op_name`: The op must have the specified name.

The result payload ops are in the same relative order as the targeted ops.
This transform op reads the `target` handle and produces the `result`
handle. It reads the payload, but does not modify it.

Traits: `NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>op_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.sequence` (transform::SequenceOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>failure_propagation_mode</code></td><td>::mlir::transform::FailurePropagationModeAttr</td><td>Silenceable error propagation policy</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |
| `extra_bindings` | variadic of any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of TransformHandleTypeInterface instance |


### `transform.split_handle` (transform::SplitHandleOp)

_Splits a handle or parameter into multiple values_

Syntax:

```
operation ::= `transform.split_handle` $handle attr-dict `:` functional-type(operands, results)
```

Splits `handle` into one or multiple handles, as specified by the number
of results of this operation. `handle` should be mapped to as many payload
ops, values or parameteres as there are results. Otherwise, this transform
will fail producing a silenceable failure by default. Each result handle
is mapped to exactly one payload unless specified otherwise by attributes
described below. The order of the payloads is preserved,  i.e., the i-th
payload is mapped to the i-th result handle.

This operation is useful for ensuring a statically known number of
payloads are tracked by the source `handle` and to extract them into
individual handles that can be further manipulated in isolation.

If there are more payloads than results, the remaining payloads are mapped to
the result with index `overflow_result`. If no `overflow_result` is
specified, the transform produces a silenceable failure.

If there are fewer payload ops than results, the transform produces a
silenceable failure if `fail_on_payload_too_small` is set to "true".
Otherwise, it succeeds and the remaining result handles are not mapped to
anything. It also succeeds if `handle` is empty and
`pass_through_empty_handle` is set to "true", regardless of
`fail_on_payload_too_small`.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pass_through_empty_handle</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>fail_on_payload_too_small</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>overflow_result</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handle` | any transform handle or parameter |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of any transform handle or parameter |


### `transform.verify` (transform::VerifyOp)

_Verifies the targeted ops_

Syntax:

```
operation ::= `transform.verify` $target attr-dict `:` type($target)
```

This transform verifies the targeted ops. If at least one op fails to
verify, the transform produces a definite failure.

Note: This op was designed for debugging purposes and should be used like an
assertion. It is intentional that this op produces a definite failure and
not a silenceable one. Correctness of the program should not depend on this
op.

This transform reads the target handle.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.yield` (transform::YieldOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any transform handle or parameter |


## Affine Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Affine/TransformOps/AffineTransformOps.td)

### `transform.affine.simplify_bounded_affine_ops` (transform::SimplifyBoundedAffineOpsOp)

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

#### Return modes

Target ops must be affine.min or affine.max ops. This transform consumes the
target handle and does not produce any handle. It reads the bounded op
handles.

TODO: Support affine.apply targets.
TODO: Allow mixed PDL_Operation/int64_t for lower_bounds and upper_bounds.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lower_bounds</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>upper_bounds</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `bounded_values` | variadic of TransformHandleTypeInterface instance |


### `transform.affine.simplify_min_max_affine_ops` (transform::SimplifyMinMaxAffineOpsOp)

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

#### Return modes

This transform consumes the target handle and does not produce any results.
This transforms definitely fails if any of the targeted operations is not an
`affine.min` or `affine.max` operation, or if the canonicalization patterns
failed to converge.
This transform silently fails if none of the operations were simplified.
Otherwise, it succeeds.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


## Bufferization Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.td)

### `transform.bufferization.buffer_loop_hoisting` (transform::BufferLoopHoistingOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.bufferization.eliminate_empty_tensors` (transform::EliminateEmptyTensorsOp)

Syntax:

```
operation ::= `transform.bufferization.eliminate_empty_tensors` $target attr-dict `:` type($target)
```

Try to eliminate all `tensor.empty` ops within the targeted op by replacing
them with another destination tensor.

"tensor.empty" ops cannot be bufferized. They can either be converted to
"bufferization.alloc_tensor" or replaced with another tensor (via this
transform). "tensor.empty" does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transformation looks for subset ops that insert a tensor that
originates from a "tensor.empty" (as per the reverse use-def chain). Such
"tensor.empty" ops are replaced with the destination subset.

Example:

```
%0 = tensor.empty() : tensor<5xf32>
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

Is rewritten with:
```
%0 = tensor.extract_slice %t[1][5][1]
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

In the above example, the subset op is "tensor.insert_slice". When tracing
back the reverse use-def chain of a the source, we end up at a
"tensor.empty" op.

The above example can bufferize without an allocation (in the absence of
other conflicts) because there is no longer a `tensor.empty` op.

See `-eliminate-empty-tensors` for more details.

#### Return modes

This transform reads the target handle and modifies the payload. It does
not produce any handle.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.bufferization.empty_tensor_to_alloc_tensor` (transform::EmptyTensorToAllocTensorOp)

Syntax:

```
operation ::= `transform.bufferization.empty_tensor_to_alloc_tensor` $target attr-dict `:` functional-type(operands, results)
```

Replace a tensor.empty with a bufferization.tensor_alloc.

#### Return modes

This operation consumes the `target` handle and produces the `transformed`
handle. `target` is expected to be a `tensor.empty` operation. The transform
always succeeds.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to tensor.empty operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | Transform IR handle to bufferization.alloc_tensor operations |


### `transform.bufferization.one_shot_bufferize` (transform::OneShotBufferizeOp)

Syntax:

```
operation ::= `transform.bufferization.one_shot_bufferize` (`layout` `{` $function_boundary_type_conversion^ `}`)?
              $target attr-dict `:` functional-type($target, results)
```

Indicates that the given `target` op should be bufferized with One-Shot
Bufferize. The bufferization can be configured with various attributes that
corresponding to options in `BufferizationOptions` and the
`one-shot-bufferize` pass. More information can be found in the pass
documentation.

The targeted ops must be modules or functions. This is because there is
always a single, bufferized replacement op for such targets.

Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
unset, this transform fails when an unknown/non-bufferizable op is found.
Many ops implement `BufferizableOpInterface` via an external model. These
external models must be registered when applying this transform op;
otherwise, said ops would be considered non-bufferizable.

#### Return modes

This operation consumes the `target` handle and produces the `transformed`
handle.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>function_boundary_type_conversion</code></td><td>::mlir::bufferization::LayoutMapOptionAttr</td><td>option for map layout</td></tr>
<tr><td><code>allow_return_allocs_from_loops</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>allow_unknown_ops</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>bufferize_function_boundaries</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>dump_alias_sets</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>test_analysis_only</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>print_conflicts</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>check_parallel_regions</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>memcpy_op</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


## Debug Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.td)

### `transform.debug.emit_param_as_remark` (transform::EmitParamAsRemarkOp)

_Prints the parameter as a diagnostic remark_

Syntax:

```
operation ::= `transform.debug.emit_param_as_remark` $param (`,` $message^)?  (`at` $anchor^)?attr-dict `:` type($param) (`,` type($anchor)^)?
```

This operation emits a diagnostic remark containing the string form of the
attributes associated with the parameter provided as attribute. It takes
as optional arguments:
  - an additional message text to prepend;
  - a handle pointing to operations the location of which will be used to
    emit the diagnostic; if multiple operations are associated, the
    diagnostic is emitted for all of their respective locations.

This operation always succeeds.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>message</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `param` | TransformParamTypeInterface instance |
| `anchor` | TransformHandleTypeInterface instance |


### `transform.debug.emit_remark_at` (transform::EmitRemarkAtOp)

_Print a message as diagnostic remark attached to payload_

Syntax:

```
operation ::= `transform.debug.emit_remark_at` $at `,` $message attr-dict `:` type($at)
```

This operation emits a diagnostic remark with the given message at the
location of each payload object associated with the argument. The argument
may be an operation or a value handle.

This operation always succeeds.

Traits: `NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>message</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `at` | any transform handle |


## DLTI Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/DLTI/TransformOps/DLTITransformOps.td)

### `transform.dlti.query` (transform::QueryOp)

_Return attribute (as param) associated to key via DTLI_

Syntax:

```
operation ::= `transform.dlti.query` $keys `at` $target attr-dict `:` functional-type(operands, results)
```

This op queries data layout and target information associated to payload
IR by way of the DLTI dialect.

A lookup is performed for the given `keys` at `target` op - or its closest
interface-implementing ancestor - by way of the `DLTIQueryInterface`, which
returns an attribute for a key. Each key should be either a (quoted) string
or a type. If more than one key is provided, the lookup continues
recursively, now on the returned attributes, with the condition that these
implement the above interface. For example if the payload IR is

```
module attributes {#dlti.map = #dlti.map<#dlti.dl_entry<"A",
                                 #dlti.map<#dlti.dl_entry<"B", 42: int>>>} {
  func.func private @f()
}
```
and we have that `%func` is a Tranform handle to op `@f`, then
`transform.dlti.query ["A", "B"] at %func` returns 42 as a param and
`transform.dlti.query ["A"] at %func` returns the `#dlti.map` attribute
containing just the key "B" and its value. Using `["B"]` or `["A","C"]` as
`keys` will yield an error.

#### Return modes

When successful, the result, `associated_attr`, associates one attribute as
a param for each op in `target`'s payload.

If the lookup fails - as no DLTI attributes/interfaces are found or entries
with the right names are missing - a silenceable failure is returned.

Traits: `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keys</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `associated_attr` | TransformParamTypeInterface instance |


## IRDL (extension) Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.td)

### `transform.irdl.collect_matching` (transform::IRDLCollectMatchingOp)

_Finds ops that match the IRDL definition without registering them._

Syntax:

```
operation ::= `transform.irdl.collect_matching` `in` $root `:` functional-type(operands, results) attr-dict-with-keyword regions
```

Traits: `NoTerminator`, `SymbolTable`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `matched` | TransformHandleTypeInterface instance |


## Func Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/TransformOps/FuncTransformOps.td)

### `transform.apply_conversion_patterns.func.func_to_llvm` (transform::ApplyFuncToLLVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.func.func_to_llvm` attr-dict
```

Collects patterns that convert Func dialect ops to LLVM dialect ops.
These patterns require an "LLVMTypeConverter".

Interfaces: `ConversionPatternDescriptorOpInterface`


### `transform.func.cast_and_call` (transform::CastAndCallOp)

_Casts values to the signature of a function and replaces them with a call_

Syntax:

```
operation ::= `transform.func.cast_and_call` ($function_name^)? ($function^)?
              ( `(` $inputs^ `)` )?
              ( `->` $outputs^ )?
              (`after` $insert_after^):(`before`)? $insertion_point
              ($conversions^)? attr-dict `:` functional-type(operands, results)
```

This transform takes value handles to a set of `inputs` and `outputs` and
attempts to cast them to the function signature of the attached function
op, then builds a call to the function and replaces the users of the
outputs. It is the responsibility of the user to ensure that the slice of
the program replaced by this operation makes sense, i.e. there is no
verification that the inputs to this operation have any relation to the
outputs outside of basic dominance requirements needed for the call.

The casting materialization functions are specified in the graph region of
this op. They must implement the `TypeConverterBuilderOpInterface`. The
order of ops within the region is irrelevant.

The target function can be specified by a symbol name or by a handle to the
operation.

This transform only reads the operand handles and only replaces the users of
the outputs with the results of the call. No handles are consumed and no
operations are removed. Users are expected to run cleanup separately if
desired.

Warning: The replacement of the uses of the outputs could invalidate certain
restricted value handle types (e.g. `transform.block_arg` if it existed, by
replacing the use with something not coming from a block argument). The
value will still exist in such cases but wouldn't verify against the type.
See the discussion here for more information:
https://github.com/llvm/llvm-project/pull/78398#discussion_r1455070087

This transform will emit a silenceable failure if:
 - The set of outputs isn't unique
 - The handle for the insertion point does not include exactly one operation
 - The insertion point op does not dominate any of the output users
 - The insertion point op is not dominated by any of the inputs
 - The function signature does not match the number of inputs/outputs

This transform will emit a definite failure if it fails to resolve the
target function, or if it fails to materialize the conversion casts of
either the inputs to the function argument types, or the call results to
the output types.

Traits: `AttrSizedOperandSegments`, `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>insert_after</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>function_name</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `insertion_point` | TransformHandleTypeInterface instance |
| `inputs` | TransformValueHandleTypeInterface instance |
| `outputs` | TransformValueHandleTypeInterface instance |
| `function` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.func.replace_func_signature` (transform::ReplaceFuncSignatureOp)

Syntax:

```
operation ::= `transform.func.replace_func_signature` $function_name
              `args_interchange` `=` $args_interchange
              `results_interchange` `=` $results_interchange
              `at` $module attr-dict `:` functional-type(operands, results)
```

This transform takes a module and a function name, and replaces the
signature of the function by reordering the arguments and results
according to the interchange arrays. The function is expected to be
defined in the module, and the interchange arrays must match the number
of arguments and results of the function.

The `adjust_func_calls` attribute indicates whether the function calls
should be adjusted to match the new signature. If set to `true`, the
function calls will be adjusted to match the new signature, otherwise
they will not be adjusted.

This transform will emit a silenceable failure if:
 - The function with the given name does not exist in the module.
 - The interchange arrays do not match the number of arguments/results.
 - The interchange arrays contain out of bound indices.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>function_name</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
<tr><td><code>args_interchange</code></td><td>::mlir::DenseI32ArrayAttr</td><td>i32 dense array attribute</td></tr>
<tr><td><code>results_interchange</code></td><td>::mlir::DenseI32ArrayAttr</td><td>i32 dense array attribute</td></tr>
<tr><td><code>adjust_func_calls</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `module` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed_module` | TransformHandleTypeInterface instance |
| `transformed_function` | TransformHandleTypeInterface instance |


## GPU Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td)

### `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` (transform::ApplyGPUPromoteShuffleToAMDGPUPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` attr-dict
```

Collects patterns that are tryin to promote `gpu.shuffle`s to specialized
AMDGPU intrinsics.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.gpu.gpu_rewrite_patterns` (transform::ApplyGPURewritePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.gpu_rewrite_patterns` attr-dict
```

Collects GPU rewrite patterns comprising:
  1. GpuAllReduceRewrite patterns
  2. GpuGlobalIdRewriter patterns
  3. GpuShuffleRewriter patterns

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` (transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` attr-dict
```

Collects patterns that convert GPU dialect ops related to wmma ops
to NVVM dialect ops.
These patterns require an "LLVMTypeConverter".

Interfaces: `ConversionPatternDescriptorOpInterface`


### `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` (transform::ApplyGPUToNVVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` attr-dict
```

Collects patterns that convert GPU dialect ops to NVVM dialect ops. These
patterns require an "LLVMTypeConverter".

Interfaces: `ConversionPatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>benefit</code></td><td>::mlir::IntegerAttr</td><td>16-bit signless integer attribute</td></tr>
</table>


### `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` (transform::ApplyGPUToROCDLConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` `chipset` `=` $chipset attr-dict
```

Collects patterns that convert GPU dialect ops to ROCDL dialect ops. These
patterns require an "LLVMTypeConverter".

Interfaces: `ConversionPatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>chipset</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>


### `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` (transform::ApplyGPUWwmaToNVVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` attr-dict
```

Collects patterns that convert GPU dialect ops related to wmma ops
to NVVM dialect ops.
These patterns require an "LLVMTypeConverter".

Interfaces: `ConversionPatternDescriptorOpInterface`


### `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` (transform::ApplyUnrollVectorsSubgroupMmaOp)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` `[` $m `,` $n `,` $k `]` attr-dict
```

Unrolls contractions to the target `m`, `n`, and `k` native vector size,
along with other vector operations based on expected usage. `transfer_read`
ops unroll based on the extract slice shape introduced by unrolling the
contractions, while elementwise and `transfer_write` ops unroll to the shape of
the C matrix (`m x n`).

This operation applies to pure vector operations and should be applied before
lowering to subgroup_mma ops.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>m</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>n</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>k</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>


### `transform.apply_patterns.gpu.eliminate_barriers` (transform::EliminateBarriersOp)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.eliminate_barriers` attr-dict
```

Removes unnecessary GPU barriers from the function. If a barrier does not
enforce any conflicting pair of memory effects, including a pair that is
enforced by another barrier, it is unnecessary and can be removed.

The approach is based on "High-Performance GPU-to-CPU Transpilation and
Optimization via High-Level Parallel Constructs" by  Moses, Ivanov,
Domke, Endo, Doerfert, and Zinenko in PPoPP 2023. Specifically, it
analyzes the memory effects of the operations before and after the given
barrier and checks if the barrier enforces any of the memory
effect-induced dependencies that aren't already enforced by another
barrier.

For example, in the following code

```mlir
  store %A
  barrier  // enforces load-after-store
  load %A
  barrier  // load-after-store already enforced by the previous barrier
  load %A
```

the second barrier can be removed.

Interfaces: `PatternDescriptorOpInterface`


### `transform.gpu.map_forall_to_blocks` (transform::MapForallToBlocks)

Syntax:

```
operation ::= `transform.gpu.map_forall_to_blocks` $target
              (`generate_gpu_launch` $generate_gpu_launch^)?
              (`grid_dims` `=` $grid_dims^)?
              attr-dict
              `:` functional-type($target, $result)
```

Target the gpu_launch op and rewrite the top level `scf.forall`
to distributed gpu.block_id attribute. If `generate_gpu_launch` attribute
is set, then first generates `gpu_launch` and moves the top level
`scf.forall` inside.

The operation searches top level `scf.forall` ops under
`gpu_launch` and maps each such op to GPU blocks. Mapping is
one-to-one and the induction variables of `scf.forall` are
rewritten to gpu.block_id according to the `thread_dim_mapping` attribute.

Dynamic, `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** scf.forall are currently supported.
Only scf.forall distributed to **at most 3 dimensions** are
currently supported.

The operation alters the block size of the given gpu_launch using the
grid_dims argument.

#### Return modes:

This operation ignores non-gpu_launch ops and drops them in the return.

If any scf.forall with tensors is found, the transform definitely
fails.

If all the `scf.forall` operations contained within the LaunchOp
referred to by the `target` handle lower to GPU properly, the
transform succeeds. Otherwise the transform definitely fails.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>grid_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>generate_gpu_launch</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.gpu.map_nested_forall_to_threads` (transform::MapNestedForallToThreads)

Syntax:

```
operation ::= `transform.gpu.map_nested_forall_to_threads` $target
              `block_dims` `=` $block_dims
              (`sync_after_distribute` `=` $sync_after_distribute^)?
              (`warp_size` `=` $warp_size^)?
              attr-dict
              `:` functional-type($target, $result)
```

Target the `gpu.launch op` and rewrite all `scf.forall` nested in it to
distributed `gpu.thread_id` attribute.

The operation searches for `scf.forall` ops nested under `target` and maps
each such op to GPU threads.

`scf.forall` induction variables are rewritten to `gpu.thread_id` according
to the `mapping` attribute.

Different types of mappings attributes are supported:
  - the block_dims is a list of integers that specifies the number of
    threads in each dimension. This is a mandatory attribute that is used
    to constrain the number of threads in each dimension. If an
    `scf.forall` op is mapped to fewer threads, predication occurs.
  - the warp_dims is a list of integers that specifies the number of
    warps in each dimension. This is an optional attribute that is used
    to constrain the number of warps in each dimension. When present, this
    attribute must be specified in a way that is compatible with the
    block_dims attribute. If an `scf.forall` op is mapped to fewer warps,
    predication occurs.

Dynamic `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** `scf.forall` are currently supported.
Only `scf.forall` distributed to **at most 3 dimensions** are
currently supported.

The `sync_after_distribute`attribute controls whether a `gpu.barrier` is
inserted after each scf.forall op. At this time, this is an all or nothing
choice. This will need to be tightened in the future.

The operation alters the block size of the given gpu_launch using the
mandatory block_dims argument.

#### Return modes:

This operation ignores non-`gpu_launch` ops and drops them in the return.

If any scf.forall with tensors is found, the transform definitely
fails.

If all the `scf.forall` operations with gpu.thread mapping contained
within the `LaunchOp` referred to by the `target` handle lower to GPU
properly, the transform succeeds. Otherwise the transform definitely
fails.

scf.forall operations with mappings other than gpu.thread are
ignored.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

#### Example:

```
gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
           threads(%tx, %ty, %tz) in (%tx = %3, %ty = %4, %tz = %5) {
  scf.forall (%i, %j) in (7, 9) {
    ... // body 1
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
  scf.forall (%i) in (12) {
    ... // body 2
  } {mapping = [#gpu.thread<x>]}
  gpu.terminator
}
```

is translated to:

```
%bdimX = arith.constant 12 : index
%bdimY = arith.constant 9 : index
gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
       threads(%tx, %ty, %tz) in (%tx = %bdimX, %ty = %bdimY, %tz = %5) {
  if (threadIdx.x < 9 && threadIdx.y < 7) {
    ... // body 1
  }
  gpu.barrier
  if (threadIdx.y < 1) {
    ... // body 2
  }
  gpu.barrier
  gpu.terminator
}
```

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>block_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>sync_after_distribute</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>warp_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


## Loop (extension) Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.td)

### `transform.loop.hoist_loop_invariant_subsets` (transform::HoistLoopInvariantSubsetsOp)

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


## Loop (SCF) Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td)

### `transform.apply_patterns.scf.for_loop_canonicalization` (transform::ApplyForLoopCanonicalizationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.scf.for_loop_canonicalization` attr-dict
```

Collects patterns for canonicalizing operations inside SCF loop bodies.
At the moment, only affine.min/max computations with iteration variables,
loop bounds and loop steps are canonicalized.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_conversion_patterns.scf.structural_conversions` (transform::ApplySCFStructuralConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.scf.structural_conversions` attr-dict
```

Collects patterns for performing structural conversions of SCF operations.

Interfaces: `ConversionPatternDescriptorOpInterface`


### `transform.apply_conversion_patterns.scf.scf_to_control_flow` (transform::ApplySCFToControlFlowPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.scf.scf_to_control_flow` attr-dict
```

Collects patterns that lower structured control flow ops to unstructured
control flow.

Interfaces: `ConversionPatternDescriptorOpInterface`


### `transform.loop.forall_to_for` (transform::ForallToForOp)

_Converts scf.forall into a nest of scf.for operations_

Syntax:

```
operation ::= `transform.loop.forall_to_for` $target attr-dict `:` functional-type(operands, results)
```

Converts the `scf.forall` operation pointed to by the given handle into a
set of nested `scf.for` operations. Each new operation corresponds to one
induction variable of the original "multifor" loop.

The operand handle must be associated with exactly one payload operation.

Loops with shared outputs are currently not supported.

#### Return Modes

Consumes the operand handle. Produces a silenceable failure if the operand
is not associated with a single `scf.forall` payload operation.
Returns as many handles as the given `forall` op has induction variables
that are associated with the generated `scf.for` loops.
Produces a silenceable failure if another number of resulting handles is
requested.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | variadic of TransformHandleTypeInterface instance |


### `transform.loop.forall_to_parallel` (transform::ForallToParallelOp)

_Converts scf.forall into a nest of scf.for operations_

Syntax:

```
operation ::= `transform.loop.forall_to_parallel` $target attr-dict `:` functional-type(operands, results)
```

Converts the `scf.forall` operation pointed to by the given handle into an
`scf.parallel` operation.

The operand handle must be associated with exactly one payload operation.

Loops with outputs are not supported.

#### Return Modes

Consumes the operand handle. Produces a silenceable failure if the operand
is not associated with a single `scf.forall` payload operation.
Returns a handle to the new `scf.parallel` operation.
Produces a silenceable failure if another number of resulting handles is
requested.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | variadic of TransformHandleTypeInterface instance |


### `transform.loop.coalesce` (transform::LoopCoalesceOp)

_Coalesces the perfect loop nest enclosed by a given loop_

Syntax:

```
operation ::= `transform.loop.coalesce` $target attr-dict `:` functional-type($target, $transformed)
```

Given a perfect loop nest identified by the outermost loop,
perform loop coalescing in a bottom-up one-by-one manner.

#### Return modes

The return handle points to the coalesced loop if coalescing happens, or
the given input loop if coalescing does not happen.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.loop.fuse_sibling` (transform::LoopFuseSiblingOp)

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

#### Return modes

This operation consumes the `target` and `source` handles and produces the
`fused_loop` handle, which points to the fused loop.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `source` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `fused_loop` | TransformHandleTypeInterface instance |


### `transform.loop.outline` (transform::LoopOutlineOp)

_Outlines a loop into a named function_

Syntax:

```
operation ::= `transform.loop.outline` $target attr-dict `:` functional-type(operands, results)
```

Moves the loop into a separate function with the specified name and replaces
the loop in the Payload IR with a call to that function. Takes care of
forwarding values that are used in the loop as function arguments. If the
operand is associated with more than one loop, each loop will be outlined
into a separate function. The provided name is used as a _base_ for forming
actual function names following `SymbolTable` auto-renaming scheme to avoid
duplicate symbols. Expects that all ops in the Payload IR have a
`SymbolTable` ancestor (typically true because of the top-level module).

#### Return Modes

Returns a handle to the list of outlined functions and a handle to the
corresponding function call operations in the same order as the operand
handle.

Produces a definite failure if outlining failed for any of the targets.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>func_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `function` | TransformHandleTypeInterface instance |
| `call` | TransformHandleTypeInterface instance |


### `transform.loop.peel` (transform::LoopPeelOp)

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

#### Return modes

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

#### Return Modes

Produces a definite failure if peeling fails.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>peel_front</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>fail_if_already_divisible</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to scf.for operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `peeled_loop` | TransformHandleTypeInterface instance |
| `remainder_loop` | TransformHandleTypeInterface instance |


### `transform.loop.pipeline` (transform::LoopPipelineOp)

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

#### Return modes

This operation ignores non-scf::For ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation pipeline
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.  The return handle points to only the subset of
successfully produced pipelined loops, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>iteration_interval</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>read_latency</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to scf.for operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.loop.promote_if_one_iteration` (transform::LoopPromoteIfOneIterationOp)

_Promote loop if it has one iteration_

Syntax:

```
operation ::= `transform.loop.promote_if_one_iteration` $target attr-dict `:` type($target)
```

Promotes the given target loop op if it has a single iteration. I.e., the
loop op is removed and only the body remains.

#### Return modes

This transform fails if the target is mapped to ops that are loops. Ops are
considered loops if they implement the `LoopLikeOpInterface`. Otherwise,
this transform always succeeds. The transform consumes the target handle and
modifies the payload.

Traits: `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.loop.unroll_and_jam` (transform::LoopUnrollAndJamOp)

_Unrolls and jam the given loop with the given unroll factor_

Syntax:

```
operation ::= `transform.loop.unroll_and_jam` $target attr-dict `:` type($target)
```

Unrolls & jams each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

#### Return modes

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.loop.unroll` (transform::LoopUnrollOp)

_Unrolls the given loop with the given unroll factor_

Syntax:

```
operation ::= `transform.loop.unroll` $target attr-dict `:` type($target)
```

Unrolls each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

#### Return modes

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.scf.take_assumed_branch` (transform::TakeAssumedBranchOp)

Syntax:

```
operation ::= `transform.scf.take_assumed_branch` $target
              (`take_else_branch` $take_else_branch^)?
              attr-dict
              `:` functional-type(operands, results)
```

Given an scf.if conditional, inject user-defined information that it is
always safe to execute only the if or else branch.

This is achieved by just replacing the scf.if by the content of one of its
branches.

This is particularly useful for user-controlled rewriting of conditionals
that exist solely to guard against out-of-bounds behavior.

At the moment, no assume or assert operation is emitted as it is not always
desirable. In the future, this may be controlled by a dedicated attribute.

#### Return modes

The transform only consumes its operand and does not produce any result.
The transform definitely fails if `take_else_branch` is specified and the
`else` region is empty.

Traits: `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>take_else_branch</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


## MemRef Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.td)

### `transform.apply_patterns.memref.alloc_to_alloca` (transform::ApplyAllocToAllocaOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.alloc_to_alloca` (`size_limit` `(` $size_limit^ `)`)? attr-dict
```

Collects patterns to rewrite scoped dynamic allocation (`alloc`/`dealloc`
pairs) into automatic allocation (`alloca`) in the same scope, for memrefs
of static shape.

The `size_limit` attribute controls the maximum allocated memory (in bytes,
subject to data layout) for which the pattern applies.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>size_limit</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>


### `transform.apply_patterns.memref.expand_ops` (transform::ApplyExpandOpsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.expand_ops` attr-dict
```

Collects patterns to rewrite ops within the memref dialect.

- Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
  AtomicRMWOpLowering pattern, e.g. with "minf" or "maxf" attributes, to
  `memref.generic_atomic_rmw` with the expanded code.
- Converts `memref.reshape` that has a target shape of a statically-known
  size to `memref.reinterpret_cast`.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.memref.expand_strided_metadata` (transform::ApplyExpandStridedMetadataPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.expand_strided_metadata` attr-dict
```

Collects patterns for expanding memref operations that modify the metadata
(sizes, offset, strides) of a memref into easier to analyze constructs.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.memref.extract_address_computations` (transform::ApplyExtractAddressComputationsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.extract_address_computations` attr-dict
```

Collects patterns for extracting address computations from operations
with memory accesses such that these memory accesses use only a base
pointer.

For instance,
```mlir
memref.load %base[%off0, ...]
```

Will be rewritten in:
```mlir
%new_base = memref.subview %base[%off0,...][1,...][1,...]
memref.load %new_base[%c0,...]
```

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.memref.fold_memref_alias_ops` (transform::ApplyFoldMemrefAliasOpsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.fold_memref_alias_ops` attr-dict
```

Collects patterns for folding memref aliasing ops (memref.subview) into
consumer load/store ops (affine.load, memref.load, nvgpu.ldmatrix,
vector.load, vector.transfer_read, affine.store, memref.store, etc.) and
other ops (e.g., memref.subview).

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims` (transform::ApplyResolveRankedShapedTypeResultDimsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims` attr-dict
```

Collects patterns that resolve `memref.dim` operations with values that are
defined by operations that implement the `ReifyRankedShapedTypeOpInterface`,
in terms of shapes of its input operands.

Interfaces: `PatternDescriptorOpInterface`


### `transform.memref.alloca_to_global` (transform::MemRefAllocaToGlobalOp)

Syntax:

```
operation ::= `transform.memref.alloca_to_global` $alloca attr-dict `:` functional-type(operands, results)
```

Inserts a new `memref.global` for each provided `memref.alloca` into the
nearest symbol table (e.g., a `builtin.module`) and replaces it with a
`memref.get_global`. This is useful, for example, for allocations that
should reside in the shared memory of a GPU, which have to be declared as
globals.

#### Example

Consider the following transform op:

```mlir
%get_global, %global =
    transform.memref.alloca_to_global %alloca
      : (!transform.op<"memref.alloca">)
        -> (!transform.any_op, !transform.any_op)
```

and the following input payload:

```mlir
module {
  func.func @func() {
    %alloca = memref.alloca() : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

then applying the transform op to the payload would result in the following
output IR:

```mlir
module {
  memref.global "private" @alloc : memref<2x32xf32>
  func.func @func() {
    %alloca = memref.get_global @alloc : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

#### Return modes

Succeeds always. The returned handles refer to the `memref.get_global` and
`memref.global` ops that were inserted by the transformation.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `alloca` | Transform IR handle to memref.alloca operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `getGlobal` | TransformHandleTypeInterface instance |
| `global` | TransformHandleTypeInterface instance |


### `transform.memref.erase_dead_alloc_and_stores` (transform::MemRefEraseDeadAllocAndStoresOp)

Syntax:

```
operation ::= `transform.memref.erase_dead_alloc_and_stores` $target attr-dict `:` functional-type($target, results)
```

This applies memory optimization on memref. In particular it does store to
load forwarding, dead store elimination and dead alloc/alloca elimination.

#### Return modes

This operation applies a set of memory optimization on the whole region of
the operand.

The transformation does not consume the target handle. It modifies the
payload. Dead allocations, loads and stores are silently dropped from all
mappings.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.memref.make_loop_independent` (transform::MemRefMakeLoopIndependentOp)

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

#### Return modes

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:
- memref.alloca: The returned handle points to the memref.subview op.

This transform op consumes the target handle and produces a result handle.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_loops</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.memref.multibuffer` (transform::MemRefMultiBufferOp)

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

#### Return modes

This operation returns the new allocation if multi-buffering
succeeds, and failure otherwise.

Traits: `FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is positive</td></tr>
<tr><td><code>skip_analysis</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to memref.alloc operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter` (transform::MemrefToLLVMTypeConverterOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter` attr-dict
```

This operation provides an "LLVMTypeConverter" that lowers memref types to
LLVM types.

The type converter can be customized as follows:
- `use_aligned_alloc`: Use aligned_alloc in place of malloc for heap
  allocations.
- `index_bitwidth`: Bitwidth of the index type, "0" indicates the size of a
  machine word.
- `use_generic_functions`: Use generic allocation and deallocation functions
  instead of the classic "malloc", "aligned_alloc" and "free" functions.
// TODO: the following two options don't really make sense for 
// memref_to_llvm_type_converter specifically.
// We should have a single to_llvm_type_converter.
- `use_bare_ptr_call_conv`: Replace FuncOp's MemRef arguments with bare 
  pointers to the MemRef element types.
- `data-layout`: String description (LLVM format) of the data layout that is
  expected on the produced module.

Interfaces: `TypeConverterBuilderOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>use_aligned_alloc</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>index_bitwidth</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>use_generic_functions</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>use_bare_ptr_call_conv</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>data_layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>


## PDL (extension) Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.td)

### `transform.pdl_match` (transform::PDLMatchOp)

_Finds ops that match the named PDL pattern_

Syntax:

```
operation ::= `transform.pdl_match` $pattern_name `in` $root attr-dict `:` functional-type(operands, results)
```

Find Payload IR ops nested within the Payload IR op associated with the
operand that match the PDL pattern identified by its name. The pattern is
expected to be defined in the closest surrounding `WithPDLPatternsOp`.

Produces a Transform IR value associated with the list of Payload IR ops
that matched the pattern. The order of results in the list is that of the
Operation::walk, clients are advised not to rely on a specific order though.
If the operand is associated with multiple Payload IR ops, finds matching
ops nested within each of those and produces a single list containing all
of the matched ops.

The transformation is considered successful regardless of whether some
Payload IR ops actually matched the pattern and only fails if the pattern
could not be looked up or compiled.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pattern_name</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `matched` | TransformHandleTypeInterface instance |


### `transform.with_pdl_patterns` (transform::WithPDLPatternsOp)

_Contains PDL patterns available for use in transforms_

Syntax:

```
operation ::= `transform.with_pdl_patterns` ($root^ `:` type($root))? attr-dict-with-keyword regions
```

This op contains a set of named PDL patterns that are available for the
Transform dialect operations to be used for pattern matching. For example,
PDLMatchOp can be used to produce a Transform IR value associated with all
Payload IR operations that match the pattern as follows:

```mlir
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  pdl.pattern @my_pattern : benefit(1) {
    %0 = pdl.operation //...
    // Regular PDL goes here.
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %1 = pdl_match @my_pattern in %arg1
    // Use %1 as handle
  }
}
```

Note that the pattern is expected to finish with a `pdl.rewrite` terminator
that points to the custom rewriter named "transform.dialect". The rewriter
actually does nothing, but the transform application will keep track of the
operations that matched the pattern.

This op is expected to contain `pdl.pattern` operations and exactly one
another Transform dialect operation that gets executed with all patterns
available. This op is a possible top-level Transform IR op, the argument of
its entry block corresponds to either the root op of the payload IR or the
ops associated with its operand when provided.

Traits: `NoTerminator`, `PossibleTopLevelTransformOpTrait`, `SymbolTable`

Interfaces: `MemoryEffectOpInterface`, `OpAsmOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `root` | TransformHandleTypeInterface instance |


## Structured (Linalg) Match Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.td)

### `transform.match.structured.body` (transform::MatchStructuredBodyOp)

_Checks if the body of the structured op satisfies some criteria_

Syntax:

```
operation ::= `transform.match.structured.body` $operand_handle attr-dict `:` type($operand_handle)
```

Checks if the body of the structured payload op satisfies one of the
following mutually exclusive criteria specified by attributes:

  * `reduction_position`: the body of the structured payload op implements
    a reduction of the `n`-th operand (`n` is the value of the attribute)
    using a single combiner operation;

  * `passthrough`: the body of the structured payload op only forwards
    inputs to the outputs (copy or broadcast).

  * `elementwise`: the body of the structured payload op represents an
    elementwise operation.

  * `contraction`: the body of the structured payload op is a contraction
    of the form `<red>(<elem>(bbarg0, bbarg1), bbarg2)` where `<elem>` and
    `<red>` are binary operations whose names are specified in the attribute
    and operands can be permuted and optionally forwarded through a chain of
    unary side effect-free operations.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operation body satisfies the specified criteria, produces a
silenceable failure otherwise. Produces a definite failure if the operand is
not associated with a single payload op.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction_position</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>passthrough</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>elementwise</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>contraction</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |


### `transform.match.structured.classify_contraction_dims` (transform::MatchStructuredClassifyContractionDimsOp)

_Checks if an operation has contraction-like dimensions and returns them_

Syntax:

```
operation ::= `transform.match.structured.classify_contraction_dims` $operand_handle attr-dict `:` functional-type(operands, results)
```

Checks if the structured payload op has contraction-like dimensions as
follows:

  C(batch, m, n) += A(batch, m, k) * B(batch, k, n)

That is:

  - 'batch' are parallel dimensions used in inputs and result;
  - 'm' are parallel dimensions used in the LHS and result;
  - 'n' are parallel dimensions used in rhe RHS and result;
  - 'k' are reduction dimensions present only in LHS and RHS.

Note that this doesn't check the operation in the body.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operation has the contraction-like dimensions, produces a
silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `batch` | TransformParamTypeInterface instance |
| `m` | TransformParamTypeInterface instance |
| `n` | TransformParamTypeInterface instance |
| `k` | TransformParamTypeInterface instance |


### `transform.match.structured.classify_convolution_dims` (transform::MatchStructuredClassifyConvolutionDimsOp)

_Checks if an operation has convolution-like dimensions and returns them_

Syntax:

```
operation ::= `transform.match.structured.classify_convolution_dims` $operand_handle attr-dict `:` functional-type(operands, results)
```

Checks if the structured payload op has convolution-like dimensions as
follows:

  C(batch, depth, oi, oc) += A(batch, depth, oi, ic) * B(fl, depth, ic, oc)

That is:

  - 'batch' are parallel dimensions used in the input and result;
  - 'output_image' ('oi') are parallel dimensions used in the input and result;
  - 'output_channel' ('oc') are parallel dimensions used in the filter and result;
  - 'filter_loop' ('fl') are reduction dimensions representing the dimensions of the sliding window;
  - 'input_channel' ('ic') are reduction dimensions present only in the input and filter.
  - 'depth' ('ic') are parallel dimensions present in the input, filter, and output.

Additionally this will match stride and dilation information for the convolution:
  - 'strides' are the static strides per convolution window dimension;
  - 'dilations' are the static dilations per convolution window dimension.

Note that this doesn't check the operation in the body.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operation has the convolution-like dimensions, produces a
silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `batch` | TransformParamTypeInterface instance |
| `output_image` | TransformParamTypeInterface instance |
| `output_channel` | TransformParamTypeInterface instance |
| `filter_loop` | TransformParamTypeInterface instance |
| `input_channel` | TransformParamTypeInterface instance |
| `depth` | TransformParamTypeInterface instance |
| `strides` | TransformParamTypeInterface instance |
| `dilations` | TransformParamTypeInterface instance |


### `transform.match.structured.dim` (transform::MatchStructuredDimOp)

_Checks if the dimensions of the structured op satisfy some criteria_

Syntax:

```
operation ::= `transform.match.structured.dim` $operand_handle `[`custom<TransformMatchDims>($raw_dim_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

Checks if the dimensions (loop ranges) of the structured payload op satisfy
the criteria specified as attributes. May capture the numeric value of the
dimension into a parameter that it returns.


 The following dimension specifications are supported:

  * `all`: all dimensions are checked and captured;
  * list of integers: the listed dimensions are checked and captured;
  * `except(` list of integers `)`: all dimensions except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last dimension and
`except(-1)` means all dimensions but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.


The following mutually exclusive conditions are available as unit
attributes:

  * `parallel`: the dimension corresponds to a parallel loop;
  * `reduction`: the dimension corresponds to a reduction loop.

If the result type is specified, associates the parameter with the (static)
values of dimensions in the same order as listed and preserving the natural
order for `all` and `except`. Specifically, if `-1, -2` are specified, the
parameter will be associated with the value of the second-to-last dimension
followed by the last dimension. If the dimension is dynamic, the parameter
will contain a negative value corresponding to kDynamic in C++.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the specified dimensions satisfy the specified criteria,
produces a silenceable failure otherwise. Produces a definite failure if
the operand is not associated with a single payload op.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>raw_dim_list</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>is_inverted</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>is_all</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>parallel</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>reduction</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformParamTypeInterface instance |


### `transform.match.structured.elemental_bitwidth` (transform::MatchStructuredElementalBitwidthOp)

_Captures the bitwidth of the value's elemental type as a parameter_

Syntax:

```
operation ::= `transform.match.structured.elemental_bitwidth` $operand_handle attr-dict `:` functional-type(operands, results)
```

Produces a transform dialect parameter associated with the bitwidth of the
elemental type of the payload value passed as the operand.
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operand is associated with exactly one payload value of
`ShapedType`. Produces a silenceable failure otherwise.

Traits: `SingleValueMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformValueHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformParamTypeInterface instance |


### `transform.match.structured.init` (transform::MatchStructuredInitOp)

_Captures init operand(s) of a structured operation_

Syntax:

```
operation ::= `transform.match.structured.init` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

Produces a transform dialect value depending on the result type:
  - If the result type is a value handle, it will be associated with the init
    operand(s) of the payload operation associated with the operand handle.
  - If the result type is an operation handle, it will be associated with the
    operation defining the init operand(s) of the payload operation associated
    with the operand handle.
  - If the result type is an affine map parameter type, it will be associated
    with the indexing map that corresponds to the init operand(s) of the
    payload operation associated with the operand handle.

For example, given the following operation:

```mlir
%arg3 = linalg.fill
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

in case of a successful match for init operand 0 this operation will return,
for each of the respective cases above:

  - A handle to `%arg3` if the result is a value handle.
  - A handle to `linalg.fill` if the result is an operation handle.
  - A parameter containing the result map of the matrix multiplication, i.e.
    `affine_map<(d0, d1, d2) -> (d0, d1)>` if the result is an affine
    map parameter.

The match succeeds if the conditions specified as attributes succeed.


 The following init specifications are supported:

  * `all`: all inits are checked and captured;
  * list of integers: the listed inits are checked and captured;
  * `except(` list of integers `)`: all inits except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last init and
`except(-1)` means all inits but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.



This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if all init(outs) indexes are in bounds, produces a silenceable
failure otherwise. Additionally, when the result is an operation handle,
produces a silenceable failure if the init(outs) specification defines
more than one init(outs) or if the operand is not an operation result.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>raw_position_list</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>is_inverted</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>is_all</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>permutation</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>projected_permutation</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | transform operation or value handle or  |


### `transform.match.structured.input` (transform::MatchStructuredInputOp)

_Captures input operand(s) of a structured operation_

Syntax:

```
operation ::= `transform.match.structured.input` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

Produces a transform dialect value depending on the result type:

  - If the result type is a value handle, it will be associated with the input
    operand(s) of the payload operation associated with the operand handle.
  - If the result type is an operation handle, it will be associated with the
    operation defining the input operand(s) of the payload operation associated
    with the operand handle.
  - If the result type is an affine map parameter type, it will be associated
    with the indexing map that corresponds to the input operand(s) of the
    payload operation associated with the operand handle.

For example, given the following operation:

```mlir
%arg1 = some.op
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

in case of a successful match for operand 0 this operation will return, for
each of the respective cases above:

  - A handle to `%arg1` if the result is a value handle.
  - A handle to `some.op` if the result is an operation handle.
  - A parameter containing the LHS map of the matrix multiplication, i.e.
    `affine_map<(d0, d1, d2) -> (d0, d2)>` if the result is an affine
    map parameter.

The match succeeds if the conditions specified as attributes succeed.


 The following input specifications are supported:

  * `all`: all inputs are checked and captured;
  * list of integers: the listed inputs are checked and captured;
  * `except(` list of integers `)`: all inputs except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last input and
`except(-1)` means all inputs but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.



This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if all input indexes are in bounds, produces a silenceable failure
otherwise. Additionally, when the result is an operation handle, produces a
silenceable failure if the input specification defines more than one input
or if the operand is not an operation result.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>raw_position_list</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>is_inverted</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>is_all</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>permutation</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>projected_permutation</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | transform operation or value handle or  |


### `transform.match.structured.num_inits` (transform::MatchStructuredNumInitsOp)

_Captures the number of init(outs) operands of a structuredoperation as parameter_

Syntax:

```
operation ::= `transform.match.structured.num_inits` $operand_handle attr-dict `:` functional-type(operands, results)
```

Produces a transform dialect parameter value associated with an integer
attribute containing the number of init(outs) operands of the payload
operation associated with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformParamTypeInterface instance |


### `transform.match.structured.num_inputs` (transform::MatchStructuredNumInputsOp)

_Captures the number of input operands of a structured operation as parameter_

Syntax:

```
operation ::= `transform.match.structured.num_inputs` $operand_handle attr-dict `:` functional-type(operands, results)
```

Produces a transform dialect parameter value associated with an integer
attribute containing the number of input operands of the payload operation
associated with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformParamTypeInterface instance |


### `transform.match.structured` (transform::MatchStructuredOp)

_Matches a structured (linalg) operation with additional conditions_

Syntax:

```
operation ::= `transform.match.structured` (`failures` `(` $failure_propagation_mode^ `)`)?$current `:` custom<SemiFunctionType>(type($current), type($outputs))attr-dict-with-keyword regions
```

Checks if the payload operation associated with the operand handle is a
structured operation, that is, an operation that implements
`LinalgOpInterface`, and that all conditions listed in the body of this
operation are satisfied. Produces a silenceable failure if the payload
operation is not structured.

The transform operations nested in the body region are applied one by one.
If any of them produces a failure, silenceable or definite, the following
operations are not applied. If the failure propagation mode is "propagate",
silenceable failures are forwarded as the result of this operation. If it is
"suppress", they are ignored and this operation immediately succeeds.
Definite failures are always propagated immediately.

In case of success, the transform values produced by this operation are
associated with the same payload as the operands of the block terminator. If
any of the nested operations produced a silenceable failure, regardless of
the failure propagation mode, the transform values produced by this
operation that correspond to the already defined terminator operands are
associated with the same payload as the already defined terminator operands.
Other values produced by this operation are associated with empty payloads.

If the failure propagation mode is not specified, it is considered
"propagate" by default. The "suppress" mode can be used to specify optional
matches.

#### Return modes

This operation only reads all operand handles and produces all resulting
handles. It succeeds in "propagate" mode if the payload operation is a
structured operation and if all the nested operations succeed. It succeeds
in "suppress" mode as long as the operand handle is associated with exactly
one payload operation. It produces a definite failure when the handle is
not associated with exactly one payload operation.

Traits: `SingleBlockImplicitTerminator<::mlir::transform::MatchStructuredYieldOp>`, `SingleBlock`, `SingleOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>failure_propagation_mode</code></td><td>::mlir::transform::FailurePropagationModeAttr</td><td>Silenceable error propagation policy</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `current` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of any transform handle or parameter |


### `transform.match.structured.rank` (transform::MatchStructuredRankOp)

_Captures the rank of a structured operation as parameter_

Syntax:

```
operation ::= `transform.match.structured.rank` $operand_handle attr-dict `:`
              custom<SemiFunctionType>(type($operand_handle), type($rank), "false")
```

Produces a transform dialect parameter value associated with an integer
attribute containing the rank of the structured payload operation associated
with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `rank` | TransformParamTypeInterface instance |


### `transform.match.structured.result` (transform::MatchStructuredResultOp)

_Captures the result of a structured payload operation in an op or value handle_

Syntax:

```
operation ::= `transform.match.structured.result` $operand_handle `[` $position `]` (`any` $any^)? (`single` $single^)?attr-dict `:` functional-type(operands, results)
```

Produces a transform dialect value handle associated with the payload value
defined as a result of the payload operation associated with the operand
handle, or an operation handle to an operation using the produced result
with additional constraints specified by the attributes as follows.

  * If `any` is specified, binds the resulting handle to any operation using
    the result and succeeds.
  * If `single` is specified, binds the resulting handle to the only
    operation using the result or fails if there is more than one (or no)
    such operation.

The number of the result is specified as `position` attribute. It may take
positive and negative values. Negative values are interpreted as counting
results from backwards, e.g., `-1` means the last result and `-2` means the
second-to-last result. In any case, the position must be in bounds for the
given payload operation. A silenceable failure is produced for out-of-bounds
positions.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.


#### Return modes

Succeeds if the position is in bounds and if the user operation could be
found when requested. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>position</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>any</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>single</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operand_handle` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | transform operation or value handle |


### `transform.match.structured.yield` (transform::MatchStructuredYieldOp)

_Terminator for transform.match.structured blocks_

Syntax:

```
operation ::= `transform.match.structured.yield` $handles attr-dict (`:` type($handles)^)?
```

Forwards the payload association from the operands to the results of the
parent op. Always succeeds.

Traits: `Terminator`

Interfaces: `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handles` | variadic of any transform handle or parameter |


## Structured (Linalg) Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td)

### `transform.apply_patterns.linalg.decompose_pack_unpack` (transform::ApplyDecomposeTensorPackUnpackPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.decompose_pack_unpack` attr-dict
```

Collect patterns to decompose linalg.pack and linalg.unpack into e.g.
tensor::PadOp, linalg::transposeOp Ops. Requires all outer dims to be unit.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.decompose_pad` (transform::ApplyDecomposeTensorPadPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.decompose_pad` attr-dict
```

Collect patterns to decompose tensor.pad into e.g. tensor::EmptyOp,
linalg::FillOp and tensor::InsertSliceOp.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.erase_unnecessary_inputs` (transform::ApplyEraseUnnecessaryInputsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.erase_unnecessary_inputs` attr-dict
```

Collects patterns that promote inputs to outputs and remove unused inputs of
`linalg.generic` ops.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.fold_add_into_dest` (transform::ApplyFoldAddIntoDestPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_add_into_dest` attr-dict
```

Collects patterns to replace linalg.add when destination passing suffices
for achieving the sum.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.fold_into_pack_and_unpack` (transform::ApplyFoldIntoPackAndUnpackPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_into_pack_and_unpack` attr-dict
```

Indicates that operations like tensor.pad and tensor.extract_slice should
be folded into linalg.pack and linalg.unpack operations, respectively.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.fold_pack_unpack_into_empty` (transform::ApplyFoldPackUnpackIntoEmptyPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_pack_unpack_into_empty` attr-dict
```

// TODO:

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>fold_single_use_only</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>


### `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` (transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` attr-dict
```

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via reassociative reshape ops.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` (transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` attr-dict
```

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via rank-reducing slices.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.pad_vectorization` (transform::ApplyPadVectorizationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.pad_vectorization` attr-dict
```

Apply patterns that vectorize tensor.pad.

These patterns rewrite tensor.pad Ops using vector.transfer_read and
vector.transfer_write operations. This is done either by:
  1. Folding tensor.pad with an existing vector.transfer_read /
  vector.transfer_write Op (generated prior to running these patterns). 
  2. Rewriting it (when matched together with q tensor.insert_slice
  consumer Op) as a vector.transfer_read + vector.transfer_write pair.

In both cases, these patterns look at producers and consumers for the
matched tensor.pad Op to find opportunities for vectorization.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.linalg.tiling_canonicalization` (transform::ApplyTilingCanonicalizationPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.tiling_canonicalization` attr-dict
```

Collects canonicalization patterns relevant to apply after tiling patterns.

Interfaces: `PatternDescriptorOpInterface`


### `transform.structured.bufferize_to_allocation` (transform::BufferizeToAllocationOp)

Syntax:

```
operation ::= `transform.structured.bufferize_to_allocation` $target attr-dict `:` type($target)
```

This transform bufferizes the targeted operation and materializes the
result in a new allocation. It replaces all original uses of the target
result with the newly allocated buffer, wrapped in a
`bufferization.to_tensor` op. It returns a handle to the newly allocated
buffer. Furthermore, it returns a handle that is mapped to all newly created
ops.

Only bufferizable ops are that bufferize to a memory write or have an
aliasing OpOperand (and do not themselves bufferize to an allocation) are
supported. They are bufferized using their BufferizableOpInterface
implementation. E.g.:

```
%0 = tensor.insert %f into %dest[%pos] : tensor<10xf32>
```

Is bufferized to:

```
%alloc = memref.alloc() : memref<10xf32>
bufferization.materialize_in_destination %dest in %alloc
memref.store %f, %alloc[%pos] : memref<10xf32>
%0 = bufferization.to_tensor %alloc restrict writable : memref<10xf32>
```

Selected ops that bufferize to an allocation (or need special handling) are
also supported:
- `tensor.pad` is lowered to an allocation, followed by a `linalg.fill` and
  and a buffer copy (all on memrefs).
- `vector.mask` is bufferized together with its region. The allocation is
  placed in front of the `vector.mask` op.

An optional memory space attribute can be specified for the materialized
buffer allocation.

If a memory copy is needed, a "bufferization.materialize_in_destination" is
used when possible. This is an op with tensor semantics that will bufferize
to a memory copy later. Which concrete op will be used for the memory copy
is up to the bufferization framework. Alternatively, a custom memcpy op can
be specified via `memcpy_op`. Currently supported are "memref.copy" and
"linalg.copy". In that case, the source of each memcpy must not have a
custom memory space. Furthermore, because the future buffer layout unknown
for a given tensor, a fully dynamic layout is assumed for best
compatibility. Users should use "bufferization.materialize_in_destination"
when possible.

"memref.alloc" is used for new buffer allocations. The buffer is deallocated
at the end of the block if the "emit_dealloc" attribute is present. If this
attribute is not present, the allocated memory will be leaked. However,
running the `-buffer-deallocation-pipeline` after all bufferization is done
will properly insert the corresponding deallocation(s). Custom allocation
ops can be specified via `alloc_op`. Currently supported are "memref.alloc"
and "memref.alloca". In case of a "memref.alloca", the buffer is not
deallocated.

If `bufferize_destination_only` is set, only the destination operands of the
op are bufferized to a new memory allocation, but not the op itself.

#### Return modes

This operation consumes the `target` handle and produces the
`allocated_buffer` and `new_ops` handles. It always succeeds.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>memory_space</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>memcpy_op</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>alloc_op</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>bufferize_destination_only</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>emit_dealloc</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `allocated_buffer` |  |
| `new_ops` |  |


### `transform.structured.continuous_tile_sizes` (transform::ContinuousTileSizesOp)

Syntax:

```
operation ::= `transform.structured.continuous_tile_sizes` $target attr-dict `:` custom<ContinuousTileSizeTypes>(type($target), type($tile_sizes), type($chunk_sizes))
```

This transform emits the IR computing the list of (1) exponentially
diminishing tile sizes that are powers of 2; and (2) the corresponding
chunk-sizes the target op should be split into along the given dimension.

For example, for `target_size` 9, and `dimension` 0 for the following
linalg op as target

```
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                  outs(%arg2: tensor<25x25xf32>)
```

the first result `tile_sizes` will be a list of diminishing tile sizes
9, 4, 2, 1; and the second result will be a list of chunk sizes
18, 4, 2, 1 that the corresponding dimension should be split into.

After the target op has been split along the given dimension (for example
using multiway split), each chunk can be tiled with the corresponding tile
size in the `tile_sizes` list generated as a result of this op.

Specifying the output type as !transform.param<i64> will cause `tile_sizes`
and `chunk_sizes` to be computed statically and not dynamically.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is non-negative</td></tr>
<tr><td><code>target_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute whose value is non-negative</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `tile_sizes` | transform any param type or any handle type |
| `chunk_sizes` | transform any param type or any handle type |


### `transform.structured.convert_conv2d_to_img2col` (transform::ConvertConv2DToImg2ColOp)

Syntax:

```
operation ::= `transform.structured.convert_conv2d_to_img2col` $target attr-dict `:` functional-type($target, results)
```

Convert linalg.conv_2d_xxx into linalg.generic (for img2col packing)
and linalg.matmul.

A convolution operation can be written as a matrix-matrix multiplication by
unfolding the cross-correlation between input and filter and explicitly copy
overlapped sliding window inputs.

Consider 2D input X with single channel input and output and 2x2 filter W:
```
[x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
[x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
[.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
[.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
[.        ,  .       ,   .,      .     ]
[x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
```

The packed input data (img2col) is a matrix with |rows| = output spatial
size, |columns| = filter spatial size. To compute the output Y(i, j) we need
to calculate the dot product between filter window at input X(x, y)) and the
filter which will look like the following where r.h.s is the img2col matrix
and l.h.s is the flattned filter:
```
[x(0,0), x(0,1), x(1,0), x(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)]
[   .  ,    .  ,    .  ,    .  ]
```

In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
the N input. For the case where N > 1 its a batched matrxi-matrix
multplication.

Returns two handles:
- One on the operation that produces the img2col tensor.
- One on the final operation of the sequence that replaces the original
  convolution.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `img2col_tensor` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.convert_to_loops` (transform::ConvertToLoopsOp)

Syntax:

```
operation ::= `transform.structured.convert_to_loops` $target attr-dict `:` functional-type(operands, results)
```

For operations that implement the `TilingInterface`, and implement
the `generateScalarImplementation` method, lowers the operation to
loops. The return handle points to all generated loops.
Fails if the payload ops cannot be lowered to loops.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.structured.decompose_interface` (transform::DecomposeInterfaceOp)

Syntax:

```
operation ::= `transform.structured.decompose_interface` $target attr-dict `:` functional-type(operands, results)
```

TODO

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.decompose` (transform::DecomposeOp)

Syntax:

```
operation ::= `transform.structured.decompose` $target attr-dict `:` functional-type(operands, results)
```

Decomposes named complex operations, such as higher-dimensional
(depthwise) convolutions, into combinations of lower-dimensional equivalents
when possible.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle decompose
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure. The return handle points to only the subset of
successfully produced computational operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.decompose_winograd_op` (transform::DecomposeWinogradOp)

Syntax:

```
operation ::= `transform.structured.decompose_winograd_op` $target attr-dict `:` functional-type($target, results)
```

Decompose winograd operations. It will convert filter, input and output
transform operations into a combination of scf, tensor, and linalg
equivalent operations. Before applying this transform operations, users
need to tile winograd transform operations into supported sizes.

#### Return modes:

This operation fails if `target` is unsupported. Otherwise, the operation
succeeds and returns a handle of the sequence that replaces the original
operations.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.eliminate_empty_tensors` (transform::EliminateLinalgOpAnchoredEmptyTensorsOp)

Syntax:

```
operation ::= `transform.structured.eliminate_empty_tensors` $target attr-dict `:` type($target)
```

Try to eliminate all `tensor.empty` op uses that are anchored on a LinalgOp
within the targeted op.

This op is similar to `bufferization.eliminate_empty_tensors`, but specific
to LinalgOps.

`tensor.empty` ops cannot be bufferized. They can either be converted to
`bufferization.alloc_tensor` or replaced with another tensor (via this
transform). `tensor.empty` does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transform looks for `tensor.empty` ops where the SSA use-def chain of
the result ends in a supported LinalgOp (always following the aliasing
OpOperand/OpResult chain). The following LinalgOps are supported:
- Only parallel iterator types.
- The use-def chain ends in an input operand of the LinalgOp.
- The LinalgOp has an unused output operand with the same shape and
  indexing map.

Example:

```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%0)
%2 = linalg.generic ins(%1) outs(%dest) {
  ^bb0(%in: f32, %out: f32):
  // out not used
}
```

Is rewritten with:
```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%dest)
%2 = linalg.generic ins(%0) outs(%1) {
  ^bb0(%in: f32, %out: f32):
  // Use %out instead of %in
}
```

After this transformation, the "ins" operand has no uses inside the body of
the LinalgOp and can be folded away with existing cleanup patterns.
Afterwards, the tensor::EmptyOp can also fold away, so that the example can
bufferize without an allocation (in the absence of other conflicts).

#### Return modes

This transform reads the target handle and modifies the payload. It does
not produce any handle.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |


### `transform.structured.flatten_elementwise` (transform::FlattenElementwiseLinalgOp)

Syntax:

```
operation ::= `transform.structured.flatten_elementwise` $target attr-dict `:` functional-type($target, results)
```

Flattens the iteration space and (applicable) operands of elementwise
linalg ops to a single dimension.

Returns one handle:
- Flattened linalg operation.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.fuse_into_containing_op` (transform::FuseIntoContainingOp)

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

#### Return modes

If at least one producer could not be fused, this operation produces a
silenceable failure.  This is the case when tiling fails or when no
producer op could be found among the remaining producers that has at least
one use within the containing op. I.e., "producers" that are not consumed
within the containing op are rejected by this operation.

This operation consumes the producer handle.
This operation only reads the containing op handle.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `producer_op` | TransformHandleTypeInterface instance |
| `containing_op` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `fused_op` | TransformHandleTypeInterface instance |
| `new_containing_op` | TransformHandleTypeInterface instance |


### `transform.structured.fuse` (transform::FuseOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>tile_interchange</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>apply_cleanup</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |


### `transform.structured.generalize` (transform::GeneralizeOp)

Syntax:

```
operation ::= `transform.structured.generalize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

Transforms a named structured operation into the generic form with the
explicit attached region.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.  The return handle points to only the subset of
successfully produced equivalent generic operations, which can be empty or
contain the original ops if they were already in generic form.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.hoist_pad.build_packing_loop_nest` (transform::HoistPadBuildPackingLoopNestOp)

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

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

The return handle points to only the subset of successfully created packing
loop nests, which can be empty.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `loop` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `packing_loop` | TransformHandleTypeInterface instance |


### `transform.structured.hoist_pad` (transform::HoistPadOp)

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

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

If all the operations referred to by the `target` handle padproperly, the
transform succeeds. Otherwise the transform produces a silenceable failure.

The return handle points to only the subset of successfully hoisted
tensor.pad operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_loops</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.hoist_redundant_vector_broadcasts` (transform::HoistRedundantVectorBroadcastsOp)

Syntax:

```
operation ::= `transform.structured.hoist_redundant_vector_broadcasts` $target attr-dict `:` functional-type(operands, results)
```

Hoist vector.extract / vector.broadcasts pairs out of immediately
enclosing scf::ForOp iteratively.

#### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.hoist_redundant_vector_transfers` (transform::HoistRedundantVectorTransfersOp)

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

#### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>verify_non_zero_trip</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.insert_slice_to_copy` (transform::InsertSliceToCopyOp)

Syntax:

```
operation ::= `transform.structured.insert_slice_to_copy` $target attr-dict `:` functional-type(operands, results)
```

Targeted rewrite of an tensor.insert_slice to linalg.copy.
This is useful to materialize copies explicitly before bufferization and
transform them, avoiding the need to rediscover them after bufferization.

If the insert_slice source is already a linalg.copy, only return the source
op (i.e. do not create an additional linalg.copy op).

#### Return modes:

The operation always succeeds and returns a handle to the relevant
linalg.copy op.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.interchange` (transform::InterchangeOp)

Syntax:

```
operation ::= `transform.structured.interchange` $target
              (`iterator_interchange` `=` $iterator_interchange^)? attr-dict
              `:` custom<SemiFunctionType>(type($target), type($transformed), "false")
```

Interchanges the iterators of the operations pointed to by the target handle
using the iterator interchange attribute.

#### Return modes

This operation ignores non-linalg::Generic ops and drops them in the return.
This operation fails if the interchange attribute is invalid.
If all the operations referred to by the `target` handle interchange
properly, the transform succeeds.
If any interchange fails, the transform produces a definite failure.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>iterator_interchange</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute whose value is non-negative</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.linalg_copy_to_memref` (transform::LinalgCopyToMemrefOp)

Syntax:

```
operation ::= `transform.structured.linalg_copy_to_memref` $target attr-dict `:` functional-type(operands, results)
```

Targeted rewrite of a linalg.copy on memrefs to a memref.copy.
This is useful when bufferizing copies to a linalg.copy, later applying some
transformations, and then rewriting the copy into a memref.copy.
If the element types of the source and destination differ, or if the source
is a scalar, the transform produces a silenceable failure.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.lower_pack` (transform::LowerPackOp)

Syntax:

```
operation ::= `transform.structured.lower_pack` $target attr-dict `:` functional-type(operands, results)
```

Rewrite a linalg.pack into tensor.pad + tensor.expand_shape + linalg.transpose.

#### Return modes

This operation ignores non-pack ops and drops them in the return. This
operation produces a silenceable failure if the rewrite fails for any
reason. If all the operations referred to by the `target` are rewritten,
the transform succeeds. Return handles to the newly produced pad,
expand_shape and transpose ops.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowerPadLikeWithInsertSlice</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to linalg.pack operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `pad_op` | Transform IR handle to tensor.pad operations |
| `expand_shape_op` | Transform IR handle to tensor.expand_shape operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |


### `transform.structured.lower_unpack` (transform::LowerUnPackOp)

Syntax:

```
operation ::= `transform.structured.lower_unpack` $target attr-dict `:` functional-type(operands, results)
```

Lower a linalg.unpack into empty + linalg.transpose + tensor.collapse_shape +
tensor.extract_slice.

#### Return modes

This operation ignores non-unpack ops and drops them in the return. This
operation produces a silenceable failure if the rewrite fails for any
reason. If all the operations referred to by the `target` are rewritten,
the transform succeeds. Return handles to the newly produced empty,
transpose, collapse_shape and extract_slice ops.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowerUnpadLikeWithExtractSlice</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | Transform IR handle to linalg.unpack operations |

#### Results:

| Result | Description |
| :----: | ----------- |
| `empty_op` | Transform IR handle to tensor.empty operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |
| `collapse_shape_op` | Transform IR handle to tensor.collapse_shape operations |
| `extract_slice_op` | Transform IR handle to tensor.extract_slice operations |


### `transform.structured.gpu.map_copy_to_threads` (transform::MapCopyToThreadsOp)

Syntax:

```
operation ::= `transform.structured.gpu.map_copy_to_threads` $target
              `total_num_threads` `=` $total_num_threads
              `desired_bit_alignment` `=` $desired_bit_alignment
              attr-dict
              `:` functional-type(operands, results)
```

Targeted mapping of a linalg.copy / tensor.pad operation on tensors to a GPU
thread mapping.

This operation implements a greedy heuristic that determines a good
distribution of threads to break down the copy/pad operation into.
The heuristic is driven by considerations related to the underlying
architecture for which good high-level decisions are needed assuming certain
hardware features. Relevant features are exposed via first-class attributes
to control the behavior of the transformation at a high level.

For now, a single heuristic is implemented and can be extended on a per-need
basis.

#### Return modes

This operation fails definitely if there is an unsupported op (i.e., not
linalg.copy / tensor.pad) among the targeted op. Otherwise, the operation
always succeeds and returns a handle to the relevant tiled linalg.copy /
tensor.pad op and the enclosing scf.forall op.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>total_num_threads</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>desired_bit_alignment</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `forall_op` | TransformHandleTypeInterface instance |
| `tiled_op` | TransformHandleTypeInterface instance |


### `transform.structured.match` (transform::MatchOp)

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

#### Return modes

This op traverses the ops nested under `target` and returns the handles to
all the operations that match the requirements.

This op fails if the target is not a handle to exactly one operation.
Otherwise it succeeds.

This operation does not consume the target handle and produces new handles:
it is a navigation op.

Traits: `NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>ops</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>interface</code></td><td>mlir::transform::MatchInterfaceEnumAttr</td><td>An interface to match</td></tr>
<tr><td><code>op_attrs</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
<tr><td><code>filter_result_type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
<tr><td><code>filter_operand_types</code></td><td>::mlir::ArrayAttr</td><td>type array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | TransformHandleTypeInterface instance |


### `transform.structured.multitile_sizes` (transform::MultiTileSizesOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>target_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>divisor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `low_size` | transform any param type or any handle type |
| `high_size` | transform any param type or any handle type |
| `split_point` | transform any param type or any handle type |


### `transform.structured.pack_greedily` (transform::PackGreedilyOp)

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

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
It returns the list of packed Linalg ops or the original op when all available
packing strategies failed to apply.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_matmul_packed_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with exactly 3 elements</td></tr>
<tr><td><code>matmul_padded_sizes_next_multiple_of</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with 0 or 3 elements</td></tr>
<tr><td><code>matmul_inner_dims_order</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute with exactly 3 elements</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `matmul_packed_sizes` | variadic of TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |


### `transform.structured.pack` (transform::PackOp)

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

#### Example

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

#### Return modes

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definite failure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_packed_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `packed_sizes` | variadic of TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |


### `transform.structured.pack_transpose` (transform::PackTransposeOp)

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

#### Return modes

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>outer_perm</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>inner_perm</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target_pack_or_un_pack_op` | TransformHandleTypeInterface instance |
| `target_linalg_op` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `packed_op` | TransformHandleTypeInterface instance |
| `pack_op` | TransformHandleTypeInterface instance |
| `un_pack_op` | TransformHandleTypeInterface instance |


### `transform.structured.pad` (transform::PadOp)

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

#### Return modes

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

#### Attributes:

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

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `pad_to_multiple_of` | variadic of transform any param type or any handle type |

#### Results:

| Result | Description |
| :----: | ----------- |
| `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |
| `copy` | TransformHandleTypeInterface instance |


### `transform.structured.pad_tiling_interface` (transform::PadTilingInterfaceOp)

Syntax:

```
operation ::= `transform.structured.pad_tiling_interface` $target
              `to`
              (`padding_sizes` custom<DynamicIndexList>($padding_sizes, $static_padding_sizes)^)?
              (`pad_to_multiple_of` $pad_to_multiple_of^)?
              attr-dict
              `:` functional-type(operands, results)
```

Pads the **iteration domain** of the operations pointed to by the target
handle using the options provided as operation attributes. Padding the
iteration domain induces a padding of the operands that is consistent
across the op semantics and, unlike for simple elementwise ops, may not be
trivially deducible or specifiable on operands only (e.g. convolutions).

The specification of `padding_sizes` follows that of `tile_sizes` during
tiling: the value "0" on a particular iterator encode "no padding". Like in
the case of tiling, an automatic completion by 0 to the operation rank
occurs.

This transformation returns a handle to the padded operation and to the
padding operation ("tensor.pad").

TODO: in the future this should be moved out of a specific Linalg
implementation file and into a more general "Structured" file.

#### Return modes

This operation ignores non-IndexingMapOpInterface ops and drops them in the
return. In the future, this operation will support all TilingInterfaceOps
for which the contract between iteration domain and operands can be 
reified.    

This operation may produce a definite failure if the padding fails for any
reason.

If all the operations referred to by the `target` handle pad properly, the
transform succeeds. Otherwise the transform produces a silenceable failure.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>padding_values</code></td><td>::mlir::ArrayAttr</td><td>array attribute</td></tr>
<tr><td><code>static_padding_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>pad_to_multiple_of</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `padding_sizes` | variadic of transform any param type or any handle type |

#### Results:

| Result | Description |
| :----: | ----------- |
| `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |


### `transform.structured.promote` (transform::PromoteOp)

Syntax:

```
operation ::= `transform.structured.promote` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

Promotes the specified operands of the target into a separate memory buffer.

At this point, this transform does not allow customizing alloc/dealloc
functions nor the behavior on copy in/out operations.

#### Return modes

This operation applies to a single Linalg op that satisfies the
`promoteSubviewsPrecondition`, otherwise it fails.

If the operations referred to by the `target` handle promote
properly, the transform succeeds.

When successful, the return handle points to the $target operation that
was modified inplace.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>operands_to_promote</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>use_full_tile_buffers</code></td><td>::mlir::ArrayAttr</td><td>1-bit boolean array attribute</td></tr>
<tr><td><code>use_full_tiles_by_default</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_original_subview_size</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_alloca</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>memory_space</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>mapping</code></td><td>::mlir::ArrayAttr</td><td>Device Mapping array attribute</td></tr>
<tr><td><code>alignment</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.replace` (transform::ReplaceOp)

Syntax:

```
operation ::= `transform.structured.replace` $target attr-dict-with-keyword regions `:`
              custom<SemiFunctionType>(type($target), type($replacement), "false")
```

Replace all `target` payload ops with the single op that is contained in
this op's region. All targets must have zero arguments and must be isolated
from above.

This op is for debugging/experiments only.

#### Return modes

This operation consumes the `target` handle.

Traits: `HasOnlyGraphRegion`, `IsolatedFromAbove`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `replacement` | TransformHandleTypeInterface instance |


### `transform.structured.rewrite_in_destination_passing_style` (transform::RewriteInDestinationPassingStyleOp)

Syntax:

```
operation ::= `transform.structured.rewrite_in_destination_passing_style` $target attr-dict
              `:` functional-type($target, results)
```

Rewrite a supported tensor operation that is not in destination-passing style
into a form that is in destination-passing style.
Currently supported operations are:
  - tensor.pad
  - tensor.generate
  - tensor.from_elements
This dichotomy hints at a future interface, for now the implementation just
switches between different implementation.

#### Return modes

This operation ignores non-unsupported ops and drops them from the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.
The return handle points to a subset of successfully produced operations:
  - `tensor.pad` case, the returned handle points to the tensor.insert_slice.
  - `tensor.generate` case, the returned handle points to the linalg.generic.
  - `tensor.from_elements` case, the returned handle points to the last
    `tensor.insert`.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.scalarize` (transform::ScalarizeOp)

Syntax:

```
operation ::= `transform.structured.scalarize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($result), "false")
```

Indicates that ops of a specific kind in the given function should be
scalarized (i.e. their dynamic dimensions tiled by 1).

#### Return modes:

This operation ignores non-Linalg ops and drops them in the return.
This operation produces definite failure if the scalarization fails for any
reason.
If all the operations referred to by the `target` handle scalarize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

The return handle points to only the subset of successfully produced
tiled-by-1 operations, which can be empty.

This operation does not return handles to the tiled loop.
We make this design choice because it is hard to know ahead of time the
number of loops that will be produced (it depends on the number of dynamic
dimensions after multiple transformations have been applied).
Loops can always be recovered by navigating from the tiled operations if
needed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | TransformHandleTypeInterface instance |


### `transform.structured.specialize` (transform::SpecializeOp)

Syntax:

```
operation ::= `transform.structured.specialize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

Transforms a generic operation into the equivalent named form.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return. If all
the operations referred to by the `target` handle specialize, the transform
succeeds; otherwise, the operation produces a silenceable failure.  The return
handle points to only the subset of successfully produced equivalent named
operations, which can be empty or contain the original ops if they were already
in named form. The supported specialization to named Linalg operations are:
- linalg.copy of any rank.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.split` (transform::SplitOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>static_chunk_sizes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>multiway</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_chunk_sizes` | transform any param type or any handle type |

#### Results:

| Result | Description |
| :----: | ----------- |
| `split_list` | TransformHandleTypeInterface instance |


### `transform.structured.split_reduction` (transform::SplitReductionOp)

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

#### Return modes

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

#### Example (default: `use_scaling_algorithm = false, use_alloc = false`):

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

#### Example (`use_scaling_algorithm = true, use_alloc = true`):

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

#### Example:

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>split_factor</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>insert_split_dimension</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>inner_parallel</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_scaling_algorithm</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>use_alloc</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `init_or_alloc_op` | TransformHandleTypeInterface instance |
| `fill_op` | TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |


### `transform.structured.tile_reduction_using_for` (transform::TileReductionUsingForOp)

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

#### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `for` op.

The `reduction_dims` can be used to specify the subset of reduction dimensions
of the operation to tile. If left unspecified, all reduction dimensions are
tiled.

#### Example:

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction_dims</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `for_op` | TransformHandleTypeInterface instance |


### `transform.structured.tile_reduction_using_forall` (transform::TileReductionUsingForallOp)

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

#### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `forall` op.

#### Example:

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction_dims</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>num_threads</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>tile_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>mapping</code></td><td>::mlir::ArrayAttr</td><td>Device Mapping array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |


### `transform.structured.tile_using_for` (transform::TileUsingForOp)

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

#### Return modes

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>interchange</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>scalable_sizes</code></td><td>::mlir::DenseBoolArrayAttr</td><td>i1 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_sizes` | variadic of transform any param type or any handle type |

#### Results:

| Result | Description |
| :----: | ----------- |
| `tiled_linalg_op` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |


### `transform.structured.tile_using_forall` (transform::TileUsingForallOp)

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

#### Return modes

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

#### Example using `num_threads`

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%3:2 = transform.structured.tile_using_forall %0 num_threads [10, 20]
   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

#### Example using `tile_sizes`

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%sz = transform.structured.match ...
%3:2 = transform.structured.tile_using_forall %0 tile_sizes [0, %sz, 20]
   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
```

Traits: `AttrSizedOperandSegments`, `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_num_threads</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>static_tile_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>mapping</code></td><td>::mlir::ArrayAttr</td><td>Device Mapping array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `num_threads` | variadic of transform any param type or any handle type |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `packed_num_threads` | transform any param type or any handle type |
| `packed_tile_sizes` | transform any param type or any handle type |

#### Results:

| Result | Description |
| :----: | ----------- |
| `tiled_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |


### `transform.structured.transpose_conv2d` (transform::TransposeConv2DOp)

Syntax:

```
operation ::= `transform.structured.transpose_conv2d` $target attr-dict `:` functional-type($target, results)
```

Convert linalg.conv_2d_nhwc_fhwc into linalg.conv_2d_nhwc_hwcf by introducing
a linalg.transpose on the filter tensor/memref.

Whilst the fhwc filter channel ordering can be desirable for certain targets
and is a more direct mapping to higher level dialects such as TOSA (which only
supports this ordering) hwcf is better suited for transformations such as
img2col which can make use of optimized BLAS routines such as GEMM.

Returns one handle:
- The final operation of the sequence that replaces the original
  convolution.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.transpose_matmul` (transform::TransposeMatmulOp)

Syntax:

```
operation ::= `transform.structured.transpose_matmul` $target (`<` $inputToTranspose^ `>`)?
              attr-dict `:` functional-type($target, results)
```

Convert Linalg matmul ops to transposed variants.

By default the LHS matrix is transposed. Specify `<rhs>` to instead
transpose RHS matrix.

#### Return modes:

This operation fails if `target` is unsupported, i.e., not a
`linalg.matmul` or `linalg.batch_matmul`. Otherwise, the operation succeeds
and returns a handle to the transposed matmul op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>inputToTranspose</code></td><td>mlir::transform::TransposeMatmulInputAttr</td><td>Input to transpose when converting matmul ops to transposed variants</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.vectorize_children_and_apply_patterns` (transform::VectorizeChildrenAndApplyPatternsOp)

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

#### Return modes:

This operation produces a definite failure if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>vectorize_padding</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>vectorize_nd_extract</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>flatten_1d_depthwise_conv</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>disable_multi_reduction_to_contract_patterns</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>disable_transfer_permutation_map_lowering_patterns</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.structured.vectorize` (transform::VectorizeOp)

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
# Regular vectorization - vector sizes are inferred from the target Op
transform.structured.vectorize %target : !transform.any_op
```

The vector sizes can be either static or dynamic (SSA values). In case of
SSA values, the handle must be mapped to exactly one payload op with
exactly one index-typed result.

Note: The input vector sizes must be bigger than or equal to their
counterpart iteration space sizes.

Typically this operator should be applied to linalg operations that have
already been tiled to the appropriate sizes.

#### Return modes:

This operation produces a silenceable failure if at least one target op is
not a Linalg op or fails to vectorize. It produces a definite failure if
the dynamic vector sizes (SSA values) do not satisfy the constraints
mentioned above.

Traits: `ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>static_vector_sizes</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>vectorize_nd_extract</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>scalable_sizes</code></td><td>::mlir::DenseBoolArrayAttr</td><td>i1 dense array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |
| `vector_sizes` | variadic of transform any param type or any handle type |


### `transform.structured.winograd_conv2d` (transform::WinogradConv2DOp)

Syntax:

```
operation ::= `transform.structured.winograd_conv2d` $target attr-dict `:` functional-type($target, results)
```

Winograd Conv2D algorithm will convert linalg Conv2D operation into batched
matrix multiply. Before the matrix multiply, it will convert filter and
input into a format suitable for batched matrix multiply. After the matrix
multiply, it will convert output to the final result tensor.

The algorithm F(m x m, r x r) is

Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

The size of output Y is m x m. The size of filter g is r x r. The size of
input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
transformation matrices.

#### Return modes:

This operation produces a silenceable failure if `target` is unsupported.
Otherwise, the operation succeeds and returns a handle of the sequence that
replaces the original convolution.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>fmr</code></td><td>mlir::linalg::WinogradConv2DFmrAttr</td><td>allowed 32-bit signless integer cases: 0, 1, 2</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


## Tensor Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Tensor/TransformOps/TensorTransformOps.td)

### `transform.apply_patterns.tensor.bubble_up_extract_slice` (transform::ApplyBubbleUpExtractSlicePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.bubble_up_extract_slice` attr-dict
```

Indicates that producers of tensor.extract_slice should swap and operate on 
the result of the slice.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.decompose_concat` (transform::ApplyDecomposeTensorConcatPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.decompose_concat` attr-dict
```

Indicates that tensor.concat ops should be decomposed into a chain of
tensor.insert_slice operations inserting into a materialized destination.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion` (transform::ApplyDropRedundantInsertSliceRankExpansionPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion` attr-dict
```

Indicates that redundant tensor.insert_slice rank reductions should be
dropped. E.g., cases where a tensor.extract_slice rank reduction immediately
follows an inverse tensor.insert_slice rank expansion.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.fold_tensor_empty` (transform::ApplyFoldTensorEmptyPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_empty` attr-dict
```

Indicates that tensor.extract_slice and reassociative reshapes should be
folded into tensor.empty.

If `fold_single_use_only` is set to "true", only tensor.empty that have a
single use are folded.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>fold_single_use_only</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>


### `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers` (transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers` attr-dict
```

Indicates that tensor.extract_slice -> vector.transfer_read and
vector.transfer_write -> tensor.insert_slice op chains should be folded into
vector tranfer read and write ops

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.fold_tensor_subset_ops` (transform::ApplyFoldTensorSubsetOpsPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops` attr-dict
```

Indicates that tensor.empty should be folded with tensor.extract_slice,
tensor.expand_shape and tensor.collapse_shape.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice` (transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice` attr-dict
```

Indicates that consecutive tensor.extract_slice/tensor.insert_slice ops
should be merged into a single op. These patterns are not canonicalizations
because the bufferization is sensitive to IR structure.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.reassociative_reshape_folding` (transform::ApplyReassociativeReshapeFoldingPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.reassociative_reshape_folding` attr-dict
```

Indicates that reassociative reshapes (tensor.collapse_shape /
tensor.expand_shape) should be folded with inverse rank expansions / rank
reductions (via tensor.insert_slice / tensor.extract_slice).

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.tensor.rewrite_as_constant` (transform::ApplyRewriteTensorOpsAsConstantPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.rewrite_as_constant` (`aggressive` $aggressive^)? attr-dict
```

Indicates that tensor ops (such as tensor.generate) should be replaced with
constants (arith.constant) when possible.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>aggressive</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>


### `transform.tensor.make_loop_independent` (transform::MakeLoopIndependentOp)

Syntax:

```
operation ::= `transform.tensor.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed)
```

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`tensor.extract_slice` inside the loop.

Currently supported operations are:
- tensor.empty: Replaced with a new tensor.empty with upper bound sizes,
  followed by a tensor.extract_slice.
- tensor.pad: Replaced by an upper bound padding, followed by a
  tensor.extract_slice.

#### Return modes

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:
- tensor.empty: The returned handle points to the tensor.extract_slice op.
- tensor.pad: The returned handle points to the tensor.extract_slice op.

This transform op consumes the target handle and produces a result handle.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_loops</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `target` | TransformHandleTypeInterface instance |

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance |


### `transform.type_conversion.tensor.cast_shape_dynamic_dims` (transform::TypeConversionCastShapeDynamicDimsOp)

Syntax:

```
operation ::= `transform.type_conversion.tensor.cast_shape_dynamic_dims` (`ignore_dynamic_info` $ignore_dynamic_info^)? attr-dict
```

Populates a type converter with conversion materialization functions that
cast a tensor value between two cast-compatible tensors. See `tensor.cast`
for more information on cast compatibility between tensors.

If `ignore_dynamic_info` is not set, this will set an additional constraint
that source materializations do not cast dynamic dimensions to static ones.

Interfaces: `TypeConverterBuilderOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>ignore_dynamic_info</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>


## Vector Transform Operations

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/TransformOps/VectorTransformOps.td)

### `transform.apply_patterns.vector.cast_away_vector_leading_one_dim` (transform::ApplyCastAwayVectorLeadingOneDimPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.cast_away_vector_leading_one_dim` attr-dict
```

Collect a set of leading one dimension removal patterns.

These patterns insert vector.shape_cast to remove leading one dimensions
to expose more canonical forms of read/write/insert/extract operations.
With them, there are more chances that we can cancel out extract-insert
pairs or forward write-read pairs.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast` (transform::ApplyDropUnitDimWithShapeCastPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast` attr-dict
```

 Apply vector patterns to fold unit dims with vector.shape_cast Ops:
  - DropUnitDimFromElementwiseOps
  - DropUnitDimsFromScfForOp
  - DropUnitDimsFromTransposeOp

Excludes patterns for vector.transfer Ops. This is complemented by
shape_cast folding patterns.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.fold_arith_extension` (transform::ApplyFoldArithExtensionPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.fold_arith_extension` attr-dict
```

Collect a set of patterns that fold arithmetic extension on floating point
into vector contract for the backends with native support.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.elementwise_to_vector` (transform::ApplyFoldElementwiseToVectorPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.elementwise_to_vector` attr-dict
```

Collect a set of patterns that fold elementwise op on vectors to the vector
dialect.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.interleave_to_shuffle` (transform::ApplyInterleaveToShufflePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.interleave_to_shuffle` attr-dict
```

Indicates that 1D vector interleave operations should be rewritten as
vector shuffle operations.

This is motivated by some current codegen backends not handling vector
interleave operations.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_bitcast` (transform::ApplyLowerBitCastPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_bitcast` attr-dict
```

Indicates that vector bitcast operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_broadcast` (transform::ApplyLowerBroadcastPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_broadcast` attr-dict
```

Indicates that vector broadcast operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_contraction` (transform::ApplyLowerContractionPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_contraction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
```

Indicates that vector contraction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowering_strategy</code></td><td>::mlir::vector::VectorContractLoweringAttr</td><td>control the lowering of `vector.contract` operations.</td></tr>
</table>


### `transform.apply_patterns.vector.lower_create_mask` (transform::ApplyLowerCreateMaskPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_create_mask` attr-dict
```

Indicates that vector create_mask-like operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_gather` (transform::ApplyLowerGatherPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_gather` attr-dict
```

Indicates that vector.gather operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_interleave` (transform::ApplyLowerInterleavePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_interleave` attr-dict
```

Indicates that vector interleave operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_masked_transfers` (transform::ApplyLowerMaskedTransfersPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_masked_transfers` attr-dict
```

Apply opt-in patterns that lower vector.mask operations surrounding
side-effecting ops:
  - MaskedTransferReadOpPattern
  - MaskedTransferWriteOpPattern
  - MaskedGatherOpPattern

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_masks` (transform::ApplyLowerMasksPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_masks` attr-dict
```

Indicates that vector.create_mask and vector.constant_mask operations
should be lowered to finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_multi_reduction` (transform::ApplyLowerMultiReductionPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_multi_reduction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
```

Indicates that vector multi_reduction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowering_strategy</code></td><td>::mlir::vector::VectorMultiReductionLoweringAttr</td><td>control the lowering of `vector.multi_reduction`.</td></tr>
</table>


### `transform.apply_patterns.vector.lower_outerproduct` (transform::ApplyLowerOuterProductPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_outerproduct` attr-dict
```

Indicates that the vector outerproduct operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_scan` (transform::ApplyLowerScanPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_scan` attr-dict
```

Indicates that vector.scan operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_shape_cast` (transform::ApplyLowerShapeCastPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_shape_cast` attr-dict
```

Indicates that vector shape_cast operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.lower_transfer` (transform::ApplyLowerTransferPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_transfer` (`max_transfer_rank` `=` $max_transfer_rank^)? attr-dict
```

Indicates that vector transfer operations should be lowered to finer-grained
vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>max_transfer_rank</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>


### `transform.apply_patterns.vector.lower_transpose` (transform::ApplyLowerTransposePatternsOp)

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

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>lowering_strategy</code></td><td>::mlir::vector::VectorTransposeLoweringAttr</td><td>control the lowering of `vector.transpose` operations.</td></tr>
<tr><td><code>avx2_lowering_strategy</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>


### `transform.apply_patterns.vector.materialize_masks` (transform::ApplyMaterializeMasksPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.materialize_masks` attr-dict
```

Indicates that mask operations should be lowered to fine-grained arithemtic
operations.

This is usually the last step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.rank_reducing_subview_patterns` (transform::ApplyRankReducingSubviewPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.rank_reducing_subview_patterns` attr-dict
```

Apply opt-in vector transfer permutation patterns that include:
  - TransferReadDropUnitDimsPattern
  - TransferWriteDropUnitDimsPattern

These patterns have the effect of rewriting a vector.transfer with unit
dimensions into a rank-reduced version thanks to subview operations.
This is complemented by shape_cast folding patterns.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.rewrite_narrow_types` (transform::ApplyRewriteNarrowTypePatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.rewrite_narrow_types` attr-dict
```

Indicates that vector narrow rewrite operations should be applied.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Warning: these patterns currently only work for little endian targets.

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.sink_mem_ops` (transform::ApplySinkVectorMemPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.sink_mem_ops` attr-dict
```

Patterns that replace redundant Vector Ops (followed by
`vector.load`/`vector.store`) with either vector.load/vector.store or
`memref.load`/`memref.store`. Currently limited to 1-element vectors.

Example:
```
vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
vector.extract %0[1] : f32 from vector<4xf32>
```
Gets converted to:
```
%c1 = arith.constant 1 : index
%0 = arith.addi %arg1, %c1 overflow<nsw> : index
%1 = memref.load %arg0[%0] : memref<?xf32>
```

Interfaces: `PatternDescriptorOpInterface`


### `transform.apply_patterns.vector.sink_ops` (transform::ApplySinkVectorPatternsOp)

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


### `transform.apply_patterns.vector.split_transfer_full_partial` (transform::ApplySplitTransferFullPartialPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.split_transfer_full_partial` (`split_transfer_strategy` `=` $split_transfer_strategy^)? attr-dict
```

Indicates that vector transfer operations should be split to full and
partial parts.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>split_transfer_strategy</code></td><td>::mlir::vector::VectorTransferSplitAttr</td><td>control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants.</td></tr>
</table>


### `transform.apply_patterns.vector.transfer_permutation_patterns` (transform::ApplyTransferPermutationPatternsOp)

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


### `transform.apply_patterns.vector.transfer_to_scf` (transform::ApplyTransferToScfPatternsOp)

Syntax:

```
operation ::= `transform.apply_patterns.vector.transfer_to_scf` oilist (
              `max_transfer_rank` `=` $max_transfer_rank
              | `full_unroll` `=` $full_unroll
              )
              attr-dict
```

Indicates that vector transfer operations should be rewritten with scf.for
loops over finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>max_transfer_rank</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>full_unroll</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>


### `transform.apply_patterns.vector.reduction_to_contract` (transform::ApplyVectorReductionToContractPatternsOp)

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


### `transform.apply_conversion_patterns.vector.vector_to_llvm` (transform::ApplyVectorToLLVMConversionPatternsOp)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.vector.vector_to_llvm` attr-dict
```

Collects patterns that convert vector dialect ops to LLVM dialect ops. These
patterns require an "LLVMTypeConverter".

The patterns can be customized as follows:
- `reassociate_fp_reductions`: Allows LLVM to reassociate floating-point
  reductions for speed.
- `force_32bit_vector_indices`: Allows the compiler to assume that vector
  indices fit in 32-bit if that yields faster code.

Interfaces: `ConversionPatternDescriptorOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reassociate_fp_reductions</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>force_32bit_vector_indices</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>use_vector_alignment</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>


<!-- Autogenerated by mlir-tblgen; don't manually edit -->


## TransformHandleTypeInterface (`TransformHandleTypeInterface`)

Types that can be used for the Transform dialect operation handle values.
Such types define the properties of Payload IR operations associated with
the handle. A user of such a handle can assume that these properties have
been verified for any Payload IR operation associated with it.

### Methods:

#### `checkPayload`

```c++
::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload);
```

Checks if the given associated objects (Payload IR operations or attributes)
satisfy the conditions defined by this type. If not, produces a silenceable
error at the specified location.

NOTE: This method *must* be implemented by the user.

## TransformParamTypeInterface (`TransformParamTypeInterface`)

Types that can be used for the Transform dialect parameter values. Such types
define the structure of the parameters associated with the value, e.g., their
underlying type. A user of the value can assume that the parameter has been
verified.

### Methods:

#### `checkPayload`

```c++
::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Attribute> payload);
```

Checks if the given associated objects (Payload IR operations or attributes)
satisfy the conditions defined by this type. If not, produces a silenceable
error at the specified location.

NOTE: This method *must* be implemented by the user.

## TransformValueHandleTypeInterface (`TransformValueHandleTypeInterface`)

Types that can be used for the Transform dialect handle values pointing to
Payload IR values. Such types define the properties of Payload IR values
associated with the handle. Users of such a handle can assume that these
properties have been verified for any Payload IR value associated with it.

### Methods:

#### `checkPayload`

```c++
::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Value> payload);
```

Checks if the given associated objects (Payload IR operations or attributes)
satisfy the conditions defined by this type. If not, produces a silenceable
error at the specified location.

NOTE: This method *must* be implemented by the user.

<!-- Autogenerated by mlir-tblgen; don't manually edit -->


## ConversionPatternDescriptorOpInterface (`ConversionPatternDescriptorOpInterface`)

This interface should be implemented by ops that select conversion patterns
of a `transform.apply_patterns` op. It provides a method to populate a
rewrite pattern set with conversion patterns.

Note: Non-conversion rewrite patterns should not be populated with
`ConversionPatternDescriptorOpInterface` because it is not generally safe
to use non-conversion rewrite patterns as part of a dialect conversion.

### Methods:

#### `populatePatterns`

```c++
void populatePatterns(::mlir::TypeConverter &typeConverter, ::mlir::RewritePatternSet &patterns);
```

Populate conversion patterns into the given pattern set with the
given type converter.

NOTE: This method *must* be implemented by the user.

#### `populateConversionTargetRules`

```c++
void populateConversionTargetRules(const ::mlir::TypeConverter &typeConverter, ::mlir::ConversionTarget &conversionTarget);
```

Populate the ConversionTarget using the final TypeConverter. The default
implementation is to do nothing. Overriding this method can be useful
in order to setup the ConversionTarget for structural type conversions.
In such a situation, an op's legality depends on using the TypeConverter
to determine whether the op's operand and result types are legal
(defined as converting to themselves).


NOTE: This method *must* be implemented by the user.

#### `getTypeConverter`

```c++
std::unique_ptr<::mlir::TypeConverter> getTypeConverter();
```

Return the type converter to be used with this pattern set. If no
type converter is specified, the default type converter of the enclosing
"apply_conversion_patterns" op is used.

NOTE: This method *must* be implemented by the user.

#### `verifyTypeConverter`

```c++
::llvm::LogicalResult verifyTypeConverter(TypeConverterBuilderOpInterface builder);
```

Verify the default type converter that is provided by the enclosing
"apply_conversion_patterns" op.

NOTE: This method *must* be implemented by the user.

## FindPayloadReplacementOpInterface (`FindPayloadReplacementOpInterface`)

This interface is queried by the `TrackingListener` and can be implemented
by payload ops to indicate that the lookup should be continue with its
operands when looking for payload op replacements.

Example: Consider the case where a tracked "test.foo" payload op is replaced
with a new "test.foo" op, but wrapped in a "tensor.reshape" op. In that
case, the mapping of the original "test.foo" op should be updated with the
new "test.foo" op. A "tensor.reshape" is a metadata-only op that should be
skipped when inspecting the replacement values of the original "test.foo"
op. More details can be found at `TrackingListener` documentation.

Note: Ops that implement `CastOpInterface` do not need to implement this
interface. Such ops are skipped by default. This interface should be
implemented by cast-like/metadata-only ops that cannot implement
`CastOpInterface`.

### Methods:

#### `getNextOperands`

```c++
::llvm::SmallVector<::mlir::Value> getNextOperands();
```

Return the operands at which the lookup for replacement payload ops
should continue.

NOTE: This method *must* be implemented by the user.

## PatternDescriptorOpInterface (`PatternDescriptorOpInterface`)

This interface should be implemented by ops that select rewrite patterns of
a `transform.apply_patterns` op. It provides a method to populate a rewrite
pattern set with patterns.

Note: Conversion patterns are rewrite patterns in MLIR, but they should not
be populated with `PatternDescriptorOpInterface` because they cannot be
used in a greedy pattern rewrite.

### Methods:

#### `populatePatterns`

```c++
void populatePatterns(::mlir::RewritePatternSet &patterns);
```

Populate rewrite patterns into the given pattern set.

NOTE: This method *must* be implemented by the user.

#### `populatePatternsWithState`

```c++
void populatePatternsWithState(::mlir::RewritePatternSet &patterns, ::mlir::transform::TransformState &state);
```

Populate rewrite patterns into the given pattern set taking into account
the transform state.

NOTE: This method *must* be implemented by the user.

## TransformOpInterface (`TransformOpInterface`)

This interface is to be implemented by operations that identify
transformations to be performed on other operations. The former are referred
to as transform IR operations. The latter are referred to as payload IR
operations. Such transform IR operations provide a fine-grain control
mechanism over how transformations are applied by using and defining
transform IR values, referred to as handles, that correspond to sets of
operations in the payload IR. Transformations are applied starting from the
operations identified by handles, but may affect other operations as well.
Further restrictions may be imposed by flows that rely on transform IR
operations to control transformations.

### Methods:

#### `apply`

```c++
::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter &rewriter, ::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
```

Applies the transformation represented by the current operation. This
accepts as arguments the object that must be populated with results of
the current transformation and a transformation state object that can be
used for queries, e.g., to obtain the list of operations on which the
transformation represented by the current op is targeted. Returns a
special status object indicating whether the transformation succeeded
or failed, and, if it failed, whether the failure is recoverable or not.

IR must be created, modified and deleted with the provided rewriter.
implementations are responsible for setting the insertion point of the
rewriter to the desired location.

NOTE: This method *must* be implemented by the user.

#### `allowsRepeatedHandleOperands`

```c++
bool allowsRepeatedHandleOperands();
```

Indicates whether the op instance allows its handle operands to be
associated with the same payload operations.

NOTE: This method *must* be implemented by the user.

## TypeConverterBuilderOpInterface (`TypeConverterBuilderOpInterface`)

This interface should be implemented by ops that specify a type converter
for a dialect conversion, or to populate a type converter with
conversions.

When such ops are intended to be used with "apply_conversion_patterns" or
other operations that expect a type converter, a non-default implementation
of `getTypeConverter` should be implemented. For use with "cast_and_call"
like ops that construct a type converter iteratively, non-default
`populateTypeMaterializations` should be implemented.

### Methods:

#### `getTypeConverter`

```c++
std::unique_ptr<::mlir::TypeConverter> getTypeConverter();
```

Return the type converter to be used with a dialect conversion.

NOTE: This method *must* be implemented by the user.

#### `getTypeConverterType`

```c++
static StringRef getTypeConverterType();
```

Return the type of type converter that this `getTypeConverter` returns.
This function is used for op verification.

NOTE: This method *must* be implemented by the user.

#### `populateTypeMaterializations`

```c++
void populateTypeMaterializations(::mlir::TypeConverter &converter);
```

Populate the given type converter with source/target materialization
functions.

NOTE: This method *must* be implemented by the user.
