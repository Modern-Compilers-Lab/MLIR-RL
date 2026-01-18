### `transform.alternatives` (transform::AlternativesOp) [¶](#transformalternatives-transformalternativesop)

`transform.alternatives`
[¶](#transformalternatives-transformalternativesop)

*Attempts sequences of transforms until one succeeds*

*Attempts sequences of transforms until one succeeds*

Syntax:

```
operation ::= `transform.alternatives` ($scope^ `:` type($scope))? (`->` type($results)^)? attr-dict-with-keyword regions
```

`` operation ::= `transform.alternatives` ($scope^ `:` type($scope))? (`->` type($results)^)? attr-dict-with-keyword regions ``

This op may have an arbitrary number of regions, each of which represents a
sequence of transform operations to be applied to the same payload IR. The
regions are visited in order of appearance, and transforms in them are
applied in their respective order of appearance. If one of these transforms
fails to apply, the remaining ops in the same region are skipped an the next
region is attempted. If all transformations in a region succeed, the
remaining regions are skipped and the entire “alternatives” transformation
succeeds. If all regions contained a failing transformation, the entire
“alternatives” transformation fails.

It is up to the nested operations to define which errors are “recoverable”
(or “silenceable”) and allow another alternatives to be attempted, and which
errors should be propagated without attempting the other alternatives.

The single operand of this operation is the scope in which the alternative
transformation sequences are attempted, that is, an operation in the payload
IR that contains all the other operations that may be modified by the
transformations. The scope operation must be isolated from above. There is
no check that the transforms are indeed scoped as their “apply” methods can
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
“alternatives” op. Therefore, each alternative region must yield the same
number of results, which should also match the number and the types of the
“alternatives” op results.

Remark: this op allows one to implement a simple “try” construct as follows:

```
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

```
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

`%result = transform.alternatives %scope {
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
}`
%result = transform.alternatives %scope {
%result = transform.alternatives %scope {
%result
=
.
%scope
{
^bb0(%arg0: !transform.any\_op):
^bb0(%arg0: !transform.any\_op):
^bb0
(
%arg0
:
!
.
):
 // Try a fallible transformation.
 // Try a fallible transformation.
// Try a fallible transformation.
 %0 = transform.fallible %arg0 // ...
 %0 = transform.fallible %arg0 // ...

%0
=
.
%arg0
// ...
 // If succeeded, yield the the result of the transformation.
 // If succeeded, yield the the result of the transformation.

// If succeeded, yield the the result of the transformation.
 transform.yield %0 : !transform.any\_op
 transform.yield %0 : !transform.any\_op

.
%0
:
!
.
}, {
}, {
},
{
^bb0(%arg0: !transform.any\_op):
^bb0(%arg0: !transform.any\_op):
^bb0
(
%arg0
:
!
.
):
 // Otherwise, the second alternative is tried and it always succeeds by
 // Otherwise, the second alternative is tried and it always succeeds by
// Otherwise, the second alternative is tried and it always succeeds by
 // returning the original handle.
 // returning the original handle.

// returning the original handle.
 transform.yield %arg0 : !transform.any\_op
 transform.yield %arg0 : !transform.any\_op

.
%arg0
:
!
.
}
}
}

Traits: `IsolatedFromAbove`, `PossibleTopLevelTransformOpTrait`, `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

`IsolatedFromAbove`
`PossibleTopLevelTransformOpTrait`
`SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionBranchOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands)

[¶](#operands)

| Operand | Description |
| --- | --- |
| `scope` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `scope` | TransformHandleTypeInterface instance |
| `scope` | TransformHandleTypeInterface instance |
 `scope` |`scope` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results)

[¶](#results)

| Result | Description |
| --- | --- |
| `results` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of TransformHandleTypeInterface instance |
| `results` | variadic of TransformHandleTypeInterface instance |
 `results` |`results` variadic of TransformHandleTypeInterface instance |

---

### `transform.annotate` (transform::AnnotateOp) [¶](#transformannotate-transformannotateop)

`transform.annotate`
[¶](#transformannotate-transformannotateop)

*Annotates the target operation with an attribute by name*

*Annotates the target operation with an attribute by name*

Syntax:

```
operation ::= `transform.annotate` $target $name attr-dict (`=` $param^)?`:` type($target) (`,` type($param)^)?
```

`` operation ::= `transform.annotate` $target $name attr-dict (`=` $param^)?`:` type($target) (`,` type($param)^)? ``

Adds an attribute with the given `name` to the `target` operation. An
optional `param` handle can be provided to give the attribute a specific
value, else a UnitAttr is added. A single attribute will be broadcasted to
all target operations, otherwise the attributes will be mapped 1:1 based on
the order within the handles.

`name`
`target`
`param`

Produces a silenceable failure if the length of the parameter payload does
not match the length of the target payload. Does not consume the provided
handles.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes)

[¶](#attributes)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `name` | ::mlir::StringAttr | string attribute |
 `name` |`name` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-1)

[¶](#operands-1)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `param` | TransformParamTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `param` | TransformParamTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `param` | TransformParamTypeInterface instance |
 `param` |`param` TransformParamTypeInterface instance |

---

### `transform.apply_patterns.canonicalization` (transform::ApplyCanonicalizationPatternsOp) [¶](#transformapply_patternscanonicalization-transformapplycanonicalizationpatternsop)

`transform.apply_patterns.canonicalization`
[¶](#transformapply_patternscanonicalization-transformapplycanonicalizationpatternsop)

*Populates canonicalization patterns*

*Populates canonicalization patterns*

Syntax:

```
operation ::= `transform.apply_patterns.canonicalization` attr-dict
```

`` operation ::= `transform.apply_patterns.canonicalization` attr-dict ``

This op populates all canonicalization patterns of all loaded dialects in
an `apply_patterns` transform.

`apply_patterns`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_cse` (transform::ApplyCommonSubexpressionEliminationOp) [¶](#transformapply_cse-transformapplycommonsubexpressioneliminationop)

`transform.apply_cse`
[¶](#transformapply_cse-transformapplycommonsubexpressioneliminationop)

*Eliminate common subexpressions in the body of the target op*

*Eliminate common subexpressions in the body of the target op*

Syntax:

```
operation ::= `transform.apply_cse` `to` $target attr-dict `:` type($target)
```

`` operation ::= `transform.apply_cse` `to` $target attr-dict `:` type($target) ``

This transform applies common subexpression elimination (CSE) to the body
of the targeted op.

This transform reads the target handle and modifies the payload. Existing
handles to operations inside of the targeted op are retained and updated if
necessary. Note that this can lead to situations where a handle, that was
previously mapped to multiple distinct (but equivalent) operations, is now
mapped to the same operation multiple times.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-2)

[¶](#operands-2)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_conversion_patterns` (transform::ApplyConversionPatternsOp) [¶](#transformapply_conversion_patterns-transformapplyconversionpatternsop)

`transform.apply_conversion_patterns`
[¶](#transformapply_conversion_patterns-transformapplyconversionpatternsop)

*Applies conversion patterns to the body of the targeted op*

*Applies conversion patterns to the body of the targeted op*

Syntax:

```
operation ::= `transform.apply_conversion_patterns` `to` $target $patterns
              (`with` `type_converter` $default_type_converter_region^)?
              attr-dict `:` type($target)
```

`` operation ::= `transform.apply_conversion_patterns` `to` $target $patterns
(`with` `type_converter` $default_type_converter_region^)?
attr-dict `:` type($target) ``

This transform applies the specified conversion patterns to the targeted op
and all nested ops. By default, this transform applies a “full” dialect
conversion. If the `partial_conversion` unit attribute is present, this
transform applies a partial dialect conversion.

`partial_conversion`

The patterns that should be applied are specified in the first graph region
of this op. They must implement the
`ConversionPatternDescriptorOpInterface`. The order in which patterns are
applied is unspecified; i.e., the ordering of ops in the region of this op
is irrelevant.

`ConversionPatternDescriptorOpInterface`

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

`TypeConverterBuilderOpInterface`
`getTypeConverter`

The `legal_ops`, `illegal_ops`, `legal_dialects`, `illegal_dialects`
attributes specify the conversion target.

`legal_ops`
`illegal_ops`
`legal_dialects`
`illegal_dialects`

This transform modifies the payload. By default, it consumes the `target`
handle. It does not produce any handles.

`target`

If the `preserve_handles` attribute is set, this transform does not consume
the `target` handle and instead updates handles based on notifications from
a tracking listener that is attached to the dialect conversion, similar to
`transform.apply_patterns`. Only replacements via `RewriterBase::replaceOp`
or `replaceOpWithNewOp` are considered “payload op replacements”. In
contrast to `transform.apply_patterns`, we allow replacement ops even if the
op name has changed. This is because conversion patterns are expected to
lower ops to different ops (from a different dialect). More details can be
found at the documentation site of `TrackingListener`.

`preserve_handles`
`target`
`transform.apply_patterns`
`RewriterBase::replaceOp`
`replaceOpWithNewOp`
`transform.apply_patterns`
`TrackingListener`

This transform produces a silenceable failure if the dialect conversion was
unsuccessful or the tracking listener failed to find a replacement op.

Traits: `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

`HasOnlyGraphRegion`
`NoTerminator`
`ReportTrackingListenerFailuresOpTrait`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionKindInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-1)

[¶](#attributes-1)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `legal_ops` | ::mlir::ArrayAttr | string array attribute |
| `illegal_ops` | ::mlir::ArrayAttr | string array attribute |
| `legal_dialects` | ::mlir::ArrayAttr | string array attribute |
| `illegal_dialects` | ::mlir::ArrayAttr | string array attribute |
| `partial_conversion` | ::mlir::UnitAttr | unit attribute |
| `preserve_handles` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `legal_ops` | ::mlir::ArrayAttr | string array attribute |
 `legal_ops` |`legal_ops` ::mlir::ArrayAttr | string array attribute || `illegal_ops` | ::mlir::ArrayAttr | string array attribute |
 `illegal_ops` |`illegal_ops` ::mlir::ArrayAttr | string array attribute || `legal_dialects` | ::mlir::ArrayAttr | string array attribute |
 `legal_dialects` |`legal_dialects` ::mlir::ArrayAttr | string array attribute || `illegal_dialects` | ::mlir::ArrayAttr | string array attribute |
 `illegal_dialects` |`illegal_dialects` ::mlir::ArrayAttr | string array attribute || `partial_conversion` | ::mlir::UnitAttr | unit attribute |
 `partial_conversion` |`partial_conversion` ::mlir::UnitAttr | unit attribute || `preserve_handles` | ::mlir::UnitAttr | unit attribute |
 `preserve_handles` |`preserve_handles` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-3)

[¶](#operands-3)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_dce` (transform::ApplyDeadCodeEliminationOp) [¶](#transformapply_dce-transformapplydeadcodeeliminationop)

`transform.apply_dce`
[¶](#transformapply_dce-transformapplydeadcodeeliminationop)

*Eliminate dead operations in the body of the target op*

*Eliminate dead operations in the body of the target op*

Syntax:

```
operation ::= `transform.apply_dce` `to` $target attr-dict `:` type($target)
```

`` operation ::= `transform.apply_dce` `to` $target attr-dict `:` type($target) ``

This transform applies dead code elimination (DCE) to the body of the
targeted op.

Note: “transform.apply\_patterns” with an empty region can also be used to
remove dead ops. However, that op applies additional simplifications such as
op folding and region simplification.

This transform reads the target handle and modifies the payload. Note that
this transform may silently remove payload ops from handles.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-4)

[¶](#operands-4)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_licm` (transform::ApplyLoopInvariantCodeMotionOp) [¶](#transformapply_licm-transformapplyloopinvariantcodemotionop)

`transform.apply_licm`
[¶](#transformapply_licm-transformapplyloopinvariantcodemotionop)

*Move loop-invariant code out of a loop-like op*

*Move loop-invariant code out of a loop-like op*

Syntax:

```
operation ::= `transform.apply_licm` `to` $target attr-dict `:` type($target)
```

`` operation ::= `transform.apply_licm` `to` $target attr-dict `:` type($target) ``

This transform moves side-effect free, loop invariant code out of the
targeted loop-like op. The targeted op must implement the
`LoopLikeOpInterface`.

`LoopLikeOpInterface`

Note: To move invariant ops from a loop nest, this transform must be applied
to each loop of the loop nest, starting with the inner-most loop.

This transform reads the target handle and modifies the payload.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-5)

[¶](#operands-5)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns` (transform::ApplyPatternsOp) [¶](#transformapply_patterns-transformapplypatternsop)

`transform.apply_patterns`
[¶](#transformapply_patterns-transformapplypatternsop)

*Greedily applies patterns to the body of the targeted op*

*Greedily applies patterns to the body of the targeted op*

Syntax:

```
operation ::= `transform.apply_patterns` `to` $target $patterns attr-dict `:` type($target)
```

`` operation ::= `transform.apply_patterns` `to` $target $patterns attr-dict `:` type($target) ``

This transform greedily applies the specified patterns to the body of the
targeted op until a fixpoint was reached. Patterns are not applied to the
targeted op itself.

The patterns that should be applied are specified in the graph region of
this op. They must implement the `PatternDescriptorOpInterface`. The order
in which patterns are applied is unspecified; i.e., the ordering of ops in
the region of this op is irrelevant.

`PatternDescriptorOpInterface`

If `apple_cse` is set, the greedy pattern rewrite is interleaved with
common subexpression elimination (CSE): both are repeated until a fixpoint
is reached.

`apple_cse`

This transform only reads the target handle and modifies the payload. If a
pattern erases or replaces a tracked op, the mapping is updated accordingly.

Only replacements via `RewriterBase::replaceOp` or `replaceOpWithNewOp` are
considered “payload op replacements”. Furthermore, only if the replacement
values are defined by the same op and that op has the same type as the
original op, the mapping is updated. Otherwise, this transform produces a
silenceable failure. More details can be found at the documentation site of
`TrackingListener`.

`RewriterBase::replaceOp`
`replaceOpWithNewOp`
`TrackingListener`

This transform also produces a silenceable failure if the pattern
application did not converge within the default number of
iterations/rewrites of the greedy pattern rewrite driver.

Traits: `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`, `TransformEachOpTrait`

`HasOnlyGraphRegion`
`NoTerminator`
`ReportTrackingListenerFailuresOpTrait`
`SingleBlock`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionKindInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-2)

[¶](#attributes-2)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `apply_cse` | ::mlir::UnitAttr | unit attribute |
| `max_iterations` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `max_num_rewrites` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `apply_cse` | ::mlir::UnitAttr | unit attribute |
 `apply_cse` |`apply_cse` ::mlir::UnitAttr | unit attribute || `max_iterations` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `max_iterations` |`max_iterations` ::mlir::IntegerAttr | 64-bit signless integer attribute || `max_num_rewrites` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `max_num_rewrites` |`max_num_rewrites` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-6)

[¶](#operands-6)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_registered_pass` (transform::ApplyRegisteredPassOp) [¶](#transformapply_registered_pass-transformapplyregisteredpassop)

`transform.apply_registered_pass`
[¶](#transformapply_registered_pass-transformapplyregisteredpassop)

*Applies the specified registered pass or pass pipeline*

*Applies the specified registered pass or pass pipeline*

Syntax:

```
operation ::= `transform.apply_registered_pass` $pass_name (`with` `options` `=`
              custom<ApplyRegisteredPassOptions>($options, $dynamic_options)^)?
              `to` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.apply_registered_pass` $pass_name (`with` `options` `=`
custom<ApplyRegisteredPassOptions>($options, $dynamic_options)^)?
`to` $target attr-dict `:` functional-type(operands, results) ``

This transform applies the specified pass or pass pipeline to the targeted
ops. The name of the pass/pipeline is specified as a string attribute, as
set during pass/pipeline registration.

Optionally, pass options may be specified via a DictionaryAttr. This
dictionary is converted to a string – formatted `key=value ...` – which
is expected to be in the exact format used by the pass on the commandline.
Values are either attributes or (SSA-values of) Transform Dialect params.
For example:

`key=value ...`

```
transform.apply_registered_pass "canonicalize"
    with options = { "top-down" = false,
                     "max-iterations" = %max_iter,
                     "test-convergence" = true,
                     "max-num-rewrites" = %max_rewrites }
    to %module
: (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
```

```
transform.apply_registered_pass "canonicalize"
    with options = { "top-down" = false,
                     "max-iterations" = %max_iter,
                     "test-convergence" = true,
                     "max-num-rewrites" = %max_rewrites }
    to %module
: (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op
```

`transform.apply_registered_pass "canonicalize"
 with options = { "top-down" = false,
 "max-iterations" = %max_iter,
 "test-convergence" = true,
 "max-num-rewrites" = %max_rewrites }
 to %module
: (!transform.any_param, !transform.any_param, !transform.any_op) -> !transform.any_op`
transform.apply\_registered\_pass "canonicalize"
transform.apply\_registered\_pass "canonicalize"
.
"canonicalize"
 with options = { "top-down" = false,
 with options = { "top-down" = false,
options =
{
"top-down"
=
,
 "max-iterations" = %max\_iter,
 "max-iterations" = %max\_iter,
"max-iterations"
=
%max\_iter
,
 "test-convergence" = true,
 "test-convergence" = true,
"test-convergence"
=
,
 "max-num-rewrites" = %max\_rewrites }
 "max-num-rewrites" = %max\_rewrites }
"max-num-rewrites"
=
%max\_rewrites
}
 to %module
 to %module
%module
: (!transform.any\_param, !transform.any\_param, !transform.any\_op) -> !transform.any\_op
: (!transform.any\_param, !transform.any\_param, !transform.any\_op) -> !transform.any\_op
:
(!
.
,
!
.
,
!
.
)
->
!
.

Options’ values which are `ArrayAttr`s are converted to comma-separated
lists of options. Likewise for params which associate multiple values.

`ArrayAttr`

This op first looks for a pass pipeline with the specified name. If no such
pipeline exists, it looks for a pass with the specified name. If no such
pass exists either, this op fails definitely.

This transform consumes the target handle and produces a new handle that is
mapped to the same op. Passes are not allowed to remove/modify the operation
that they operate on, so the target op is guaranteed to still exist. The
target handle is invalidated because a pass may arbitrarily modify the body
of targeted ops.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-3)

[¶](#attributes-3)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `pass_name` | ::mlir::StringAttr | string attribute |
| `options` | ::mlir::DictionaryAttr | dictionary of named attribute values |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `pass_name` | ::mlir::StringAttr | string attribute |
 `pass_name` |`pass_name` ::mlir::StringAttr | string attribute || `options` | ::mlir::DictionaryAttr | dictionary of named attribute values |
 `options` |`options` ::mlir::DictionaryAttr | dictionary of named attribute values |

---

#### Operands: [¶](#operands-7)

[¶](#operands-7)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_options` | variadic of TransformParamTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `dynamic_options` | variadic of TransformParamTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `dynamic_options` | variadic of TransformParamTypeInterface instance |
 `dynamic_options` |`dynamic_options` variadic of TransformParamTypeInterface instance |

---

#### Results: [¶](#results-1)

[¶](#results-1)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.apply_conversion_patterns.dialect_to_llvm` (transform::ApplyToLLVMConversionPatternsOp) [¶](#transformapply_conversion_patternsdialect_to_llvm-transformapplytollvmconversionpatternsop)

`transform.apply_conversion_patterns.dialect_to_llvm`
[¶](#transformapply_conversion_patternsdialect_to_llvm-transformapplytollvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.dialect_to_llvm` $dialect_name attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.dialect_to_llvm` $dialect_name attr-dict ``

Collects patterns that convert ops from the specified dialect to LLVM
dialect ops. These patterns require an “LLVMTypeConverter”.

Note: Only dialects that implement the `ConvertToLLVMPatternInterface` are
supported. Any conversion target modifications by interface implementations
are currently ignored. The conversion target is fully specified by the
enclosing “apply\_conversion\_patterns” op.

`ConvertToLLVMPatternInterface`

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-4)

[¶](#attributes-4)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `dialect_name` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `dialect_name` | ::mlir::StringAttr | string attribute |
 `dialect_name` |`dialect_name` ::mlir::StringAttr | string attribute |

---

### `transform.cast` (transform::CastOp) [¶](#transformcast-transformcastop)

`transform.cast`
[¶](#transformcast-transformcastop)

Syntax:

```
operation ::= `transform.cast` $input attr-dict `:` type($input) `to` type($output)
```

`` operation ::= `transform.cast` $input attr-dict `:` type($input) `to` type($output) ``

Traits: `TransformEachOpTrait`

`TransformEachOpTrait`

Interfaces: `CastOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`CastOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-8)

[¶](#operands-8)

| Operand | Description |
| --- | --- |
| `input` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `input` | TransformHandleTypeInterface instance |
| `input` | TransformHandleTypeInterface instance |
 `input` |`input` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-2)

[¶](#results-2)

| Result | Description |
| --- | --- |
| `output` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `output` | TransformHandleTypeInterface instance |
| `output` | TransformHandleTypeInterface instance |
 `output` |`output` TransformHandleTypeInterface instance |

---

### `transform.collect_matching` (transform::CollectMatchingOp) [¶](#transformcollect_matching-transformcollectmatchingop)

`transform.collect_matching`
[¶](#transformcollect_matching-transformcollectmatchingop)

*Collects all payload ops that match the given named matcher*

*Collects all payload ops that match the given named matcher*

Syntax:

```
operation ::= `transform.collect_matching` $matcher `in` $root attr-dict `:` functional-type($root, $results)
```

`` operation ::= `transform.collect_matching` $matcher `in` $root attr-dict `:` functional-type($root, $results) ``

Collects operations or other payload IR objects nested under `root`
(inclusive) that match the given matcher expressed as a named sequence. The
matcher sequence must accept exactly one argument that it is not allowed to
modify. It must yield as many values as this op has results. Each of the
yielded values must be associated with exactly one payload object. If any
operation in the matcher sequence produces a silenceable failure, the
matcher advances to the next payload operation in the walk order without
finishing the sequence.

`root`

The i-th result of this operation is constructed by concatenating the i-th
yielded payload IR objects of all successful matcher sequence applications.
All results are guaranteed to be mapped to the same number of payload IR
objects.

The operation succeeds unless the matcher sequence produced a definite
failure for any invocation.

Interfaces: `MemoryEffectOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`SymbolUserOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-5)

[¶](#attributes-5)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `matcher` | ::mlir::SymbolRefAttr | symbol reference attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `matcher` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `matcher` |`matcher` ::mlir::SymbolRefAttr | symbol reference attribute |

---

#### Operands: [¶](#operands-9)

[¶](#operands-9)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-3)

[¶](#results-3)

| Result | Description |
| --- | --- |
| `results` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of any transform handle or parameter |
| `results` | variadic of any transform handle or parameter |
 `results` |`results` variadic of any transform handle or parameter |

---

### `transform.foreach_match` (transform::ForeachMatchOp) [¶](#transformforeach_match-transformforeachmatchop)

`transform.foreach_match`
[¶](#transformforeach_match-transformforeachmatchop)

*Applies named sequences when a named matcher succeeds*

*Applies named sequences when a named matcher succeeds*

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

`` operation ::= `transform.foreach_match` oilist( `restrict_root` $restrict_root
| `flatten_results` $flatten_results
)
`in`
$root (`,` $forwarded_inputs^)?
custom<ForeachMatchSymbols>($matchers, $actions)
attr-dict
`:` functional-type(operands, results) ``

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

`transform.named_sequence`
`Operation::walk`

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

`root`

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

`foreach_match`
`flatten_results`
`root`

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

`restrict_root`

The operation succeeds if none of the matchers produced a definite failure
during application and if all of the applied actions produced success. Note
that it also succeeds if all the matchers failed on all payload operations,
i.e. failure to apply is not an error. The operation produces a silenceable
failure if any applied action produced a silenceable failure. In this case,
the resulting handle is associated with an empty payload. The operation
produces a definite failure if any of the applied matchers or actions
produced a definite failure.

Interfaces: `MemoryEffectOpInterface`, `OpAsmOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`OpAsmOpInterface`
`SymbolUserOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-6)

[¶](#attributes-6)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `restrict_root` | ::mlir::UnitAttr | unit attribute |
| `flatten_results` | ::mlir::UnitAttr | unit attribute |
| `matchers` | ::mlir::ArrayAttr | symbol ref array attribute |
| `actions` | ::mlir::ArrayAttr | symbol ref array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `restrict_root` | ::mlir::UnitAttr | unit attribute |
 `restrict_root` |`restrict_root` ::mlir::UnitAttr | unit attribute || `flatten_results` | ::mlir::UnitAttr | unit attribute |
 `flatten_results` |`flatten_results` ::mlir::UnitAttr | unit attribute || `matchers` | ::mlir::ArrayAttr | symbol ref array attribute |
 `matchers` |`matchers` ::mlir::ArrayAttr | symbol ref array attribute || `actions` | ::mlir::ArrayAttr | symbol ref array attribute |
 `actions` |`actions` ::mlir::ArrayAttr | symbol ref array attribute |

---

#### Operands: [¶](#operands-10)

[¶](#operands-10)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |
| `forwarded_inputs` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `forwarded_inputs` | variadic of any transform handle or parameter |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance || `forwarded_inputs` | variadic of any transform handle or parameter |
 `forwarded_inputs` |`forwarded_inputs` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-4)

[¶](#results-4)

| Result | Description |
| --- | --- |
| `updated` | TransformHandleTypeInterface instance |
| `forwarded_outputs` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `updated` | TransformHandleTypeInterface instance |
| `forwarded_outputs` | variadic of any transform handle or parameter |
| `updated` | TransformHandleTypeInterface instance |
 `updated` |`updated` TransformHandleTypeInterface instance || `forwarded_outputs` | variadic of any transform handle or parameter |
 `forwarded_outputs` |`forwarded_outputs` variadic of any transform handle or parameter |

---

### `transform.foreach` (transform::ForeachOp) [¶](#transformforeach-transformforeachop)

`transform.foreach`
[¶](#transformforeach-transformforeachop)

*Executes the body for each element of the payload*

*Executes the body for each element of the payload*

Syntax:

```
operation ::= `transform.foreach` $targets oilist(`with_zip_shortest` $with_zip_shortest) `:` type($targets) (`->` type($results)^)? $body attr-dict
```

`` operation ::= `transform.foreach` $targets oilist(`with_zip_shortest` $with_zip_shortest) `:` type($targets) (`->` type($results)^)? $body attr-dict ``

Execute the op’s body - its single region block - exactly once per
element of the payload associated to a target handle. The body’s
transformations are applied in order of appearance until reaching the
(implicit) YieldOp terminator.

Each iteration gets executed by co-indexing the payloads of the arguments
and mapping the body’s arguments to these tuples, as though iterating over
the zipped together `targets`. As such, in each iteration, the size of the
payload of each of the body’s block arguments is exactly one. The attribute
`zip_shortest` can be used if the targets vary in their number of payloads;
this will limit the iterations to only the number of payloads found in the
shortest target.

`targets`
`zip_shortest`

This op always reads the target handles. Furthermore, it consumes a handle
if there is a transform op in the body that consumes the corresponding
block argument. Handles can point to ops, values, or parameters.

---

#### Return Modes [¶](#return-modes)

[¶](#return-modes)

This op produces as many result handles as the body’s terminating YieldOp
has operands. For each result, the payloads of the corresponding YieldOp
operand are merged and mapped to the same resulting handle.

If the target handles do not associate payloads of the same size, a
silencable failure will be generated.

During application, if any transformation in the sequence fails, the entire
sequence fails immediately with the same failure, leaving the payload IR in
a potentially invalid state, i.e., this operation offers no transformation
rollback capabilities.

Traits: `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

`SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionBranchOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-7)

[¶](#attributes-7)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `with_zip_shortest` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `with_zip_shortest` | ::mlir::UnitAttr | unit attribute |
 `with_zip_shortest` |`with_zip_shortest` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-11)

[¶](#operands-11)

| Operand | Description |
| --- | --- |
| `targets` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `targets` | variadic of any transform handle or parameter |
| `targets` | variadic of any transform handle or parameter |
 `targets` |`targets` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-5)

[¶](#results-5)

| Result | Description |
| --- | --- |
| `results` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of any transform handle or parameter |
| `results` | variadic of any transform handle or parameter |
 `results` |`results` variadic of any transform handle or parameter |

---

### `transform.get_consumers_of_result` (transform::GetConsumersOfResult) [¶](#transformget_consumers_of_result-transformgetconsumersofresult)

`transform.get_consumers_of_result`
[¶](#transformget_consumers_of_result-transformgetconsumersofresult)

*Get handle to the consumers of this operation’s result number*

*Get handle to the consumers of this operation’s result number*

Syntax:

```
operation ::= `transform.get_consumers_of_result` $target `[` $result_number `]` attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_consumers_of_result` $target `[` $result_number `]` attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to all operations that
consume the SSA value defined by the `target` and `result_number`
arguments.
This operation applies to a single payload operation, otherwise it produces
a definite failure.
The return handle points to the consuming operations operations, which can
be empty.

`target`
`result_number`

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-8)

[¶](#attributes-8)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `result_number` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `result_number` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `result_number` |`result_number` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-12)

[¶](#operands-12)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-6)

[¶](#results-6)

| Result | Description |
| --- | --- |
| `consumers` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `consumers` | TransformHandleTypeInterface instance |
| `consumers` | TransformHandleTypeInterface instance |
 `consumers` |`consumers` TransformHandleTypeInterface instance |

---

### `transform.get_defining_op` (transform::GetDefiningOp) [¶](#transformget_defining_op-transformgetdefiningop)

`transform.get_defining_op`
[¶](#transformget_defining_op-transformgetdefiningop)

*Get handle to the defining op of a value*

*Get handle to the defining op of a value*

Syntax:

```
operation ::= `transform.get_defining_op` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_defining_op` $target attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to the defining op of
the targeted value.

This transform produces a silenceable failure if the targeted value is a
block argument.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-13)

[¶](#operands-13)

| Operand | Description |
| --- | --- |
| `target` | TransformValueHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformValueHandleTypeInterface instance |
| `target` | TransformValueHandleTypeInterface instance |
 `target` |`target` TransformValueHandleTypeInterface instance |

---

#### Results: [¶](#results-7)

[¶](#results-7)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.get_operand` (transform::GetOperandOp) [¶](#transformget_operand-transformgetoperandop)

`transform.get_operand`
[¶](#transformget_operand-transformgetoperandop)

*Get a handle to the operand(s) of the targeted op*

*Get a handle to the operand(s) of the targeted op*

Syntax:

```
operation ::= `transform.get_operand` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_operand` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to the operands of the
given `target` operation specified by the given set of positions. There are
three possible modes:

`target`

* Position list directly, i.e. `%target[0, 1, 2]`. This will return the
  operands at the specified positions.
* Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
  all operands except those at the given positions.
* All, i.e. `%target[all]`. This will return all operands of the operation.

- Position list directly, i.e. `%target[0, 1, 2]`. This will return the
  operands at the specified positions.
`%target[0, 1, 2]`- Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
  all operands except those at the given positions.
`%target[except(0, 1, 2)]`- All, i.e. `%target[all]`. This will return all operands of the operation.
`%target[all]`

This transform produces a silenceable failure if any of the operand indices
exceeds the number of operands in the target. It reads the target handle and
produces the result handle.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-9)

[¶](#attributes-9)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `is_inverted` | ::mlir::UnitAttr | unit attribute |
| `is_all` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `raw_position_list` |`raw_position_list` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `is_inverted` | ::mlir::UnitAttr | unit attribute |
 `is_inverted` |`is_inverted` ::mlir::UnitAttr | unit attribute || `is_all` | ::mlir::UnitAttr | unit attribute |
 `is_all` |`is_all` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-14)

[¶](#operands-14)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-8)

[¶](#results-8)

| Result | Description |
| --- | --- |
| `result` | TransformValueHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformValueHandleTypeInterface instance |
| `result` | TransformValueHandleTypeInterface instance |
 `result` |`result` TransformValueHandleTypeInterface instance |

---

### `transform.get_parent_op` (transform::GetParentOp) [¶](#transformget_parent_op-transformgetparentop)

`transform.get_parent_op`
[¶](#transformget_parent_op-transformgetparentop)

*Gets handles to the closest parent ops*

*Gets handles to the closest parent ops*

Syntax:

```
operation ::= `transform.get_parent_op` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_parent_op` $target attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to the parents of the
targeted payload ops (in the same order).

Requirements that parent ops must fulfill can be optionally specified. In
that case for each target op, the closest parent op that fulfills all
requirements, is returned.

* `isolated_from_above`: the parent op must be isolated from above
* `allow_empty_results`: get\_parent\_op is allowed to return an empty list
  and still succeeds. In such a case, if `get_parent_op` fails for any
  operation in the list, the entire transform returns an empty handle.
* `op_name`: the parent op must have the specified name
* `nth_parent`: get the n-th parent of that satisfies the above requirements

- `isolated_from_above`: the parent op must be isolated from above
`isolated_from_above`- `allow_empty_results`: get\_parent\_op is allowed to return an empty list
  and still succeeds. In such a case, if `get_parent_op` fails for any
  operation in the list, the entire transform returns an empty handle.
`allow_empty_results`
`get_parent_op`- `op_name`: the parent op must have the specified name
`op_name`- `nth_parent`: get the n-th parent of that satisfies the above requirements
`nth_parent`

If `deduplicate` is set, the result handle does not contain any duplicate
ops. For example, given the list
“(childof(A), childof(B), childof(B), childof(A), childof(B))”, the
resulting list will be just “(A, B)”. Note that no other semantic ordering
is applied, e.g., “B” may itself be a parent of “A”. This may have an impact
on the further transformation applied to the handle produced here.

`deduplicate`

If any of the given Payload IR ops has no such suitable parent, then:

* if `allow_empty_results` is set, the result handle is empty
* otherwise, the transformation produces a silenceable failure.

- if `allow_empty_results` is set, the result handle is empty
`allow_empty_results`- otherwise, the transformation produces a silenceable failure.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-10)

[¶](#attributes-10)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `isolated_from_above` | ::mlir::UnitAttr | unit attribute |
| `allow_empty_results` | ::mlir::UnitAttr | unit attribute |
| `op_name` | ::mlir::StringAttr | string attribute |
| `deduplicate` | ::mlir::UnitAttr | unit attribute |
| `nth_parent` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `isolated_from_above` | ::mlir::UnitAttr | unit attribute |
 `isolated_from_above` |`isolated_from_above` ::mlir::UnitAttr | unit attribute || `allow_empty_results` | ::mlir::UnitAttr | unit attribute |
 `allow_empty_results` |`allow_empty_results` ::mlir::UnitAttr | unit attribute || `op_name` | ::mlir::StringAttr | string attribute |
 `op_name` |`op_name` ::mlir::StringAttr | string attribute || `deduplicate` | ::mlir::UnitAttr | unit attribute |
 `deduplicate` |`deduplicate` ::mlir::UnitAttr | unit attribute || `nth_parent` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |
 `nth_parent` |`nth_parent` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

---

#### Operands: [¶](#operands-15)

[¶](#operands-15)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-9)

[¶](#results-9)

| Result | Description |
| --- | --- |
| `parent` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `parent` | TransformHandleTypeInterface instance |
| `parent` | TransformHandleTypeInterface instance |
 `parent` |`parent` TransformHandleTypeInterface instance |

---

### `transform.get_producer_of_operand` (transform::GetProducerOfOperand) [¶](#transformget_producer_of_operand-transformgetproducerofoperand)

`transform.get_producer_of_operand`
[¶](#transformget_producer_of_operand-transformgetproducerofoperand)

*Get handle to the producer of this operation’s operand number*

*Get handle to the producer of this operation’s operand number*

Syntax:

```
operation ::= `transform.get_producer_of_operand` $target `[` $operand_number `]` attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_producer_of_operand` $target `[` $operand_number `]` attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to operation that
produces the SSA value defined by the `target` and `operand_number`
arguments. If the origin of the SSA value is not an operations (i.e. it is
a block argument), the transform produces a silenceable failure.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.

`target`
`operand_number`

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-11)

[¶](#attributes-11)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `operand_number` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `operand_number` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `operand_number` |`operand_number` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-16)

[¶](#operands-16)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-10)

[¶](#results-10)

| Result | Description |
| --- | --- |
| `producer` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `producer` | TransformHandleTypeInterface instance |
| `producer` | TransformHandleTypeInterface instance |
 `producer` |`producer` TransformHandleTypeInterface instance |

---

### `transform.get_result` (transform::GetResultOp) [¶](#transformget_result-transformgetresultop)

`transform.get_result`
[¶](#transformget_result-transformgetresultop)

*Get a handle to the result(s) of the targeted op*

*Get a handle to the result(s) of the targeted op*

Syntax:

```
operation ::= `transform.get_result` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.get_result` $target `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op correspond to the OpResults of the
given `target` operation. Optionally `result_number` can be specified to
select a specific result.

`target`
`result_number`

This transform fails silently if the targeted operation does not have enough
results. It reads the target handle and produces the result handle.

The handle defined by this Transform op corresponds to the results of the
given `target` operation specified by the given set of positions. There are
three possible modes:

`target`

* Position list directly, i.e. `%target[0, 1, 2]`. This will return the
  results at the specified positions.
* Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
  all results except those at the given positions.
* All, i.e. `%target[all]`. This will return all results of the operation.

- Position list directly, i.e. `%target[0, 1, 2]`. This will return the
  results at the specified positions.
`%target[0, 1, 2]`- Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
  all results except those at the given positions.
`%target[except(0, 1, 2)]`- All, i.e. `%target[all]`. This will return all results of the operation.
`%target[all]`

This transform produces a silenceable failure if any of the result indices
exceeds the number of results returned by the target. It reads the target
handle and produces the result handle.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-12)

[¶](#attributes-12)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `is_inverted` | ::mlir::UnitAttr | unit attribute |
| `is_all` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `raw_position_list` |`raw_position_list` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `is_inverted` | ::mlir::UnitAttr | unit attribute |
 `is_inverted` |`is_inverted` ::mlir::UnitAttr | unit attribute || `is_all` | ::mlir::UnitAttr | unit attribute |
 `is_all` |`is_all` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-17)

[¶](#operands-17)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-11)

[¶](#results-11)

| Result | Description |
| --- | --- |
| `result` | TransformValueHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformValueHandleTypeInterface instance |
| `result` | TransformValueHandleTypeInterface instance |
 `result` |`result` TransformValueHandleTypeInterface instance |

---

### `transform.get_type` (transform::GetTypeOp) [¶](#transformget_type-transformgettypeop)

`transform.get_type`
[¶](#transformget_type-transformgettypeop)

*Get a parameter containing the type of the given value*

*Get a parameter containing the type of the given value*

Syntax:

```
operation ::= `transform.get_type` (`elemental` $elemental^)? $value attr-dict `:`functional-type(operands, results)
```

`` operation ::= `transform.get_type` (`elemental` $elemental^)? $value attr-dict `:`functional-type(operands, results) ``

This operation creates a new Transform parameter containing the
type(s) of the value(s) associated with the operand handle.

This transform never fails.

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-13)

[¶](#attributes-13)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `elemental` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `elemental` | ::mlir::UnitAttr | unit attribute |
 `elemental` |`elemental` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-18)

[¶](#operands-18)

| Operand | Description |
| --- | --- |
| `value` | TransformValueHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `value` | TransformValueHandleTypeInterface instance |
| `value` | TransformValueHandleTypeInterface instance |
 `value` |`value` TransformValueHandleTypeInterface instance |

---

#### Results: [¶](#results-12)

[¶](#results-12)

| Result | Description |
| --- | --- |
| `type_param` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `type_param` | TransformParamTypeInterface instance |
| `type_param` | TransformParamTypeInterface instance |
 `type_param` |`type_param` TransformParamTypeInterface instance |

---

### `transform.include` (transform::IncludeOp) [¶](#transforminclude-transformincludeop)

`transform.include`
[¶](#transforminclude-transformincludeop)

*Includes a named transform sequence*

*Includes a named transform sequence*

Syntax:

```
operation ::= `transform.include` $target `failures` `(` $failure_propagation_mode `)``(` $operands `)` attr-dict `:` functional-type($operands, $results)
```

``` operation ::= `transform.include` $target `failures` `(` $failure_propagation_mode `)``(` $operands `)` attr-dict `:` functional-type($operands, $results) ```

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

`failure_propagation_mode`
`propagate`
`suppress`
`transform.yield`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallOpInterface`, `MatchOpInterface`, `MemoryEffectOpInterface`, `SymbolUserOpInterface`, `TransformOpInterface`

`ArgAndResultAttrsOpInterface`
`CallOpInterface`
`MatchOpInterface`
`MemoryEffectOpInterface`
`SymbolUserOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-14)

[¶](#attributes-14)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `target` | ::mlir::SymbolRefAttr | symbol reference attribute |
| `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `target` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `target` |`target` ::mlir::SymbolRefAttr | symbol reference attribute || `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |
 `failure_propagation_mode` |`failure_propagation_mode` ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy || `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
 `arg_attrs` |`arg_attrs` ::mlir::ArrayAttr | Array of dictionary attributes || `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
 `res_attrs` |`res_attrs` ::mlir::ArrayAttr | Array of dictionary attributes |

---

#### Operands: [¶](#operands-19)

[¶](#operands-19)

| Operand | Description |
| --- | --- |
| `operands` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operands` | variadic of any transform handle or parameter |
| `operands` | variadic of any transform handle or parameter |
 `operands` |`operands` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-13)

[¶](#results-13)

| Result | Description |
| --- | --- |
| `results` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of any transform handle or parameter |
| `results` | variadic of any transform handle or parameter |
 `results` |`results` variadic of any transform handle or parameter |

---

### `transform.match.operation_empty` (transform::MatchOperationEmptyOp) [¶](#transformmatchoperation_empty-transformmatchoperationemptyop)

`transform.match.operation_empty`
[¶](#transformmatchoperation_empty-transformmatchoperationemptyop)

*Matches if the handle is not associated to any op*

*Matches if the handle is not associated to any op*

Syntax:

```
operation ::= `transform.match.operation_empty` $operand_handle attr-dict `:` type($operand_handle)
```

`` operation ::= `transform.match.operation_empty` $operand_handle attr-dict `:` type($operand_handle) ``

Succeeds if the handle is not associated to any op.

Traits: `AtMostOneOpMatcher`

`AtMostOneOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-20)

[¶](#operands-20)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

### `transform.match.operation_name` (transform::MatchOperationNameOp) [¶](#transformmatchoperation_name-transformmatchoperationnameop)

`transform.match.operation_name`
[¶](#transformmatchoperation_name-transformmatchoperationnameop)

*Matches a single operation of one of the given kinds*

*Matches a single operation of one of the given kinds*

Syntax:

```
operation ::= `transform.match.operation_name` $operand_handle $op_names attr-dict `:` type($operand_handle)
```

`` operation ::= `transform.match.operation_name` $operand_handle $op_names attr-dict `:` type($operand_handle) ``

Succeeds if the operation associated with the operand handle has one of the
given operation names. Produces a silenceable failure otherwise.

If more than one payload operation is associated with the operand handle,
produces a definite failure.

Traits: `SingleOpMatcher`

`SingleOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-15)

[¶](#attributes-15)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `op_names` | ::mlir::ArrayAttr | string array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `op_names` | ::mlir::ArrayAttr | string array attribute |
 `op_names` |`op_names` ::mlir::ArrayAttr | string array attribute |

---

#### Operands: [¶](#operands-21)

[¶](#operands-21)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

### `transform.match.param.cmpi` (transform::MatchParamCmpIOp) [¶](#transformmatchparamcmpi-transformmatchparamcmpiop)

`transform.match.param.cmpi`
[¶](#transformmatchparamcmpi-transformmatchparamcmpiop)

*Matches if two parameter lists are associated with the same value*

*Matches if two parameter lists are associated with the same value*

Syntax:

```
operation ::= `transform.match.param.cmpi` $predicate $param `,` $reference attr-dict `:` type($param)
```

`` operation ::= `transform.match.param.cmpi` $predicate $param `,` $reference attr-dict `:` type($param) ``

Succeeds if all of the co-indexed values associated with the given
parameters relate as specified by the predicate (greater than, less than,
equal to, or their combinations). Comparison treats all values as signed.
Produces a silenceable failure otherwise.

Traits: `SameTypeOperands`

`SameTypeOperands`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-16)

[¶](#attributes-16)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `predicate` | ::mlir::transform::MatchCmpIPredicateAttr | allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5 |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `predicate` | ::mlir::transform::MatchCmpIPredicateAttr | allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5 |
 `predicate` |`predicate` ::mlir::transform::MatchCmpIPredicateAttr | allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5 |

---

#### Operands: [¶](#operands-22)

[¶](#operands-22)

| Operand | Description |
| --- | --- |
| `param` | TransformParamTypeInterface instance |
| `reference` | TransformParamTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `param` | TransformParamTypeInterface instance |
| `reference` | TransformParamTypeInterface instance |
| `param` | TransformParamTypeInterface instance |
 `param` |`param` TransformParamTypeInterface instance || `reference` | TransformParamTypeInterface instance |
 `reference` |`reference` TransformParamTypeInterface instance |

---

### `transform.merge_handles` (transform::MergeHandlesOp) [¶](#transformmerge_handles-transformmergehandlesop)

`transform.merge_handles`
[¶](#transformmerge_handles-transformmergehandlesop)

*Merges handles into one pointing to the union of payload ops*

*Merges handles into one pointing to the union of payload ops*

Syntax:

```
operation ::= `transform.merge_handles` (`deduplicate` $deduplicate^)? $handles attr-dict `:` type($result)
```

`` operation ::= `transform.merge_handles` (`deduplicate` $deduplicate^)? $handles attr-dict `:` type($result) ``

Creates a new Transform IR handle value that points to the same Payload IR
operations/values/parameters as the operand handles. The Payload IR elements
are listed in the same order as they are in the operand handles, grouped by
operand handle, e.g., all Payload IR associated with the first handle comes
first, then all Payload IR associated with the second handle and so on. If
`deduplicate` is set, do not add the given Payload IR operation, value, or
parameter more than once to the final list regardless of it coming from the
same or different handles. Consumes the operands and produces a new handle.

`deduplicate`

Traits: `SameOperandsAndResultType`

`SameOperandsAndResultType`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-17)

[¶](#attributes-17)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `deduplicate` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `deduplicate` | ::mlir::UnitAttr | unit attribute |
 `deduplicate` |`deduplicate` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-23)

[¶](#operands-23)

| Operand | Description |
| --- | --- |
| `handles` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `handles` | variadic of any transform handle or parameter |
| `handles` | variadic of any transform handle or parameter |
 `handles` |`handles` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-14)

[¶](#results-14)

| Result | Description |
| --- | --- |
| `result` | any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | any transform handle or parameter |
| `result` | any transform handle or parameter |
 `result` |`result` any transform handle or parameter |

---

### `transform.named_sequence` (transform::NamedSequenceOp) [¶](#transformnamed_sequence-transformnamedsequenceop)

`transform.named_sequence`
[¶](#transformnamed_sequence-transformnamedsequenceop)

*Named transform sequence that can be included elsewhere*

*Named transform sequence that can be included elsewhere*

Defines a named (callable, function-like) sequence of other Transform
dialect operations that can be included using `transform.include` as part of
another Transform dialect construct. This sequence is not processed
immediately but rather dispatched to when the inclusion is processed. The
arguments and results can be used to communicate a subset of mapping into
the named sequence. The sequence must consist of a single block and end with
a `transform.yield` terminator. The operands of the terminator become the
results of the `transform.include`.

`transform.include`
`transform.yield`
`transform.include`

When dispatched to, the operations in the named sequence are executed one by
one, similarly to the regular unnamed sequence. The failure propagation mode
is specified on the `transform.include`. Different inclusions may use
different failure propagation modes. This transform operation always
succeeds by itself, but the inclusion may fail if any of the operations
fail.

`transform.include`

Named sequences can only appear at the top-level of the Transform dialect
nesting structure. That is, they cannot be nested in other Transform dialect
operations. Furthermore, one of the ancestors must have the `SymbolTable`
trait and have the `transform.with_named_sequence` attribute attached.

`SymbolTable`
`transform.with_named_sequence`

Named sequences may include other named sequences via `transform.include`,
but recursion is *not* allowed.

`transform.include`
*not*

Traits: `IsolatedFromAbove`

`IsolatedFromAbove`

Interfaces: `ArgAndResultAttrsOpInterface`, `CallableOpInterface`, `FunctionOpInterface`, `MemoryEffectOpInterface`, `Symbol`, `TransformOpInterface`

`ArgAndResultAttrsOpInterface`
`CallableOpInterface`
`FunctionOpInterface`
`MemoryEffectOpInterface`
`Symbol`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-18)

[¶](#attributes-18)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `sym_name` | ::mlir::StringAttr | string attribute |
| `function_type` | ::mlir::TypeAttr | function type attribute |
| `sym_visibility` | ::mlir::StringAttr | string attribute |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `sym_name` | ::mlir::StringAttr | string attribute |
 `sym_name` |`sym_name` ::mlir::StringAttr | string attribute || `function_type` | ::mlir::TypeAttr | function type attribute |
 `function_type` |`function_type` ::mlir::TypeAttr | function type attribute || `sym_visibility` | ::mlir::StringAttr | string attribute |
 `sym_visibility` |`sym_visibility` ::mlir::StringAttr | string attribute || `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
 `arg_attrs` |`arg_attrs` ::mlir::ArrayAttr | Array of dictionary attributes || `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
 `res_attrs` |`res_attrs` ::mlir::ArrayAttr | Array of dictionary attributes |

---

### `transform.num_associations` (transform::NumAssociationsOp) [¶](#transformnum_associations-transformnumassociationsop)

`transform.num_associations`
[¶](#transformnum_associations-transformnumassociationsop)

*Returns the number of payload objects associated with the argument*

*Returns the number of payload objects associated with the argument*

Syntax:

```
operation ::= `transform.num_associations` $handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.num_associations` $handle attr-dict `:` functional-type(operands, results) ``

Given an argument, handle or parameter, returns a new parameter associated
with a single 64-bit number that corresponds to the number of payload
objects (operations or values for a handle, attributes for a parameter)
associated with the argument.

Always succeeds.

Traits: `ParamProducerTransformOpTrait`

`ParamProducerTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-24)

[¶](#operands-24)

| Operand | Description |
| --- | --- |
| `handle` | any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `handle` | any transform handle or parameter |
| `handle` | any transform handle or parameter |
 `handle` |`handle` any transform handle or parameter |

---

#### Results: [¶](#results-15)

[¶](#results-15)

| Result | Description |
| --- | --- |
| `num` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `num` | TransformParamTypeInterface instance |
| `num` | TransformParamTypeInterface instance |
 `num` |`num` TransformParamTypeInterface instance |

---

### `transform.param.constant` (transform::ParamConstantOp) [¶](#transformparamconstant-transformparamconstantop)

`transform.param.constant`
[¶](#transformparamconstant-transformparamconstantop)

*Produces a new transform dialect parameter value associated with the given attribute*

*Produces a new transform dialect parameter value associated with the given attribute*

Syntax:

```
operation ::= `transform.param.constant` $value attr-dict `->` type($param)
```

`` operation ::= `transform.param.constant` $value attr-dict `->` type($param) ``

Produces a new transform dialect parameter associated with the singleton
list containing the given attribute. The operation itself always succeeds,
but the general association check may fail if the parameter type does not
accept the given kind of attribute as valid.

Traits: `ParamProducerTransformOpTrait`

`ParamProducerTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-19)

[¶](#attributes-19)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `value` | ::mlir::Attribute | any attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `value` | ::mlir::Attribute | any attribute |
 `value` |`value` ::mlir::Attribute | any attribute |

---

#### Results: [¶](#results-16)

[¶](#results-16)

| Result | Description |
| --- | --- |
| `param` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `param` | TransformParamTypeInterface instance |
| `param` | TransformParamTypeInterface instance |
 `param` |`param` TransformParamTypeInterface instance |

---

### `transform.print` (transform::PrintOp) [¶](#transformprint-transformprintop)

`transform.print`
[¶](#transformprint-transformprintop)

*Dump each payload op*

*Dump each payload op*

Syntax:

```
operation ::= `transform.print` $target attr-dict (`:` type($target)^)?
```

`` operation ::= `transform.print` $target attr-dict (`:` type($target)^)? ``

Prints each payload op that is associated with the `target` operand to
`stdout`. It also prints the `name` string attribute. If no target is
specified, the top-level op is dumped.

`target`
`stdout`
`name`

This op is useful for printf-style debugging.

Supported printing flag attributes:

* `assume_verified` – skips verification when the unit attribute is
  specified. This improves performace but may lead to crashes and
  unexpected behavior when the printed payload op is invalid.
* `use_local_scope` – prints in local scope when the unit attribute is
  specified. This improves performance but may not be identical to
  printing within the full module.
* `skip_regions` – does not print regions of operations when the unit
  attribute is specified.

- `assume_verified` – skips verification when the unit attribute is
  specified. This improves performace but may lead to crashes and
  unexpected behavior when the printed payload op is invalid.
`assume_verified`- `use_local_scope` – prints in local scope when the unit attribute is
  specified. This improves performance but may not be identical to
  printing within the full module.
`use_local_scope`- `skip_regions` – does not print regions of operations when the unit
  attribute is specified.
`skip_regions`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-20)

[¶](#attributes-20)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | string attribute |
| `assume_verified` | ::mlir::UnitAttr | unit attribute |
| `use_local_scope` | ::mlir::UnitAttr | unit attribute |
| `skip_regions` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `name` | ::mlir::StringAttr | string attribute |
 `name` |`name` ::mlir::StringAttr | string attribute || `assume_verified` | ::mlir::UnitAttr | unit attribute |
 `assume_verified` |`assume_verified` ::mlir::UnitAttr | unit attribute || `use_local_scope` | ::mlir::UnitAttr | unit attribute |
 `use_local_scope` |`use_local_scope` ::mlir::UnitAttr | unit attribute || `skip_regions` | ::mlir::UnitAttr | unit attribute |
 `skip_regions` |`skip_regions` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-25)

[¶](#operands-25)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.replicate` (transform::ReplicateOp) [¶](#transformreplicate-transformreplicateop)

`transform.replicate`
[¶](#transformreplicate-transformreplicateop)

*Lists payload ops multiple times in the new handle*

*Lists payload ops multiple times in the new handle*

Syntax:

```
operation ::= `transform.replicate` `num` `(` $pattern `)` $handles attr-dict `:` type($pattern) `,` type($handles)
```

`` operation ::= `transform.replicate` `num` `(` $pattern `)` $handles attr-dict `:` type($pattern) `,` type($handles) ``

Produces a new handle associated with a list of payload IR ops that is
computed by repeating the list of payload IR ops associated with the
operand handle as many times as the “pattern” handle has associated
operations. For example, if pattern is associated with [op1, op2] and the
operand handle is associated with [op3, op4, op5], the resulting handle
will be associated with [op3, op4, op5, op3, op4, op5].

This transformation is useful to “align” the sizes of payload IR lists
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

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-26)

[¶](#operands-26)

| Operand | Description |
| --- | --- |
| `pattern` | TransformHandleTypeInterface instance |
| `handles` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `pattern` | TransformHandleTypeInterface instance |
| `handles` | variadic of any transform handle or parameter |
| `pattern` | TransformHandleTypeInterface instance |
 `pattern` |`pattern` TransformHandleTypeInterface instance || `handles` | variadic of any transform handle or parameter |
 `handles` |`handles` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-17)

[¶](#results-17)

| Result | Description |
| --- | --- |
| `replicated` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `replicated` | variadic of any transform handle or parameter |
| `replicated` | variadic of any transform handle or parameter |
 `replicated` |`replicated` variadic of any transform handle or parameter |

---

### `transform.select` (transform::SelectOp) [¶](#transformselect-transformselectop)

`transform.select`
[¶](#transformselect-transformselectop)

*Select payload ops by name*

*Select payload ops by name*

Syntax:

```
operation ::= `transform.select` $op_name `in` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.select` $op_name `in` $target attr-dict `:` functional-type(operands, results) ``

The handle defined by this Transform op corresponds to all operations among
`target` that have the specified properties. Currently the following
properties are supported:

`target`

* `op_name`: The op must have the specified name.

- `op_name`: The op must have the specified name.
`op_name`

The result payload ops are in the same relative order as the targeted ops.
This transform op reads the `target` handle and produces the `result`
handle. It reads the payload, but does not modify it.

`target`
`result`

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-21)

[¶](#attributes-21)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `op_name` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `op_name` | ::mlir::StringAttr | string attribute |
 `op_name` |`op_name` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-27)

[¶](#operands-27)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-18)

[¶](#results-18)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.sequence` (transform::SequenceOp) [¶](#transformsequence-transformsequenceop)

`transform.sequence`
[¶](#transformsequence-transformsequenceop)

*Contains a sequence of other transform ops to apply*

*Contains a sequence of other transform ops to apply*

Syntax:

```
operation ::= `transform.sequence` custom<SequenceOpOperands>($root, type($root), $extra_bindings, type($extra_bindings)) (`->` type($results)^)? `failures` `(` $failure_propagation_mode `)` attr-dict-with-keyword regions
```

`` operation ::= `transform.sequence` custom<SequenceOpOperands>($root, type($root), $extra_bindings, type($extra_bindings)) (`->` type($results)^)? `failures` `(` $failure_propagation_mode `)` attr-dict-with-keyword regions ``

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

`failure_propagation_mode`
`propagate`
`suppress`

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.

The type of the block argument must match the type of the operand. If the
sequence is a top-level transform (without an operand), it can be used for
matching operations if the specified type within the top-level container
payload IR (including the container op itself). E.g.:

```
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

```
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

`transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
 // %arg1 is mapped to the top-level container of the payload IR, which is
 // typically a module
}
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.op<"func.func>"):
 // %arg1 is mapped to all "func.func" ops within and including the
 // top-level container of the payload IR. Nested operations that have the
 // specified op type are not included.
}`
transform.sequence failures(propagate) {
transform.sequence failures(propagate) {
.
(
)
{
^bb1(%arg1: !transform.any\_op):
^bb1(%arg1: !transform.any\_op):
^bb1
(
%arg1
:
!
.
):
 // %arg1 is mapped to the top-level container of the payload IR, which is
 // %arg1 is mapped to the top-level container of the payload IR, which is
// %arg1 is mapped to the top-level container of the payload IR, which is
 // typically a module
 // typically a module

// typically a module
}
}

}




transform.sequence failures(propagate) {
transform.sequence failures(propagate) {
.
(
)
{
^bb1(%arg1: !transform.op<"func.func>"):
^bb1(%arg1: !transform.op<"func.func>"):
^bb1
(
%arg1
:
!
.
<
"func.func>"
):
 // %arg1 is mapped to all "func.func" ops within and including the
 // %arg1 is mapped to all "func.func" ops within and including the
// %arg1 is mapped to all "func.func" ops within and including the
 // top-level container of the payload IR. Nested operations that have the
 // top-level container of the payload IR. Nested operations that have the

// top-level container of the payload IR. Nested operations that have the
 // specified op type are not included.
 // specified op type are not included.

// specified op type are not included.
}
}

}

The body of the sequence terminates with an implicit or explicit
`transform.yield` op. The operands of the terminator are returned as the
results of the sequence op.

`transform.yield`

Traits: `AttrSizedOperandSegments`, `PossibleTopLevelTransformOpTrait`, `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

`AttrSizedOperandSegments`
`PossibleTopLevelTransformOpTrait`
`SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`
`SingleBlock`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `OpAsmOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`OpAsmOpInterface`
`RegionBranchOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-22)

[¶](#attributes-22)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |
 `failure_propagation_mode` |`failure_propagation_mode` ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |

---

#### Operands: [¶](#operands-28)

[¶](#operands-28)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |
| `extra_bindings` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `extra_bindings` | variadic of any transform handle or parameter |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance || `extra_bindings` | variadic of any transform handle or parameter |
 `extra_bindings` |`extra_bindings` variadic of any transform handle or parameter |

---

#### Results: [¶](#results-19)

[¶](#results-19)

| Result | Description |
| --- | --- |
| `results` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of TransformHandleTypeInterface instance |
| `results` | variadic of TransformHandleTypeInterface instance |
 `results` |`results` variadic of TransformHandleTypeInterface instance |

---

### `transform.split_handle` (transform::SplitHandleOp) [¶](#transformsplit_handle-transformsplithandleop)

`transform.split_handle`
[¶](#transformsplit_handle-transformsplithandleop)

*Splits a handle or parameter into multiple values*

*Splits a handle or parameter into multiple values*

Syntax:

```
operation ::= `transform.split_handle` $handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.split_handle` $handle attr-dict `:` functional-type(operands, results) ``

Splits `handle` into one or multiple handles, as specified by the number
of results of this operation. `handle` should be mapped to as many payload
ops, values or parameteres as there are results. Otherwise, this transform
will fail producing a silenceable failure by default. Each result handle
is mapped to exactly one payload unless specified otherwise by attributes
described below. The order of the payloads is preserved, i.e., the i-th
payload is mapped to the i-th result handle.

`handle`
`handle`

This operation is useful for ensuring a statically known number of
payloads are tracked by the source `handle` and to extract them into
individual handles that can be further manipulated in isolation.

`handle`

If there are more payloads than results, the remaining payloads are mapped to
the result with index `overflow_result`. If no `overflow_result` is
specified, the transform produces a silenceable failure.

`overflow_result`
`overflow_result`

If there are fewer payload ops than results, the transform produces a
silenceable failure if `fail_on_payload_too_small` is set to “true”.
Otherwise, it succeeds and the remaining result handles are not mapped to
anything. It also succeeds if `handle` is empty and
`pass_through_empty_handle` is set to “true”, regardless of
`fail_on_payload_too_small`.

`fail_on_payload_too_small`
`handle`
`pass_through_empty_handle`
`fail_on_payload_too_small`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-23)

[¶](#attributes-23)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `pass_through_empty_handle` | ::mlir::BoolAttr | bool attribute |
| `fail_on_payload_too_small` | ::mlir::BoolAttr | bool attribute |
| `overflow_result` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `pass_through_empty_handle` | ::mlir::BoolAttr | bool attribute |
 `pass_through_empty_handle` |`pass_through_empty_handle` ::mlir::BoolAttr | bool attribute || `fail_on_payload_too_small` | ::mlir::BoolAttr | bool attribute |
 `fail_on_payload_too_small` |`fail_on_payload_too_small` ::mlir::BoolAttr | bool attribute || `overflow_result` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `overflow_result` |`overflow_result` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-29)

[¶](#operands-29)

| Operand | Description |
| --- | --- |
| `handle` | any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `handle` | any transform handle or parameter |
| `handle` | any transform handle or parameter |
 `handle` |`handle` any transform handle or parameter |

---

#### Results: [¶](#results-20)

[¶](#results-20)

| Result | Description |
| --- | --- |
| `results` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of any transform handle or parameter |
| `results` | variadic of any transform handle or parameter |
 `results` |`results` variadic of any transform handle or parameter |

---

### `transform.verify` (transform::VerifyOp) [¶](#transformverify-transformverifyop)

`transform.verify`
[¶](#transformverify-transformverifyop)

*Verifies the targeted ops*

*Verifies the targeted ops*

Syntax:

```
operation ::= `transform.verify` $target attr-dict `:` type($target)
```

`` operation ::= `transform.verify` $target attr-dict `:` type($target) ``

This transform verifies the targeted ops. If at least one op fails to
verify, the transform produces a definite failure.

Note: This op was designed for debugging purposes and should be used like an
assertion. It is intentional that this op produces a definite failure and
not a silenceable one. Correctness of the program should not depend on this
op.

This transform reads the target handle.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-30)

[¶](#operands-30)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.yield` (transform::YieldOp) [¶](#transformyield-transformyieldop)

`transform.yield`
[¶](#transformyield-transformyieldop)

*Yields operation handles from a transform IR region*

*Yields operation handles from a transform IR region*

Syntax:

```
operation ::= `transform.yield` operands attr-dict (`:` type($operands)^)?
```

`` operation ::= `transform.yield` operands attr-dict (`:` type($operands)^)? ``

This terminator operation yields operation handles from regions of the
transform IR ops back to the containing op. It is not itself associated with
any transformation on the payload IR and is used for flow purposes only.

Traits: `Terminator`

`Terminator`

Interfaces: `MemoryEffectOpInterface`

`MemoryEffectOpInterface`

---

#### Operands: [¶](#operands-31)

[¶](#operands-31)

| Operand | Description |
| --- | --- |
| `operands` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operands` | variadic of any transform handle or parameter |
| `operands` | variadic of any transform handle or parameter |
 `operands` |`operands` variadic of any transform handle or parameter |

---

### `transform.tune.alternatives` ( mlir::transform::tune ::AlternativesOp) [¶](#transformtunealternatives--mlirtransformtune-alternativesop)

`transform.tune.alternatives`
[¶](#transformtunealternatives--mlirtransformtune-alternativesop)

*Represents a choice among its regions, i.e. sub-schedules*

*Represents a choice among its regions, i.e. sub-schedules*

Syntax:

```
operation ::= `transform.tune.alternatives` `<` $name `>`
              (`selected_region` `=` custom<AlternativesOpSelectedRegion>(
              $selected_region_attr, $selected_region_param)^)?
              attr-dict-with-keyword
              (`:` type($selected_region_param)^)?
              (`->` type($results)^)?
              regions
```

`` operation ::= `transform.tune.alternatives` `<` $name `>`
(`selected_region` `=` custom<AlternativesOpSelectedRegion>(
$selected_region_attr, $selected_region_param)^)?
attr-dict-with-keyword
(`:` type($selected_region_param)^)?
(`->` type($results)^)?
regions ``

This op represents a choice over which of its regions is to be used.

When `selected_region` is provided, the semantics are that this op is to be
substituted for by the selected region, meaning the region’s results become
the results of this op. Without a provided `selected_region`, the semantics
are that this non-deterministic choice is yet to be resolved – which in
terms of the op’s interpreted semantics is a failure.

`selected_region`
`selected_region`

The `selected_region` argument is either an `IntegerAttr` or a param holding
an `IntegerAttr`, which should provide a valid zero-based index with respect
to the number of alternatives, i.e. regions.

`selected_region`
`IntegerAttr`
`IntegerAttr`

Traits: `NoRegionArguments`, `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

`NoRegionArguments`
`SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionBranchOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionBranchOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-24)

[¶](#attributes-24)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `name` | ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |
 `name` |`name` ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |An Attribute containing a string

```
Syntax:

```
string-attribute ::= string-literal (`:` type)?
```



A string attribute is an attribute that represents a string literal value.



Examples:



```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```
```

An Attribute containing a string

```
Syntax:

```
string-attribute ::= string-literal (`:` type)?
```



A string attribute is an attribute that represents a string literal value.



Examples:



```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```
```

```` Syntax:

```
string-attribute ::= string-literal (`:` type)?
```

A string attribute is an attribute that represents a string literal value.

Examples:

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
``` ````

```
string-attribute ::= string-literal (`:` type)?
```

`` string-attribute ::= string-literal (`:` type)? ``

A string attribute is an attribute that represents a string literal value.

Examples:

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```

`&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string`
&quot;An important string&quot;
&quot;An important string&quot;
&
;
&
;
&quot;string with a type&quot; : !dialect.string
&quot;string with a type&quot; : !dialect.string
&
;
&
;
:
!
.| `selected_region_attr` | ::mlir::IntegerAttr | arbitrary integer attribute |
 `selected_region_attr` |`selected_region_attr` ::mlir::IntegerAttr | arbitrary integer attribute |

---

#### Operands: [¶](#operands-32)

[¶](#operands-32)

| Operand | Description |
| --- | --- |
| `selected_region_param` | TransformParamTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `selected_region_param` | TransformParamTypeInterface instance |
| `selected_region_param` | TransformParamTypeInterface instance |
 `selected_region_param` |`selected_region_param` TransformParamTypeInterface instance |

---

#### Results: [¶](#results-21)

[¶](#results-21)

| Result | Description |
| --- | --- |
| `results` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of any transform handle or parameter |
| `results` | variadic of any transform handle or parameter |
 `results` |`results` variadic of any transform handle or parameter |

---

### `transform.tune.knob` ( mlir::transform::tune ::KnobOp) [¶](#transformtuneknob--mlirtransformtune-knobop)

`transform.tune.knob`
[¶](#transformtuneknob--mlirtransformtune-knobop)

*Represents a tunable parameter with a set of options*

*Represents a tunable parameter with a set of options*

Syntax:

```
operation ::= `transform.tune.knob` `<` $name `>` (`=` $selected^ `from`)? `options` `=` $options attr-dict `->` type(results)
```

`` operation ::= `transform.tune.knob` `<` $name `>` (`=` $selected^ `from`)? `options` `=` $options attr-dict `->` type(results) ``

Provides a representation for “tunables” within schedules.

Each op represents a single tunable, which has a `name` and a set
of valid `options` described by an attribute. Without a specified
`selected` option, this op represents a non-deterministic choice
that has yet to be resolved – as such, the interpreter runtime
semantics is to raise a failure.

`name`
`options`
`selected`

The non-deterministic choice is resolved through providing a
`selected` attribute. When provided, the interpreter runtime
semantics are to return the `selected` attribute as a param through
the op’s result.

`selected`
`selected`


---

In case the `options` attribute is an `ArrayAttr`, the verifier
checks that the provided `selected` attribute occurs in `options`.

`options`
`ArrayAttr`
`selected`
`options`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-25)

[¶](#attributes-25)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `name` | ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |
| `options` | ::mlir::Attribute | any attribute |
| `selected` | ::mlir::Attribute | any attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `name` | ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |
 `name` |`name` ::mlir::StringAttr | An Attribute containing a string  ``` Syntax:  ``` string-attribute ::= string-literal (`:` type)? ```   A string attribute is an attribute that represents a string literal value.  Examples:  ``` &quot;An important string&quot; &quot;string with a type&quot; : !dialect.string ``` ``` |An Attribute containing a string

```
Syntax:

```
string-attribute ::= string-literal (`:` type)?
```



A string attribute is an attribute that represents a string literal value.



Examples:



```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```
```

An Attribute containing a string

```
Syntax:

```
string-attribute ::= string-literal (`:` type)?
```



A string attribute is an attribute that represents a string literal value.



Examples:



```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```
```

```` Syntax:

```
string-attribute ::= string-literal (`:` type)?
```

A string attribute is an attribute that represents a string literal value.

Examples:

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
``` ````

```
string-attribute ::= string-literal (`:` type)?
```

`` string-attribute ::= string-literal (`:` type)? ``

A string attribute is an attribute that represents a string literal value.

Examples:

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```

```
&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string
```

`&quot;An important string&quot;
&quot;string with a type&quot; : !dialect.string`
&quot;An important string&quot;
&quot;An important string&quot;
&
;
&
;
&quot;string with a type&quot; : !dialect.string
&quot;string with a type&quot; : !dialect.string
&
;
&
;
:
!
.| `options` | ::mlir::Attribute | any attribute |
 `options` |`options` ::mlir::Attribute | any attribute || `selected` | ::mlir::Attribute | any attribute |
 `selected` |`selected` ::mlir::Attribute | any attribute |

---

#### Results: [¶](#results-22)

[¶](#results-22)

| Result | Description |
| --- | --- |
| `result` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformParamTypeInterface instance |
| `result` | TransformParamTypeInterface instance |
 `result` |`result` TransformParamTypeInterface instance |

---

### `transform.smt.constrain_params` ( mlir::transform::smt ::ConstrainParamsOp) [¶](#transformsmtconstrain_params--mlirtransformsmt-constrainparamsop)

`transform.smt.constrain_params`
[¶](#transformsmtconstrain_params--mlirtransformsmt-constrainparamsop)

*Express contraints on params interpreted as symbolic values*

*Express contraints on params interpreted as symbolic values*

Syntax:

```
operation ::= `transform.smt.constrain_params` `(` $params `)` attr-dict `:` functional-type(operands, results) $body
```

`` operation ::= `transform.smt.constrain_params` `(` $params `)` attr-dict `:` functional-type(operands, results) $body ``

Allows expressing constraints on params using the SMT dialect.

Each Transform-dialect param provided as an operand has a corresponding
argument of SMT-type in the region. The SMT-Dialect ops in the region use
these params-as-SMT-vars as operands, thereby expressing relevant
constraints on their allowed values.

Computations w.r.t. passed-in params can also be expressed through the
region’s SMT-ops. Namely, the constraints express relationships to other
SMT-variables which can then be yielded from the region (with `smt.yield`).

`smt.yield`

The semantics of this op is that all the ops in the region together express
a constraint on the params-interpreted-as-smt-vars. The op fails in case the
expressed constraint is not satisfiable per SMTLIB semantics. Otherwise the
op succeeds and any one satisfying assignment is used to map the
SMT-variables yielded in the region to `transform.param`s.

`transform.param`


---

TODO: currently the operational semantics per the Transform interpreter is
to always fail. The intention is build out support for hooking in your own
operational semantics so you can invoke your favourite solver to determine
satisfiability of the corresponding constraint problem.

Traits: `SingleBlockImplicitTerminator<::mlir::smt::YieldOp>`, `SingleBlock`

`SingleBlockImplicitTerminator<::mlir::smt::YieldOp>`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-33)

[¶](#operands-33)

| Operand | Description |
| --- | --- |
| `params` | variadic of TransformParamTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `params` | variadic of TransformParamTypeInterface instance |
| `params` | variadic of TransformParamTypeInterface instance |
 `params` |`params` variadic of TransformParamTypeInterface instance |

---

#### Results: [¶](#results-23)

[¶](#results-23)

| Result | Description |
| --- | --- |
| `results` | variadic of TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | variadic of TransformParamTypeInterface instance |
| `results` | variadic of TransformParamTypeInterface instance |
 `results` |`results` variadic of TransformParamTypeInterface instance |

---

### `transform.affine.simplify_bounded_affine_ops` (transform::SimplifyBoundedAffineOpsOp) [¶](#transformaffinesimplify_bounded_affine_ops-transformsimplifyboundedaffineopsop)

`transform.affine.simplify_bounded_affine_ops`
[¶](#transformaffinesimplify_bounded_affine_ops-transformsimplifyboundedaffineopsop)

Syntax:

```
operation ::= `transform.affine.simplify_bounded_affine_ops` $target `with` `[` ($bounded_values^ `:` type($bounded_values))? `]`
              `within` $lower_bounds `and` $upper_bounds attr-dict
              `:` type($target)
```

`` operation ::= `transform.affine.simplify_bounded_affine_ops` $target `with` `[` ($bounded_values^ `:` type($bounded_values))? `]`
`within` $lower_bounds `and` $upper_bounds attr-dict
`:` type($target) ``

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

`%0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1
%1 = transform.structured.match ops{["gpu.lane_id"]} in %arg1
transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [32]
// Multiple bounds can be specified.
transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 5] and [32, 50]`

Bounded op handles (`%1` and `%2) must be mapped to ops that have a single
result of index type. The sets of target ops and bounded ops must not
overlap.

`%1`

---

#### Return modes [¶](#return-modes-1)

[¶](#return-modes-1)

Target ops must be affine.min or affine.max ops. This transform consumes the
target handle and does not produce any handle. It reads the bounded op
handles.

TODO: Support affine.apply targets.
TODO: Allow mixed PDL\_Operation/int64\_t for lower\_bounds and upper\_bounds.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-26)

[¶](#attributes-26)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lower_bounds` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `upper_bounds` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lower_bounds` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `lower_bounds` |`lower_bounds` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `upper_bounds` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `upper_bounds` |`upper_bounds` ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

---

#### Operands: [¶](#operands-34)

[¶](#operands-34)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `bounded_values` | variadic of TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `bounded_values` | variadic of TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `bounded_values` | variadic of TransformHandleTypeInterface instance |
 `bounded_values` |`bounded_values` variadic of TransformHandleTypeInterface instance |

---

### `transform.affine.simplify_min_max_affine_ops` (transform::SimplifyMinMaxAffineOpsOp) [¶](#transformaffinesimplify_min_max_affine_ops-transformsimplifyminmaxaffineopsop)

`transform.affine.simplify_min_max_affine_ops`
[¶](#transformaffinesimplify_min_max_affine_ops-transformsimplifyminmaxaffineopsop)

Syntax:

```
operation ::= `transform.affine.simplify_min_max_affine_ops` $target attr-dict `:` type($target)
```

`` operation ::= `transform.affine.simplify_min_max_affine_ops` $target attr-dict `:` type($target) ``

Simplify the targeted `affine.min` / `affine.max` ops using the
`mlir::affine::simplifyAffineMinMaxOps` transform.

`affine.min`
`affine.max`
`mlir::affine::simplifyAffineMinMaxOps`

Example:

```
%0 = transform.structured.match ops{["affine.max"]} in %arg1
transform.affine.simplify_min_max_affine_ops %0 : !transform.any_op
```

`%0 = transform.structured.match ops{["affine.max"]} in %arg1
transform.affine.simplify_min_max_affine_ops %0 : !transform.any_op`

---

#### Return modes [¶](#return-modes-2)

[¶](#return-modes-2)

This transform consumes the target handle and does not produce any results.
This transforms definitely fails if any of the targeted operations is not an
`affine.min` or `affine.max` operation, or if the canonicalization patterns
failed to converge.
This transform silently fails if none of the operations were simplified.
Otherwise, it succeeds.

`affine.min`
`affine.max`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-35)

[¶](#operands-35)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns.arm_neon.vector_contract_to_bfmmla` (transform::ApplyArmNeonContractionToBFMMLAPatternsOp) [¶](#transformapply_patternsarm_neonvector_contract_to_bfmmla-transformapplyarmneoncontractiontobfmmlapatternsop)

`transform.apply_patterns.arm_neon.vector_contract_to_bfmmla`
[¶](#transformapply_patternsarm_neonvector_contract_to_bfmmla-transformapplyarmneoncontractiontobfmmlapatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.arm_neon.vector_contract_to_bfmmla` attr-dict
```

`` operation ::= `transform.apply_patterns.arm_neon.vector_contract_to_bfmmla` attr-dict ``

Indicates that vector contract operations should be lowered to
to ArmNeon dialect operations mapping to instructions from FEAT\_BF16.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.arm_neon.vector_contract_to_i8mm` (transform::ApplyArmNeonContractionToI8MMPatternsOp) [¶](#transformapply_patternsarm_neonvector_contract_to_i8mm-transformapplyarmneoncontractiontoi8mmpatternsop)

`transform.apply_patterns.arm_neon.vector_contract_to_i8mm`
[¶](#transformapply_patternsarm_neonvector_contract_to_i8mm-transformapplyarmneoncontractiontoi8mmpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.arm_neon.vector_contract_to_i8mm` attr-dict
```

`` operation ::= `transform.apply_patterns.arm_neon.vector_contract_to_i8mm` attr-dict ``

Indicates that vector contract operations should be lowered to
to ArmNeon dialect operations mapping to instructions from FEAT\_I8MM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla` (transform::ApplyArmSVELowerContractionToBFMMLAPatternsOp) [¶](#transformapply_patternsarm_svevector_contract_to_bfmmla-transformapplyarmsvelowercontractiontobfmmlapatternsop)

`transform.apply_patterns.arm_sve.vector_contract_to_bfmmla`
[¶](#transformapply_patternsarm_svevector_contract_to_bfmmla-transformapplyarmsvelowercontractiontobfmmlapatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla` attr-dict
```

`` operation ::= `transform.apply_patterns.arm_sve.vector_contract_to_bfmmla` attr-dict ``

Indicates that vector contract operations should be lowered to
ArmSVE dialect operations mapping to instructions from FEAT\_BF16.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.arm_sve.vector_contract_to_i8mm` (transform::ApplyArmSVELowerContractionToI8MMPatternsOp) [¶](#transformapply_patternsarm_svevector_contract_to_i8mm-transformapplyarmsvelowercontractiontoi8mmpatternsop)

`transform.apply_patterns.arm_sve.vector_contract_to_i8mm`
[¶](#transformapply_patternsarm_svevector_contract_to_i8mm-transformapplyarmsvelowercontractiontoi8mmpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.arm_sve.vector_contract_to_i8mm` attr-dict
```

`` operation ::= `transform.apply_patterns.arm_sve.vector_contract_to_i8mm` attr-dict ``

Indicates that vector contract operations should be lowered to
to ArmSVE dialect operations mapping to instructions from FEAT\_I8MM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.bufferization.buffer_loop_hoisting` (transform::BufferLoopHoistingOp) [¶](#transformbufferizationbuffer_loop_hoisting-transformbufferloophoistingop)

`transform.bufferization.buffer_loop_hoisting`
[¶](#transformbufferizationbuffer_loop_hoisting-transformbufferloophoistingop)

Syntax:

```
operation ::= `transform.bufferization.buffer_loop_hoisting` $target attr-dict `:` type($target)
```

`` operation ::= `transform.bufferization.buffer_loop_hoisting` $target attr-dict `:` type($target) ``

Hoist buffer allocations (“memref.alloc” and “memref.alloca”) from loops
within the targeted op. This transform assumes that there are no buffer
deallocation ops in the IR.

This transform reads the `target` handle and modifies the payload.

`target`

Traits: `TransformEachOpTrait`

`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-36)

[¶](#operands-36)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.bufferization.eliminate_empty_tensors` (transform::EliminateEmptyTensorsOp) [¶](#transformbufferizationeliminate_empty_tensors-transformeliminateemptytensorsop)

`transform.bufferization.eliminate_empty_tensors`
[¶](#transformbufferizationeliminate_empty_tensors-transformeliminateemptytensorsop)

Syntax:

```
operation ::= `transform.bufferization.eliminate_empty_tensors` $target attr-dict `:` type($target)
```

`` operation ::= `transform.bufferization.eliminate_empty_tensors` $target attr-dict `:` type($target) ``

Try to eliminate all `tensor.empty` ops within the targeted op by replacing
them with another destination tensor.

`tensor.empty`

“tensor.empty” ops cannot be bufferized. They can either be converted to
“bufferization.alloc\_tensor” or replaced with another tensor (via this
transform). “tensor.empty” does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transformation looks for subset ops that insert a tensor that
originates from a “tensor.empty” (as per the reverse use-def chain). Such
“tensor.empty” ops are replaced with the destination subset.

Example:

```
%0 = tensor.empty() : tensor<5xf32>
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

`%0 = tensor.empty() : tensor<5xf32>
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]`

Is rewritten with:

```
%0 = tensor.extract_slice %t[1][5][1]
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

`%0 = tensor.extract_slice %t[1][5][1]
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]`

In the above example, the subset op is “tensor.insert\_slice”. When tracing
back the reverse use-def chain of a the source, we end up at a
“tensor.empty” op.

The above example can bufferize without an allocation (in the absence of
other conflicts) because there is no longer a `tensor.empty` op.

`tensor.empty`

See `-eliminate-empty-tensors` for more details.

`-eliminate-empty-tensors`

---

#### Return modes [¶](#return-modes-3)

[¶](#return-modes-3)

This transform reads the target handle and modifies the payload. It does
not produce any handle.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-37)

[¶](#operands-37)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.bufferization.empty_tensor_to_alloc_tensor` (transform::EmptyTensorToAllocTensorOp) [¶](#transformbufferizationempty_tensor_to_alloc_tensor-transformemptytensortoalloctensorop)

`transform.bufferization.empty_tensor_to_alloc_tensor`
[¶](#transformbufferizationempty_tensor_to_alloc_tensor-transformemptytensortoalloctensorop)

Syntax:

```
operation ::= `transform.bufferization.empty_tensor_to_alloc_tensor` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.bufferization.empty_tensor_to_alloc_tensor` $target attr-dict `:` functional-type(operands, results) ``

Replace a tensor.empty with a bufferization.tensor\_alloc.

---

#### Return modes [¶](#return-modes-4)

[¶](#return-modes-4)

This operation consumes the `target` handle and produces the `transformed`
handle. `target` is expected to be a `tensor.empty` operation. The transform
always succeeds.

`target`
`transformed`
`target`
`tensor.empty`

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-38)

[¶](#operands-38)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to tensor.empty operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to tensor.empty operations |
| `target` | Transform IR handle to tensor.empty operations |
 `target` |`target` Transform IR handle to tensor.empty operations |

---

#### Results: [¶](#results-24)

[¶](#results-24)

| Result | Description |
| --- | --- |
| `transformed` | Transform IR handle to bufferization.alloc\_tensor operations |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | Transform IR handle to bufferization.alloc\_tensor operations |
| `transformed` | Transform IR handle to bufferization.alloc\_tensor operations |
 `transformed` |`transformed` Transform IR handle to bufferization.alloc\_tensor operations |

---

### `transform.bufferization.one_shot_bufferize` (transform::OneShotBufferizeOp) [¶](#transformbufferizationone_shot_bufferize-transformoneshotbufferizeop)

`transform.bufferization.one_shot_bufferize`
[¶](#transformbufferizationone_shot_bufferize-transformoneshotbufferizeop)

Syntax:

```
operation ::= `transform.bufferization.one_shot_bufferize` (`layout` `{` $function_boundary_type_conversion^ `}`)?
              $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.bufferization.one_shot_bufferize` (`layout` `{` $function_boundary_type_conversion^ `}`)?
$target attr-dict `:` functional-type($target, results) ``

Indicates that the given `target` op should be bufferized with One-Shot
Bufferize. The bufferization can be configured with various attributes that
corresponding to options in `BufferizationOptions` and the
`one-shot-bufferize` pass. More information can be found in the pass
documentation.

`target`
`BufferizationOptions`
`one-shot-bufferize`

The targeted ops must be modules or functions. This is because there is
always a single, bufferized replacement op for such targets.

Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
unset, this transform fails when an unknown/non-bufferizable op is found.
Many ops implement `BufferizableOpInterface` via an external model. These
external models must be registered when applying this transform op;
otherwise, said ops would be considered non-bufferizable.

`BufferizableOpInterface`
`allow_unknown_ops`
`allow_unknown_ops`
`BufferizableOpInterface`

---

#### Return modes [¶](#return-modes-5)

[¶](#return-modes-5)

This operation consumes the `target` handle and produces the `transformed`
handle.

`target`
`transformed`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-27)

[¶](#attributes-27)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_boundary_type_conversion` | ::mlir::bufferization::LayoutMapOptionAttr | option for map layout |
| `allow_return_allocs_from_loops` | ::mlir::BoolAttr | bool attribute |
| `allow_unknown_ops` | ::mlir::BoolAttr | bool attribute |
| `bufferize_function_boundaries` | ::mlir::BoolAttr | bool attribute |
| `dump_alias_sets` | ::mlir::BoolAttr | bool attribute |
| `test_analysis_only` | ::mlir::BoolAttr | bool attribute |
| `print_conflicts` | ::mlir::BoolAttr | bool attribute |
| `check_parallel_regions` | ::mlir::BoolAttr | bool attribute |
| `memcpy_op` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `function_boundary_type_conversion` | ::mlir::bufferization::LayoutMapOptionAttr | option for map layout |
 `function_boundary_type_conversion` |`function_boundary_type_conversion` ::mlir::bufferization::LayoutMapOptionAttr | option for map layout || `allow_return_allocs_from_loops` | ::mlir::BoolAttr | bool attribute |
 `allow_return_allocs_from_loops` |`allow_return_allocs_from_loops` ::mlir::BoolAttr | bool attribute || `allow_unknown_ops` | ::mlir::BoolAttr | bool attribute |
 `allow_unknown_ops` |`allow_unknown_ops` ::mlir::BoolAttr | bool attribute || `bufferize_function_boundaries` | ::mlir::BoolAttr | bool attribute |
 `bufferize_function_boundaries` |`bufferize_function_boundaries` ::mlir::BoolAttr | bool attribute || `dump_alias_sets` | ::mlir::BoolAttr | bool attribute |
 `dump_alias_sets` |`dump_alias_sets` ::mlir::BoolAttr | bool attribute || `test_analysis_only` | ::mlir::BoolAttr | bool attribute |
 `test_analysis_only` |`test_analysis_only` ::mlir::BoolAttr | bool attribute || `print_conflicts` | ::mlir::BoolAttr | bool attribute |
 `print_conflicts` |`print_conflicts` ::mlir::BoolAttr | bool attribute || `check_parallel_regions` | ::mlir::BoolAttr | bool attribute |
 `check_parallel_regions` |`check_parallel_regions` ::mlir::BoolAttr | bool attribute || `memcpy_op` | ::mlir::StringAttr | string attribute |
 `memcpy_op` |`memcpy_op` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-39)

[¶](#operands-39)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-25)

[¶](#results-25)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.debug.emit_param_as_remark` (transform::EmitParamAsRemarkOp) [¶](#transformdebugemit_param_as_remark-transformemitparamasremarkop)

`transform.debug.emit_param_as_remark`
[¶](#transformdebugemit_param_as_remark-transformemitparamasremarkop)

*Prints the parameter as a diagnostic remark*

*Prints the parameter as a diagnostic remark*

Syntax:

```
operation ::= `transform.debug.emit_param_as_remark` $param (`,` $message^)?  (`at` $anchor^)?attr-dict `:` type($param) (`,` type($anchor)^)?
```

`` operation ::= `transform.debug.emit_param_as_remark` $param (`,` $message^)? (`at` $anchor^)?attr-dict `:` type($param) (`,` type($anchor)^)? ``

This operation emits a diagnostic remark containing the string form of the
attributes associated with the parameter provided as attribute. It takes
as optional arguments:

* an additional message text to prepend;
* a handle pointing to operations the location of which will be used to
  emit the diagnostic; if multiple operations are associated, the
  diagnostic is emitted for all of their respective locations.

- an additional message text to prepend;
- a handle pointing to operations the location of which will be used to
  emit the diagnostic; if multiple operations are associated, the
  diagnostic is emitted for all of their respective locations.

This operation always succeeds.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-28)

[¶](#attributes-28)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `message` | ::mlir::StringAttr | string attribute |
 `message` |`message` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-40)

[¶](#operands-40)

| Operand | Description |
| --- | --- |
| `param` | TransformParamTypeInterface instance |
| `anchor` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `param` | TransformParamTypeInterface instance |
| `anchor` | TransformHandleTypeInterface instance |
| `param` | TransformParamTypeInterface instance |
 `param` |`param` TransformParamTypeInterface instance || `anchor` | TransformHandleTypeInterface instance |
 `anchor` |`anchor` TransformHandleTypeInterface instance |

---

### `transform.debug.emit_remark_at` (transform::EmitRemarkAtOp) [¶](#transformdebugemit_remark_at-transformemitremarkatop)

`transform.debug.emit_remark_at`
[¶](#transformdebugemit_remark_at-transformemitremarkatop)

*Print a message as diagnostic remark attached to payload*

*Print a message as diagnostic remark attached to payload*

Syntax:

```
operation ::= `transform.debug.emit_remark_at` $at `,` $message attr-dict `:` type($at)
```

`` operation ::= `transform.debug.emit_remark_at` $at `,` $message attr-dict `:` type($at) ``

This operation emits a diagnostic remark with the given message at the
location of each payload object associated with the argument. The argument
may be an operation or a value handle.

This operation always succeeds.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-29)

[¶](#attributes-29)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `message` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `message` | ::mlir::StringAttr | string attribute |
 `message` |`message` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-41)

[¶](#operands-41)

| Operand | Description |
| --- | --- |
| `at` | any transform handle |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `at` | any transform handle |
| `at` | any transform handle |
 `at` |`at` any transform handle |

---

### `transform.dlti.query` (transform::QueryOp) [¶](#transformdltiquery-transformqueryop)

`transform.dlti.query`
[¶](#transformdltiquery-transformqueryop)

*Return attribute (as param) associated to key via DTLI*

*Return attribute (as param) associated to key via DTLI*

Syntax:

```
operation ::= `transform.dlti.query` $keys `at` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.dlti.query` $keys `at` $target attr-dict `:` functional-type(operands, results) ``

This op queries data layout and target information associated to payload
IR by way of the DLTI dialect.

A lookup is performed for the given `keys` at `target` op - or its closest
interface-implementing ancestor - by way of the `DLTIQueryInterface`, which
returns an attribute for a key. Each key should be either a (quoted) string
or a type. If more than one key is provided, the lookup continues
recursively, now on the returned attributes, with the condition that these
implement the above interface. For example if the payload IR is

`keys`
`target`
`DLTIQueryInterface`

```
module attributes {#dlti.map = #dlti.map<#dlti.dl_entry<"A",
                                 #dlti.map<#dlti.dl_entry<"B", 42: int>>>} {
  func.func private @f()
}
```

`module attributes {#dlti.map = #dlti.map<#dlti.dl_entry<"A",
#dlti.map<#dlti.dl_entry<"B", 42: int>>>} {
func.func private @f()
}`

and we have that `%func` is a Tranform handle to op `@f`, then
`transform.dlti.query ["A", "B"] at %func` returns 42 as a param and
`transform.dlti.query ["A"] at %func` returns the `#dlti.map` attribute
containing just the key “B” and its value. Using `["B"]` or `["A","C"]` as
`keys` will yield an error.

`%func`
`@f`
`transform.dlti.query ["A", "B"] at %func`
`transform.dlti.query ["A"] at %func`
`#dlti.map`
`["B"]`
`["A","C"]`
`keys`

---

#### Return modes [¶](#return-modes-6)

[¶](#return-modes-6)

When successful, the result, `associated_attr`, associates one attribute as
a param for each op in `target`’s payload.

`associated_attr`
`target`

If the lookup fails - as no DLTI attributes/interfaces are found or entries
with the right names are missing - a silenceable failure is returned.

Traits: `TransformEachOpTrait`

`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-30)

[¶](#attributes-30)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `keys` | ::mlir::ArrayAttr | array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `keys` | ::mlir::ArrayAttr | array attribute |
 `keys` |`keys` ::mlir::ArrayAttr | array attribute |

---

#### Operands: [¶](#operands-42)

[¶](#operands-42)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-26)

[¶](#results-26)

| Result | Description |
| --- | --- |
| `associated_attr` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `associated_attr` | TransformParamTypeInterface instance |
| `associated_attr` | TransformParamTypeInterface instance |
 `associated_attr` |`associated_attr` TransformParamTypeInterface instance |

---

### `transform.irdl.collect_matching` (transform::IRDLCollectMatchingOp) [¶](#transformirdlcollect_matching-transformirdlcollectmatchingop)

`transform.irdl.collect_matching`
[¶](#transformirdlcollect_matching-transformirdlcollectmatchingop)

*Finds ops that match the IRDL definition without registering them.*

*Finds ops that match the IRDL definition without registering them.*

Syntax:

```
operation ::= `transform.irdl.collect_matching` `in` $root `:` functional-type(operands, results) attr-dict-with-keyword regions
```

`` operation ::= `transform.irdl.collect_matching` `in` $root `:` functional-type(operands, results) attr-dict-with-keyword regions ``

Traits: `NoTerminator`, `SymbolTable`

`NoTerminator`
`SymbolTable`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-43)

[¶](#operands-43)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-27)

[¶](#results-27)

| Result | Description |
| --- | --- |
| `matched` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `matched` | TransformHandleTypeInterface instance |
| `matched` | TransformHandleTypeInterface instance |
 `matched` |`matched` TransformHandleTypeInterface instance |

---

### `transform.apply_conversion_patterns.func.func_to_llvm` (transform::ApplyFuncToLLVMConversionPatternsOp) [¶](#transformapply_conversion_patternsfuncfunc_to_llvm-transformapplyfunctollvmconversionpatternsop)

`transform.apply_conversion_patterns.func.func_to_llvm`
[¶](#transformapply_conversion_patternsfuncfunc_to_llvm-transformapplyfunctollvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.func.func_to_llvm` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.func.func_to_llvm` attr-dict ``

Collects patterns that convert Func dialect ops to LLVM dialect ops.
These patterns require an “LLVMTypeConverter”.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

### `transform.func.cast_and_call` (transform::CastAndCallOp) [¶](#transformfunccast_and_call-transformcastandcallop)

`transform.func.cast_and_call`
[¶](#transformfunccast_and_call-transformcastandcallop)

*Casts values to the signature of a function and replaces them with a call*

*Casts values to the signature of a function and replaces them with a call*

Syntax:

```
operation ::= `transform.func.cast_and_call` ($function_name^)? ($function^)?
              ( `(` $inputs^ `)` )?
              ( `->` $outputs^ )?
              (`after` $insert_after^):(`before`)? $insertion_point
              ($conversions^)? attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.func.cast_and_call` ($function_name^)? ($function^)?
( `(` $inputs^ `)` )?
( `->` $outputs^ )?
(`after` $insert_after^):(`before`)? $insertion_point
($conversions^)? attr-dict `:` functional-type(operands, results) ``

This transform takes value handles to a set of `inputs` and `outputs` and
attempts to cast them to the function signature of the attached function
op, then builds a call to the function and replaces the users of the
outputs. It is the responsibility of the user to ensure that the slice of
the program replaced by this operation makes sense, i.e. there is no
verification that the inputs to this operation have any relation to the
outputs outside of basic dominance requirements needed for the call.

`inputs`
`outputs`

The casting materialization functions are specified in the graph region of
this op. They must implement the `TypeConverterBuilderOpInterface`. The
order of ops within the region is irrelevant.

`TypeConverterBuilderOpInterface`

The target function can be specified by a symbol name or by a handle to the
operation.

This transform only reads the operand handles and only replaces the users of
the outputs with the results of the call. No handles are consumed and no
operations are removed. Users are expected to run cleanup separately if
desired.

Warning: The replacement of the uses of the outputs could invalidate certain
restricted value handle types (e.g. `transform.block_arg` if it existed, by
replacing the use with something not coming from a block argument). The
value will still exist in such cases but wouldn’t verify against the type.
See the discussion here for more information:
<https://github.com/llvm/llvm-project/pull/78398#discussion_r1455070087>

`transform.block_arg`
<https://github.com/llvm/llvm-project/pull/78398#discussion_r1455070087>

This transform will emit a silenceable failure if:

* The set of outputs isn’t unique
* The handle for the insertion point does not include exactly one operation
* The insertion point op does not dominate any of the output users
* The insertion point op is not dominated by any of the inputs
* The function signature does not match the number of inputs/outputs

- The set of outputs isn’t unique
- The handle for the insertion point does not include exactly one operation
- The insertion point op does not dominate any of the output users
- The insertion point op is not dominated by any of the inputs
- The function signature does not match the number of inputs/outputs

This transform will emit a definite failure if it fails to resolve the
target function, or if it fails to materialize the conversion casts of
either the inputs to the function argument types, or the call results to
the output types.

Traits: `AttrSizedOperandSegments`, `HasOnlyGraphRegion`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

`AttrSizedOperandSegments`
`HasOnlyGraphRegion`
`NoTerminator`
`ReportTrackingListenerFailuresOpTrait`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionKindInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-31)

[¶](#attributes-31)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `insert_after` | ::mlir::UnitAttr | unit attribute |
| `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `insert_after` | ::mlir::UnitAttr | unit attribute |
 `insert_after` |`insert_after` ::mlir::UnitAttr | unit attribute || `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `function_name` |`function_name` ::mlir::SymbolRefAttr | symbol reference attribute |

---

#### Operands: [¶](#operands-44)

[¶](#operands-44)

| Operand | Description |
| --- | --- |
| `insertion_point` | TransformHandleTypeInterface instance |
| `inputs` | TransformValueHandleTypeInterface instance |
| `outputs` | TransformValueHandleTypeInterface instance |
| `function` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `insertion_point` | TransformHandleTypeInterface instance |
| `inputs` | TransformValueHandleTypeInterface instance |
| `outputs` | TransformValueHandleTypeInterface instance |
| `function` | TransformHandleTypeInterface instance |
| `insertion_point` | TransformHandleTypeInterface instance |
 `insertion_point` |`insertion_point` TransformHandleTypeInterface instance || `inputs` | TransformValueHandleTypeInterface instance |
 `inputs` |`inputs` TransformValueHandleTypeInterface instance || `outputs` | TransformValueHandleTypeInterface instance |
 `outputs` |`outputs` TransformValueHandleTypeInterface instance || `function` | TransformHandleTypeInterface instance |
 `function` |`function` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-28)

[¶](#results-28)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.func.deduplicate_func_args` (transform::DeduplicateFuncArgsOp) [¶](#transformfuncdeduplicate_func_args-transformdeduplicatefuncargsop)

`transform.func.deduplicate_func_args`
[¶](#transformfuncdeduplicate_func_args-transformdeduplicatefuncargsop)

Syntax:

```
operation ::= `transform.func.deduplicate_func_args` $function_name
              `at` $module attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.func.deduplicate_func_args` $function_name
`at` $module attr-dict `:` functional-type(operands, results) ``

This transform takes a module and a function name, and deduplicates
the arguments of the function. The function is expected to be defined in
the module.

This transform will emit a silenceable failure if:

* The function with the given name does not exist in the module.
* The function does not have duplicate arguments.
* The function does not have a single call.

- The function with the given name does not exist in the module.
- The function does not have duplicate arguments.
- The function does not have a single call.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-32)

[¶](#attributes-32)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `function_name` |`function_name` ::mlir::SymbolRefAttr | symbol reference attribute |

---

#### Operands: [¶](#operands-45)

[¶](#operands-45)

| Operand | Description |
| --- | --- |
| `module` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `module` | TransformHandleTypeInterface instance |
| `module` | TransformHandleTypeInterface instance |
 `module` |`module` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-29)

[¶](#results-29)

| Result | Description |
| --- | --- |
| `transformed_module` | TransformHandleTypeInterface instance |
| `transformed_function` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed_module` | TransformHandleTypeInterface instance |
| `transformed_function` | TransformHandleTypeInterface instance |
| `transformed_module` | TransformHandleTypeInterface instance |
 `transformed_module` |`transformed_module` TransformHandleTypeInterface instance || `transformed_function` | TransformHandleTypeInterface instance |
 `transformed_function` |`transformed_function` TransformHandleTypeInterface instance |

---

### `transform.func.replace_func_signature` (transform::ReplaceFuncSignatureOp) [¶](#transformfuncreplace_func_signature-transformreplacefuncsignatureop)

`transform.func.replace_func_signature`
[¶](#transformfuncreplace_func_signature-transformreplacefuncsignatureop)

Syntax:

```
operation ::= `transform.func.replace_func_signature` $function_name
              `args_interchange` `=` $args_interchange
              `results_interchange` `=` $results_interchange
              `at` $module attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.func.replace_func_signature` $function_name
`args_interchange` `=` $args_interchange
`results_interchange` `=` $results_interchange
`at` $module attr-dict `:` functional-type(operands, results) ``

This transform takes a module and a function name, and replaces the
signature of the function by reordering the arguments and results
according to the interchange arrays. The function is expected to be
defined in the module, and the interchange arrays must match the number
of arguments and results of the function.

The `adjust_func_calls` attribute indicates whether the function calls
should be adjusted to match the new signature. If set to `true`, the
function calls will be adjusted to match the new signature, otherwise
they will not be adjusted.

`adjust_func_calls`
`true`

This transform will emit a silenceable failure if:

* The function with the given name does not exist in the module.
* The interchange arrays do not match the number of arguments/results.
* The interchange arrays contain out of bound indices.

- The function with the given name does not exist in the module.
- The interchange arrays do not match the number of arguments/results.
- The interchange arrays contain out of bound indices.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-33)

[¶](#attributes-33)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |
| `args_interchange` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
| `results_interchange` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
| `adjust_func_calls` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `function_name` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `function_name` |`function_name` ::mlir::SymbolRefAttr | symbol reference attribute || `args_interchange` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
 `args_interchange` |`args_interchange` ::mlir::DenseI32ArrayAttr | i32 dense array attribute || `results_interchange` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
 `results_interchange` |`results_interchange` ::mlir::DenseI32ArrayAttr | i32 dense array attribute || `adjust_func_calls` | ::mlir::UnitAttr | unit attribute |
 `adjust_func_calls` |`adjust_func_calls` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-46)

[¶](#operands-46)

| Operand | Description |
| --- | --- |
| `module` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `module` | TransformHandleTypeInterface instance |
| `module` | TransformHandleTypeInterface instance |
 `module` |`module` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-30)

[¶](#results-30)

| Result | Description |
| --- | --- |
| `transformed_module` | TransformHandleTypeInterface instance |
| `transformed_function` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed_module` | TransformHandleTypeInterface instance |
| `transformed_function` | TransformHandleTypeInterface instance |
| `transformed_module` | TransformHandleTypeInterface instance |
 `transformed_module` |`transformed_module` TransformHandleTypeInterface instance || `transformed_function` | TransformHandleTypeInterface instance |
 `transformed_function` |`transformed_function` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` (transform::ApplyGPUPromoteShuffleToAMDGPUPatternsOp) [¶](#transformapply_patternsgpugpu_shuffle_to_amdgpu-transformapplygpupromoteshuffletoamdgpupatternsop)

`transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu`
[¶](#transformapply_patternsgpugpu_shuffle_to_amdgpu-transformapplygpupromoteshuffletoamdgpupatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` (`chipset` `=` $chipset^)? attr-dict
```

`` operation ::= `transform.apply_patterns.gpu.gpu_shuffle_to_amdgpu` (`chipset` `=` $chipset^)? attr-dict ``

Collects patterns that are tryin to promote `gpu.shuffle`s to specialized
AMDGPU intrinsics.

`gpu.shuffle`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-34)

[¶](#attributes-34)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `chipset` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `chipset` | ::mlir::StringAttr | string attribute |
 `chipset` |`chipset` ::mlir::StringAttr | string attribute |

---

### `transform.apply_patterns.gpu.gpu_rewrite_patterns` (transform::ApplyGPURewritePatternsOp) [¶](#transformapply_patternsgpugpu_rewrite_patterns-transformapplygpurewritepatternsop)

`transform.apply_patterns.gpu.gpu_rewrite_patterns`
[¶](#transformapply_patternsgpugpu_rewrite_patterns-transformapplygpurewritepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.gpu_rewrite_patterns` attr-dict
```

`` operation ::= `transform.apply_patterns.gpu.gpu_rewrite_patterns` attr-dict ``

Collects GPU rewrite patterns comprising:

1. GpuAllReduceRewrite patterns
2. GpuGlobalIdRewriter patterns
3. GpuShuffleRewriter patterns

- GpuAllReduceRewrite patterns
- GpuGlobalIdRewriter patterns
- GpuShuffleRewriter patterns

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` (transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp) [¶](#transformapply_conversion_patternsgpugpu_subgroup_reduce_to_nvvm-transformapplygpusubgroupreducetonvvmconversionpatternsop)

`transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm`
[¶](#transformapply_conversion_patternsgpugpu_subgroup_reduce_to_nvvm-transformapplygpusubgroupreducetonvvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm` attr-dict ``

Collects patterns that convert GPU dialect ops related to wmma ops
to NVVM dialect ops.
These patterns require an “LLVMTypeConverter”.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

### `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` (transform::ApplyGPUToNVVMConversionPatternsOp) [¶](#transformapply_conversion_patternsgpugpu_to_nvvm-transformapplygputonvvmconversionpatternsop)

`transform.apply_conversion_patterns.gpu.gpu_to_nvvm`
[¶](#transformapply_conversion_patternsgpugpu_to_nvvm-transformapplygputonvvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_nvvm` attr-dict ``

Collects patterns that convert GPU dialect ops to NVVM dialect ops. These
patterns require an “LLVMTypeConverter”.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-35)

[¶](#attributes-35)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `benefit` | ::mlir::IntegerAttr | 16-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `benefit` | ::mlir::IntegerAttr | 16-bit signless integer attribute |
 `benefit` |`benefit` ::mlir::IntegerAttr | 16-bit signless integer attribute |

---

### `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` (transform::ApplyGPUToROCDLConversionPatternsOp) [¶](#transformapply_conversion_patternsgpugpu_to_rocdl-transformapplygputorocdlconversionpatternsop)

`transform.apply_conversion_patterns.gpu.gpu_to_rocdl`
[¶](#transformapply_conversion_patternsgpugpu_to_rocdl-transformapplygputorocdlconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` `chipset` `=` $chipset attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.gpu.gpu_to_rocdl` `chipset` `=` $chipset attr-dict ``

Collects patterns that convert GPU dialect ops to ROCDL dialect ops. These
patterns require an “LLVMTypeConverter”.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-36)

[¶](#attributes-36)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `chipset` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `chipset` | ::mlir::StringAttr | string attribute |
 `chipset` |`chipset` ::mlir::StringAttr | string attribute |

---

### `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` (transform::ApplyGPUWwmaToNVVMConversionPatternsOp) [¶](#transformapply_conversion_patternsgpugpu_wmma_to_nvvm-transformapplygpuwwmatonvvmconversionpatternsop)

`transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm`
[¶](#transformapply_conversion_patternsgpugpu_wmma_to_nvvm-transformapplygpuwwmatonvvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm` attr-dict ``

Collects patterns that convert GPU dialect ops related to wmma ops
to NVVM dialect ops.
These patterns require an “LLVMTypeConverter”.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

### `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` (transform::ApplyUnrollVectorsSubgroupMmaOp) [¶](#transformapply_patternsgpuunroll_vectors_subgroup_mma-transformapplyunrollvectorssubgroupmmaop)

`transform.apply_patterns.gpu.unroll_vectors_subgroup_mma`
[¶](#transformapply_patternsgpuunroll_vectors_subgroup_mma-transformapplyunrollvectorssubgroupmmaop)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` `[` $m `,` $n `,` $k `]` attr-dict
```

`` operation ::= `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma` `[` $m `,` $n `,` $k `]` attr-dict ``

Unrolls contractions to the target `m`, `n`, and `k` native vector size,
along with other vector operations based on expected usage. `transfer_read`
ops unroll based on the extract slice shape introduced by unrolling the
contractions, while elementwise and `transfer_write` ops unroll to the shape of
the C matrix (`m x n`).

`m`
`n`
`k`
`transfer_read`
`transfer_write`
`m x n`

This operation applies to pure vector operations and should be applied before
lowering to subgroup\_mma ops.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-37)

[¶](#attributes-37)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `m` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `n` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `k` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `m` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `m` |`m` ::mlir::IntegerAttr | 64-bit signless integer attribute || `n` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `n` |`n` ::mlir::IntegerAttr | 64-bit signless integer attribute || `k` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `k` |`k` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

### `transform.apply_patterns.gpu.eliminate_barriers` (transform::EliminateBarriersOp) [¶](#transformapply_patternsgpueliminate_barriers-transformeliminatebarriersop)

`transform.apply_patterns.gpu.eliminate_barriers`
[¶](#transformapply_patternsgpueliminate_barriers-transformeliminatebarriersop)

Syntax:

```
operation ::= `transform.apply_patterns.gpu.eliminate_barriers` attr-dict
```

`` operation ::= `transform.apply_patterns.gpu.eliminate_barriers` attr-dict ``

Removes unnecessary GPU barriers from the function. If a barrier does not
enforce any conflicting pair of memory effects, including a pair that is
enforced by another barrier, it is unnecessary and can be removed.

The approach is based on “High-Performance GPU-to-CPU Transpilation and
Optimization via High-Level Parallel Constructs” by Moses, Ivanov,
Domke, Endo, Doerfert, and Zinenko in PPoPP 2023. Specifically, it
analyzes the memory effects of the operations before and after the given
barrier and checks if the barrier enforces any of the memory
effect-induced dependencies that aren’t already enforced by another
barrier.

For example, in the following code

```
  store %A
  barrier  // enforces load-after-store
  load %A
  barrier  // load-after-store already enforced by the previous barrier
  load %A
```

```
  store %A
  barrier  // enforces load-after-store
  load %A
  barrier  // load-after-store already enforced by the previous barrier
  load %A
```

 `store %A
 barrier // enforces load-after-store
 load %A
 barrier // load-after-store already enforced by the previous barrier
 load %A`
 store %A
 store %A
%A
 barrier // enforces load-after-store
 barrier // enforces load-after-store
// enforces load-after-store
 load %A
 load %A

%A
 barrier // load-after-store already enforced by the previous barrier
 barrier // load-after-store already enforced by the previous barrier
// load-after-store already enforced by the previous barrier
 load %A
 load %A

%A

the second barrier can be removed.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.gpu.map_forall_to_blocks` (transform::MapForallToBlocks) [¶](#transformgpumap_forall_to_blocks-transformmapforalltoblocks)

`transform.gpu.map_forall_to_blocks`
[¶](#transformgpumap_forall_to_blocks-transformmapforalltoblocks)

Syntax:

```
operation ::= `transform.gpu.map_forall_to_blocks` $target
              (`generate_gpu_launch` $generate_gpu_launch^)?
              (`grid_dims` `=` $grid_dims^)?
              attr-dict
              `:` functional-type($target, $result)
```

`` operation ::= `transform.gpu.map_forall_to_blocks` $target
(`generate_gpu_launch` $generate_gpu_launch^)?
(`grid_dims` `=` $grid_dims^)?
attr-dict
`:` functional-type($target, $result) ``

Target the gpu\_launch op and rewrite the top level `scf.forall`
to distributed gpu.block\_id attribute. If `generate_gpu_launch` attribute
is set, then first generates `gpu_launch` and moves the top level
`scf.forall` inside.

`scf.forall`
`generate_gpu_launch`
`gpu_launch`
`scf.forall`

The operation searches top level `scf.forall` ops under
`gpu_launch` and maps each such op to GPU blocks. Mapping is
one-to-one and the induction variables of `scf.forall` are
rewritten to gpu.block\_id according to the `thread_dim_mapping` attribute.

`scf.forall`
`gpu_launch`
`scf.forall`
`thread_dim_mapping`

Dynamic, `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

`scf.forall`

Only **bufferized** scf.forall are currently supported.
Only scf.forall distributed to **at most 3 dimensions** are
currently supported.

**bufferized**
**at most 3 dimensions**

The operation alters the block size of the given gpu\_launch using the
grid\_dims argument.

---

#### Return modes: [¶](#return-modes-7)

[¶](#return-modes-7)

This operation ignores non-gpu\_launch ops and drops them in the return.

If any scf.forall with tensors is found, the transform definitely
fails.

If all the `scf.forall` operations contained within the LaunchOp
referred to by the `target` handle lower to GPU properly, the
transform succeeds. Otherwise the transform definitely fails.

`scf.forall`
`target`

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-38)

[¶](#attributes-38)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `grid_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `generate_gpu_launch` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `grid_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `grid_dims` |`grid_dims` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `generate_gpu_launch` | ::mlir::UnitAttr | unit attribute |
 `generate_gpu_launch` |`generate_gpu_launch` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-47)

[¶](#operands-47)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-31)

[¶](#results-31)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.gpu.map_nested_forall_to_threads` (transform::MapNestedForallToThreads) [¶](#transformgpumap_nested_forall_to_threads-transformmapnestedforalltothreads)

`transform.gpu.map_nested_forall_to_threads`
[¶](#transformgpumap_nested_forall_to_threads-transformmapnestedforalltothreads)

Syntax:

```
operation ::= `transform.gpu.map_nested_forall_to_threads` $target
              `block_dims` `=` $block_dims
              (`sync_after_distribute` `=` $sync_after_distribute^)?
              (`warp_size` `=` $warp_size^)?
              attr-dict
              `:` functional-type($target, $result)
```

`` operation ::= `transform.gpu.map_nested_forall_to_threads` $target
`block_dims` `=` $block_dims
(`sync_after_distribute` `=` $sync_after_distribute^)?
(`warp_size` `=` $warp_size^)?
attr-dict
`:` functional-type($target, $result) ``

Target the `gpu.launch op` and rewrite all `scf.forall` nested in it to
distributed `gpu.thread_id` attribute.

`gpu.launch op`
`scf.forall`
`gpu.thread_id`

The operation searches for `scf.forall` ops nested under `target` and maps
each such op to GPU threads.

`scf.forall`
`target`

`scf.forall` induction variables are rewritten to `gpu.thread_id` according
to the `mapping` attribute.

`scf.forall`
`gpu.thread_id`
`mapping`

Different types of mappings attributes are supported:

* the block\_dims is a list of integers that specifies the number of
  threads in each dimension. This is a mandatory attribute that is used
  to constrain the number of threads in each dimension. If an
  `scf.forall` op is mapped to fewer threads, predication occurs.
* the warp\_dims is a list of integers that specifies the number of
  warps in each dimension. This is an optional attribute that is used
  to constrain the number of warps in each dimension. When present, this
  attribute must be specified in a way that is compatible with the
  block\_dims attribute. If an `scf.forall` op is mapped to fewer warps,
  predication occurs.

- the block\_dims is a list of integers that specifies the number of
  threads in each dimension. This is a mandatory attribute that is used
  to constrain the number of threads in each dimension. If an
  `scf.forall` op is mapped to fewer threads, predication occurs.
`scf.forall`- the warp\_dims is a list of integers that specifies the number of
  warps in each dimension. This is an optional attribute that is used
  to constrain the number of warps in each dimension. When present, this
  attribute must be specified in a way that is compatible with the
  block\_dims attribute. If an `scf.forall` op is mapped to fewer warps,
  predication occurs.
`scf.forall`

Dynamic `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

`scf.forall`

Only **bufferized** `scf.forall` are currently supported.
Only `scf.forall` distributed to **at most 3 dimensions** are
currently supported.

**bufferized**
`scf.forall`
`scf.forall`
**at most 3 dimensions**

The `sync_after_distribute`attribute controls whether a `gpu.barrier` is
inserted after each scf.forall op. At this time, this is an all or nothing
choice. This will need to be tightened in the future.

`sync_after_distribute`
`gpu.barrier`

The operation alters the block size of the given gpu\_launch using the
mandatory block\_dims argument.

---

#### Return modes: [¶](#return-modes-8)

[¶](#return-modes-8)

This operation ignores non-`gpu_launch` ops and drops them in the return.

`gpu_launch`

If any scf.forall with tensors is found, the transform definitely
fails.

If all the `scf.forall` operations with gpu.thread mapping contained
within the `LaunchOp` referred to by the `target` handle lower to GPU
properly, the transform succeeds. Otherwise the transform definitely
fails.

`scf.forall`
`LaunchOp`
`target`

scf.forall operations with mappings other than gpu.thread are
ignored.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

---

#### Example: [¶](#example)

[¶](#example)

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

`gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
threads(%tx, %ty, %tz) in (%tx = %3, %ty = %4, %tz = %5) {
scf.forall (%i, %j) in (7, 9) {
... // body 1
} {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
scf.forall (%i) in (12) {
... // body 2
} {mapping = [#gpu.thread<x>]}
gpu.terminator
}`

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

`%bdimX = arith.constant 12 : index
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
}`

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-39)

[¶](#attributes-39)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `block_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `sync_after_distribute` | ::mlir::BoolAttr | bool attribute |
| `warp_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `block_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `block_dims` |`block_dims` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `sync_after_distribute` | ::mlir::BoolAttr | bool attribute |
 `sync_after_distribute` |`sync_after_distribute` ::mlir::BoolAttr | bool attribute || `warp_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `warp_size` |`warp_size` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-48)

[¶](#operands-48)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-32)

[¶](#results-32)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.loop.hoist_loop_invariant_subsets` (transform::HoistLoopInvariantSubsetsOp) [¶](#transformloophoist_loop_invariant_subsets-transformhoistloopinvariantsubsetsop)

`transform.loop.hoist_loop_invariant_subsets`
[¶](#transformloophoist_loop_invariant_subsets-transformhoistloopinvariantsubsetsop)

*Hoist loop invariant subset ops*

*Hoist loop invariant subset ops*

Syntax:

```
operation ::= `transform.loop.hoist_loop_invariant_subsets` $target attr-dict `:` type($target)
```

`` operation ::= `transform.loop.hoist_loop_invariant_subsets` $target attr-dict `:` type($target) ``

This transform hoists loop-invariant subset ops out of the targeted
loop-like op. It looks for matching subset extraction/insertion op pairs and
hoists them. The loop body operates on a newly introduced region iter\_arg.

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

`%r = scf.for ... iter_args(%t = %a) -> (tensor<?xf32>) {
%0 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
%1 = "test.foo"(%0) : (tensor<5xf32>) -> (tensor<5xf32>)
%2 = tensor.insert_slice %1 into %t[0][5][1]
: tensor<5xf32> into tensor<?xf32>
scf.yield %2 : tensor<?xf32>
}`

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

`%0 = tensor.extract_slice %a[0][5][1] : tensor<?xf32> to tensor<5xf32>
%new_loop:2 = scf.for ... iter_args(%t = %a, %h = %0) -> (tensor<?xf32>) {
%1 = "test.foo"(%h) : (tensor<5xf32>) -> (tensor<5xf32>)
scf.yield %t, %2 : tensor<?xf32>, tensor<5xf32>
}
%r = tensor.insert_slice %new_loop#1 into %new_loop#0
: tensor<5xf32> into tensor<?xf32>`

Subset ops are hoisted only if there are no conflicting subset ops. E.g.,
if there were a second overlapping extraction in the above example, no ops
could be hoisted safely.

This transform reads the target handle and modifies the payload. This
transform does not invalidate any handles, but loop-like ops are replaced
with new loop-like ops when a subset op is hoisted. The transform rewriter
updates all handles accordingly.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-49)

[¶](#operands-49)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns.scf.for_loop_canonicalization` (transform::ApplyForLoopCanonicalizationPatternsOp) [¶](#transformapply_patternsscffor_loop_canonicalization-transformapplyforloopcanonicalizationpatternsop)

`transform.apply_patterns.scf.for_loop_canonicalization`
[¶](#transformapply_patternsscffor_loop_canonicalization-transformapplyforloopcanonicalizationpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.scf.for_loop_canonicalization` attr-dict
```

`` operation ::= `transform.apply_patterns.scf.for_loop_canonicalization` attr-dict ``

Collects patterns for canonicalizing operations inside SCF loop bodies.
At the moment, only affine.min/max computations with iteration variables,
loop bounds and loop steps are canonicalized.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_conversion_patterns.scf.structural_conversions` (transform::ApplySCFStructuralConversionPatternsOp) [¶](#transformapply_conversion_patternsscfstructural_conversions-transformapplyscfstructuralconversionpatternsop)

`transform.apply_conversion_patterns.scf.structural_conversions`
[¶](#transformapply_conversion_patternsscfstructural_conversions-transformapplyscfstructuralconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.scf.structural_conversions` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.scf.structural_conversions` attr-dict ``

Collects patterns for performing structural conversions of SCF operations.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

### `transform.apply_conversion_patterns.scf.scf_to_control_flow` (transform::ApplySCFToControlFlowPatternsOp) [¶](#transformapply_conversion_patternsscfscf_to_control_flow-transformapplyscftocontrolflowpatternsop)

`transform.apply_conversion_patterns.scf.scf_to_control_flow`
[¶](#transformapply_conversion_patternsscfscf_to_control_flow-transformapplyscftocontrolflowpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.scf.scf_to_control_flow` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.scf.scf_to_control_flow` attr-dict ``

Collects patterns that lower structured control flow ops to unstructured
control flow.

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

### `transform.loop.forall_to_for` (transform::ForallToForOp) [¶](#transformloopforall_to_for-transformforalltoforop)

`transform.loop.forall_to_for`
[¶](#transformloopforall_to_for-transformforalltoforop)

*Converts scf.forall into a nest of scf.for operations*

*Converts scf.forall into a nest of scf.for operations*

Syntax:

```
operation ::= `transform.loop.forall_to_for` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.forall_to_for` $target attr-dict `:` functional-type(operands, results) ``

Converts the `scf.forall` operation pointed to by the given handle into a
set of nested `scf.for` operations. Each new operation corresponds to one
induction variable of the original “multifor” loop.

`scf.forall`
`scf.for`

The operand handle must be associated with exactly one payload operation.

Loops with shared outputs are currently not supported.

---

#### Return Modes [¶](#return-modes-9)

[¶](#return-modes-9)

Consumes the operand handle. Produces a silenceable failure if the operand
is not associated with a single `scf.forall` payload operation.
Returns as many handles as the given `forall` op has induction variables
that are associated with the generated `scf.for` loops.
Produces a silenceable failure if another number of resulting handles is
requested.

`scf.forall`
`forall`
`scf.for`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-50)

[¶](#operands-50)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-33)

[¶](#results-33)

| Result | Description |
| --- | --- |
| `transformed` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | variadic of TransformHandleTypeInterface instance |
| `transformed` | variadic of TransformHandleTypeInterface instance |
 `transformed` |`transformed` variadic of TransformHandleTypeInterface instance |

---

### `transform.loop.forall_to_parallel` (transform::ForallToParallelOp) [¶](#transformloopforall_to_parallel-transformforalltoparallelop)

`transform.loop.forall_to_parallel`
[¶](#transformloopforall_to_parallel-transformforalltoparallelop)

*Converts scf.forall into a nest of scf.for operations*

*Converts scf.forall into a nest of scf.for operations*

Syntax:

```
operation ::= `transform.loop.forall_to_parallel` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.forall_to_parallel` $target attr-dict `:` functional-type(operands, results) ``

Converts the `scf.forall` operation pointed to by the given handle into an
`scf.parallel` operation.

`scf.forall`
`scf.parallel`

The operand handle must be associated with exactly one payload operation.

Loops with outputs are not supported.

---

#### Return Modes [¶](#return-modes-10)

[¶](#return-modes-10)

Consumes the operand handle. Produces a silenceable failure if the operand
is not associated with a single `scf.forall` payload operation.
Returns a handle to the new `scf.parallel` operation.
Produces a silenceable failure if another number of resulting handles is
requested.

`scf.forall`
`scf.parallel`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-51)

[¶](#operands-51)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-34)

[¶](#results-34)

| Result | Description |
| --- | --- |
| `transformed` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | variadic of TransformHandleTypeInterface instance |
| `transformed` | variadic of TransformHandleTypeInterface instance |
 `transformed` |`transformed` variadic of TransformHandleTypeInterface instance |

---

### `transform.loop.coalesce` (transform::LoopCoalesceOp) [¶](#transformloopcoalesce-transformloopcoalesceop)

`transform.loop.coalesce`
[¶](#transformloopcoalesce-transformloopcoalesceop)

*Coalesces the perfect loop nest enclosed by a given loop*

*Coalesces the perfect loop nest enclosed by a given loop*

Syntax:

```
operation ::= `transform.loop.coalesce` $target attr-dict `:` functional-type($target, $transformed)
```

`` operation ::= `transform.loop.coalesce` $target attr-dict `:` functional-type($target, $transformed) ``

Given a perfect loop nest identified by the outermost loop,
perform loop coalescing in a bottom-up one-by-one manner.

---

#### Return modes [¶](#return-modes-11)

[¶](#return-modes-11)

The return handle points to the coalesced loop if coalescing happens, or
the given input loop if coalescing does not happen.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-52)

[¶](#operands-52)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-35)

[¶](#results-35)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.loop.fuse_sibling` (transform::LoopFuseSiblingOp) [¶](#transformloopfuse_sibling-transformloopfusesiblingop)

`transform.loop.fuse_sibling`
[¶](#transformloopfuse_sibling-transformloopfusesiblingop)

*Fuse a loop into another loop, assuming the fusion is legal.*

*Fuse a loop into another loop, assuming the fusion is legal.*

Syntax:

```
operation ::= `transform.loop.fuse_sibling` $target `into` $source attr-dict  `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.fuse_sibling` $target `into` $source attr-dict `:` functional-type(operands, results) ``

Fuses the `target` loop into the `source` loop assuming they are
independent of each other. In the fused loop, the arguments, body and
results of `target` are placed *before* those of `source`.

`target`
`source`
`target`
*before*
`source`

For fusion of two `scf.for` loops, the bounds and step size must match. For
fusion of two `scf.forall` loops, the bounds and the mapping must match.
Otherwise a silencable failure is produced.

`scf.for`
`scf.forall`

The `target` and `source` handles must refer to exactly one operation,
otherwise a definite failure is produced. It is the responsibility of the
user to ensure that the `target` and `source` loops are independent of each
other – this op will only perform rudimentary legality checks.

`target`
`source`
`target`
`source`

---

#### Return modes [¶](#return-modes-12)

[¶](#return-modes-12)

This operation consumes the `target` and `source` handles and produces the
`fused_loop` handle, which points to the fused loop.

`target`
`source`
`fused_loop`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-53)

[¶](#operands-53)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `source` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `source` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `source` | TransformHandleTypeInterface instance |
 `source` |`source` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-36)

[¶](#results-36)

| Result | Description |
| --- | --- |
| `fused_loop` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `fused_loop` | TransformHandleTypeInterface instance |
| `fused_loop` | TransformHandleTypeInterface instance |
 `fused_loop` |`fused_loop` TransformHandleTypeInterface instance |

---

### `transform.loop.outline` (transform::LoopOutlineOp) [¶](#transformloopoutline-transformloopoutlineop)

`transform.loop.outline`
[¶](#transformloopoutline-transformloopoutlineop)

*Outlines a loop into a named function*

*Outlines a loop into a named function*

Syntax:

```
operation ::= `transform.loop.outline` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.outline` $target attr-dict `:` functional-type(operands, results) ``

Moves the loop into a separate function with the specified name and replaces
the loop in the Payload IR with a call to that function. Takes care of
forwarding values that are used in the loop as function arguments. If the
operand is associated with more than one loop, each loop will be outlined
into a separate function. The provided name is used as a *base* for forming
actual function names following `SymbolTable` auto-renaming scheme to avoid
duplicate symbols. Expects that all ops in the Payload IR have a
`SymbolTable` ancestor (typically true because of the top-level module).

*base*
`SymbolTable`
`SymbolTable`

---

#### Return Modes [¶](#return-modes-13)

[¶](#return-modes-13)

Returns a handle to the list of outlined functions and a handle to the
corresponding function call operations in the same order as the operand
handle.

Produces a definite failure if outlining failed for any of the targets.

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-40)

[¶](#attributes-40)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `func_name` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `func_name` | ::mlir::StringAttr | string attribute |
 `func_name` |`func_name` ::mlir::StringAttr | string attribute |

---

#### Operands: [¶](#operands-54)

[¶](#operands-54)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-37)

[¶](#results-37)

| Result | Description |
| --- | --- |
| `function` | TransformHandleTypeInterface instance |
| `call` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `function` | TransformHandleTypeInterface instance |
| `call` | TransformHandleTypeInterface instance |
| `function` | TransformHandleTypeInterface instance |
 `function` |`function` TransformHandleTypeInterface instance || `call` | TransformHandleTypeInterface instance |
 `call` |`call` TransformHandleTypeInterface instance |

---

### `transform.loop.peel` (transform::LoopPeelOp) [¶](#transformlooppeel-transformlooppeelop)

`transform.loop.peel`
[¶](#transformlooppeel-transformlooppeelop)

*Peels the first or last iteration of the loop*

*Peels the first or last iteration of the loop*

Syntax:

```
operation ::= `transform.loop.peel` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.peel` $target attr-dict `:` functional-type(operands, results) ``

Rewrite the given loop with a main loop and a partial (first or last) loop.
When the `peelFront` option is set to true, the first iteration is peeled off.
Otherwise, updates the given loop so that its step evenly divides its range and puts
the remaining iteration into a separate loop or a conditional.

`peelFront`

In the absence of sufficient static information, this op may peel a loop,
even if the step always divides the range evenly at runtime.

---

#### Return modes [¶](#return-modes-14)

[¶](#return-modes-14)

This operation ignores non-scf::ForOp ops and drops them in the return.
The op returns two loops, the peeled loop which has trip count divisible
by the step, and the remainder loop.

When `peelFront` is true, the first result (remainder loop) executes all
but the first iteration of the target loop. The second result (peeled
loop) corresponds to the first iteration of the loop which can be
canonicalized away in the following optimizations.

`peelFront`

When `peelFront` is false, the first result (peeled loop) is the portion
of the target loop with the highest upper bound that is divisible by the
step. The second result (remainder loop) contains the remaining iterations.

`peelFront`

Note that even though the Payload IR modification may be performed
in-place, this operation consumes the operand handle and produces a new one.

---

#### Return Modes [¶](#return-modes-15)

[¶](#return-modes-15)

Produces a definite failure if peeling fails.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-41)

[¶](#attributes-41)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `peel_front` | ::mlir::BoolAttr | bool attribute |
| `fail_if_already_divisible` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `peel_front` | ::mlir::BoolAttr | bool attribute |
 `peel_front` |`peel_front` ::mlir::BoolAttr | bool attribute || `fail_if_already_divisible` | ::mlir::BoolAttr | bool attribute |
 `fail_if_already_divisible` |`fail_if_already_divisible` ::mlir::BoolAttr | bool attribute |

---

#### Operands: [¶](#operands-55)

[¶](#operands-55)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to scf.for operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to scf.for operations |
| `target` | Transform IR handle to scf.for operations |
 `target` |`target` Transform IR handle to scf.for operations |

---

#### Results: [¶](#results-38)

[¶](#results-38)

| Result | Description |
| --- | --- |
| `peeled_loop` | TransformHandleTypeInterface instance |
| `remainder_loop` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `peeled_loop` | TransformHandleTypeInterface instance |
| `remainder_loop` | TransformHandleTypeInterface instance |
| `peeled_loop` | TransformHandleTypeInterface instance |
 `peeled_loop` |`peeled_loop` TransformHandleTypeInterface instance || `remainder_loop` | TransformHandleTypeInterface instance |
 `remainder_loop` |`remainder_loop` TransformHandleTypeInterface instance |

---

### `transform.loop.pipeline` (transform::LoopPipelineOp) [¶](#transformlooppipeline-transformlooppipelineop)

`transform.loop.pipeline`
[¶](#transformlooppipeline-transformlooppipelineop)

*Applies software pipelining to the loop*

*Applies software pipelining to the loop*

Syntax:

```
operation ::= `transform.loop.pipeline` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.pipeline` $target attr-dict `:` functional-type(operands, results) ``

Transforms the given loops one by one to achieve software pipelining for
each of them. That is, performs some amount of reads from memory before the
loop rather than inside the loop, the same amount of writes into memory
after the loop, and updates each iteration to read the data for a following
iteration rather than the current one.

The amount is specified by the attributes.

The values read and about to be stored are transferred as loop iteration
arguments. Currently supports memref and vector transfer operations as
memory reads/writes.

---

#### Return modes [¶](#return-modes-16)

[¶](#return-modes-16)

This operation ignores non-scf::For ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation pipeline
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure. The return handle points to only the subset of
successfully produced pipelined loops, which can be empty.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-42)

[¶](#attributes-42)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `iteration_interval` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `read_latency` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `iteration_interval` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `iteration_interval` |`iteration_interval` ::mlir::IntegerAttr | 64-bit signless integer attribute || `read_latency` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `read_latency` |`read_latency` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-56)

[¶](#operands-56)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to scf.for operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to scf.for operations |
| `target` | Transform IR handle to scf.for operations |
 `target` |`target` Transform IR handle to scf.for operations |

---

#### Results: [¶](#results-39)

[¶](#results-39)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.loop.promote_if_one_iteration` (transform::LoopPromoteIfOneIterationOp) [¶](#transformlooppromote_if_one_iteration-transformlooppromoteifoneiterationop)

`transform.loop.promote_if_one_iteration`
[¶](#transformlooppromote_if_one_iteration-transformlooppromoteifoneiterationop)

*Promote loop if it has one iteration*

*Promote loop if it has one iteration*

Syntax:

```
operation ::= `transform.loop.promote_if_one_iteration` $target attr-dict `:` type($target)
```

`` operation ::= `transform.loop.promote_if_one_iteration` $target attr-dict `:` type($target) ``

Promotes the given target loop op if it has a single iteration. I.e., the
loop op is removed and only the body remains.

---

#### Return modes [¶](#return-modes-17)

[¶](#return-modes-17)

This transform fails if the target is mapped to ops that are loops. Ops are
considered loops if they implement the `LoopLikeOpInterface`. Otherwise,
this transform always succeeds. The transform consumes the target handle and
modifies the payload.

`LoopLikeOpInterface`

Traits: `TransformEachOpTrait`

`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-57)

[¶](#operands-57)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.loop.unroll_and_jam` (transform::LoopUnrollAndJamOp) [¶](#transformloopunroll_and_jam-transformloopunrollandjamop)

`transform.loop.unroll_and_jam`
[¶](#transformloopunroll_and_jam-transformloopunrollandjamop)

*Unrolls and jam the given loop with the given unroll factor*

*Unrolls and jam the given loop with the given unroll factor*

Syntax:

```
operation ::= `transform.loop.unroll_and_jam` $target attr-dict `:` type($target)
```

`` operation ::= `transform.loop.unroll_and_jam` $target attr-dict `:` type($target) ``

Unrolls & jams each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

---

#### Return modes [¶](#return-modes-18)

[¶](#return-modes-18)

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

`scf.for`
`affine.for`
`target`

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-43)

[¶](#attributes-43)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |
 `factor` |`factor` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

---

#### Operands: [¶](#operands-58)

[¶](#operands-58)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.loop.unroll` (transform::LoopUnrollOp) [¶](#transformloopunroll-transformloopunrollop)

`transform.loop.unroll`
[¶](#transformloopunroll-transformloopunrollop)

*Unrolls the given loop with the given unroll factor*

*Unrolls the given loop with the given unroll factor*

Syntax:

```
operation ::= `transform.loop.unroll` $target attr-dict `:` type($target)
```

`` operation ::= `transform.loop.unroll` $target attr-dict `:` type($target) ``

Unrolls each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

---

#### Return modes [¶](#return-modes-19)

[¶](#return-modes-19)

This operation ignores non-`scf.for`, non-`affine.for` ops and drops them
in the return. If all the operations referred to by the `target` operand
unroll properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

`scf.for`
`affine.for`
`target`

Does not return handles as the operation may result in the loop being
removed after a full unrolling.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-44)

[¶](#attributes-44)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |
 `factor` |`factor` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |

---

#### Operands: [¶](#operands-59)

[¶](#operands-59)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.loop.parallel_for_to_nested_fors` (transform::ParallelForToNestedForOps) [¶](#transformloopparallel_for_to_nested_fors-transformparallelfortonestedforops)

`transform.loop.parallel_for_to_nested_fors`
[¶](#transformloopparallel_for_to_nested_fors-transformparallelfortonestedforops)

*Converts scf.parallel into a nest of scf.for operations*

*Converts scf.parallel into a nest of scf.for operations*

Syntax:

```
operation ::= `transform.loop.parallel_for_to_nested_fors` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.loop.parallel_for_to_nested_fors` $target attr-dict `:` functional-type(operands, results) ``

Converts the `scf.parallel` operation pointed to by the given handle into a
set of nested `scf.for` operations. Each new operation corresponds to one
dimension of the original parallel loop.

`scf.parallel`
`scf.for`

The operand handle must be associated with exactly one payload operation.

Loops with shared outputs are currently not supported.

---

#### Return Modes [¶](#return-modes-20)

[¶](#return-modes-20)

Consumes the operand handle. Produces a silenceable failure if the operand
is not associated with a single `scf.parallel` payload operation.
Returns as many handles as the given `parallel` op has dimensions that are
associated with the generated `scf.for` loops.
Produces a silenceable failure if another number of resulting handles is
requested.

`scf.parallel`
`parallel`
`scf.for`

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-60)

[¶](#operands-60)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-40)

[¶](#results-40)

| Result | Description |
| --- | --- |
| `transformed` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | variadic of TransformHandleTypeInterface instance |
| `transformed` | variadic of TransformHandleTypeInterface instance |
 `transformed` |`transformed` variadic of TransformHandleTypeInterface instance |

---

### `transform.scf.take_assumed_branch` (transform::TakeAssumedBranchOp) [¶](#transformscftake_assumed_branch-transformtakeassumedbranchop)

`transform.scf.take_assumed_branch`
[¶](#transformscftake_assumed_branch-transformtakeassumedbranchop)

Syntax:

```
operation ::= `transform.scf.take_assumed_branch` $target
              (`take_else_branch` $take_else_branch^)?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.scf.take_assumed_branch` $target
(`take_else_branch` $take_else_branch^)?
attr-dict
`:` functional-type(operands, results) ``

Given an scf.if conditional, inject user-defined information that it is
always safe to execute only the if or else branch.

This is achieved by just replacing the scf.if by the content of one of its
branches.

This is particularly useful for user-controlled rewriting of conditionals
that exist solely to guard against out-of-bounds behavior.

At the moment, no assume or assert operation is emitted as it is not always
desirable. In the future, this may be controlled by a dedicated attribute.

---

#### Return modes [¶](#return-modes-21)

[¶](#return-modes-21)

The transform only consumes its operand and does not produce any result.
The transform definitely fails if `take_else_branch` is specified and the
`else` region is empty.

`take_else_branch`
`else`

Traits: `TransformEachOpTrait`

`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-45)

[¶](#attributes-45)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `take_else_branch` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `take_else_branch` | ::mlir::UnitAttr | unit attribute |
 `take_else_branch` |`take_else_branch` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-61)

[¶](#operands-61)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns.memref.alloc_to_alloca` (transform::ApplyAllocToAllocaOp) [¶](#transformapply_patternsmemrefalloc_to_alloca-transformapplyalloctoallocaop)

`transform.apply_patterns.memref.alloc_to_alloca`
[¶](#transformapply_patternsmemrefalloc_to_alloca-transformapplyalloctoallocaop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.alloc_to_alloca` (`size_limit` `(` $size_limit^ `)`)? attr-dict
```

`` operation ::= `transform.apply_patterns.memref.alloc_to_alloca` (`size_limit` `(` $size_limit^ `)`)? attr-dict ``

Collects patterns to rewrite scoped dynamic allocation (`alloc`/`dealloc`
pairs) into automatic allocation (`alloca`) in the same scope, for memrefs
of static shape.

`alloc`
`dealloc`
`alloca`

The `size_limit` attribute controls the maximum allocated memory (in bytes,
subject to data layout) for which the pattern applies.

`size_limit`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-46)

[¶](#attributes-46)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `size_limit` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `size_limit` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `size_limit` |`size_limit` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

### `transform.apply_patterns.memref.expand_ops` (transform::ApplyExpandOpsPatternsOp) [¶](#transformapply_patternsmemrefexpand_ops-transformapplyexpandopspatternsop)

`transform.apply_patterns.memref.expand_ops`
[¶](#transformapply_patternsmemrefexpand_ops-transformapplyexpandopspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.expand_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.memref.expand_ops` attr-dict ``

Collects patterns to rewrite ops within the memref dialect.

* Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
  AtomicRMWOpLowering pattern, e.g. with “minf” or “maxf” attributes, to
  `memref.generic_atomic_rmw` with the expanded code.
* Converts `memref.reshape` that has a target shape of a statically-known
  size to `memref.reinterpret_cast`.

- Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
  AtomicRMWOpLowering pattern, e.g. with “minf” or “maxf” attributes, to
  `memref.generic_atomic_rmw` with the expanded code.
`atomic_rmw`
`memref.generic_atomic_rmw`- Converts `memref.reshape` that has a target shape of a statically-known
  size to `memref.reinterpret_cast`.
`memref.reshape`
`memref.reinterpret_cast`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.memref.expand_strided_metadata` (transform::ApplyExpandStridedMetadataPatternsOp) [¶](#transformapply_patternsmemrefexpand_strided_metadata-transformapplyexpandstridedmetadatapatternsop)

`transform.apply_patterns.memref.expand_strided_metadata`
[¶](#transformapply_patternsmemrefexpand_strided_metadata-transformapplyexpandstridedmetadatapatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.expand_strided_metadata` attr-dict
```

`` operation ::= `transform.apply_patterns.memref.expand_strided_metadata` attr-dict ``

Collects patterns for expanding memref operations that modify the metadata
(sizes, offset, strides) of a memref into easier to analyze constructs.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.memref.extract_address_computations` (transform::ApplyExtractAddressComputationsPatternsOp) [¶](#transformapply_patternsmemrefextract_address_computations-transformapplyextractaddresscomputationspatternsop)

`transform.apply_patterns.memref.extract_address_computations`
[¶](#transformapply_patternsmemrefextract_address_computations-transformapplyextractaddresscomputationspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.extract_address_computations` attr-dict
```

`` operation ::= `transform.apply_patterns.memref.extract_address_computations` attr-dict ``

Collects patterns for extracting address computations from operations
with memory accesses such that these memory accesses use only a base
pointer.

For instance,

```
memref.load %base[%off0, ...]
```

```
memref.load %base[%off0, ...]
```

`memref.load %base[%off0, ...]`
memref.load %base[%off0, ...]
memref.load %base[%off0, ...]
memref
.
%base
[
%off0
,
...]

Will be rewritten in:

```
%new_base = memref.subview %base[%off0,...][1,...][1,...]
memref.load %new_base[%c0,...]
```

```
%new_base = memref.subview %base[%off0,...][1,...][1,...]
memref.load %new_base[%c0,...]
```

`%new_base = memref.subview %base[%off0,...][1,...][1,...]
memref.load %new_base[%c0,...]`
%new\_base = memref.subview %base[%off0,...][1,...][1,...]
%new\_base = memref.subview %base[%off0,...][1,...][1,...]
%new\_base
=
memref
.
%base
[
%off0
,...][
1
,...][
1
,...]
memref.load %new\_base[%c0,...]
memref.load %new\_base[%c0,...]
memref
.
%new\_base
[
%c0
,...]

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.memref.fold_memref_alias_ops` (transform::ApplyFoldMemrefAliasOpsPatternsOp) [¶](#transformapply_patternsmemreffold_memref_alias_ops-transformapplyfoldmemrefaliasopspatternsop)

`transform.apply_patterns.memref.fold_memref_alias_ops`
[¶](#transformapply_patternsmemreffold_memref_alias_ops-transformapplyfoldmemrefaliasopspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.fold_memref_alias_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.memref.fold_memref_alias_ops` attr-dict ``

Collects patterns for folding memref aliasing ops (memref.subview) into
consumer load/store ops (affine.load, memref.load, nvgpu.ldmatrix,
vector.load, vector.transfer\_read, affine.store, memref.store, etc.) and
other ops (e.g., memref.subview).

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims` (transform::ApplyResolveRankedShapedTypeResultDimsPatternsOp) [¶](#transformapply_patternsmemrefresolve_ranked_shaped_type_result_dims-transformapplyresolverankedshapedtyperesultdimspatternsop)

`transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims`
[¶](#transformapply_patternsmemrefresolve_ranked_shaped_type_result_dims-transformapplyresolverankedshapedtyperesultdimspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims` attr-dict
```

`` operation ::= `transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims` attr-dict ``

Collects patterns that resolve `memref.dim` operations with values that are
defined by operations that implement the `ReifyRankedShapedTypeOpInterface`,
in terms of shapes of its input operands.

`memref.dim`
`ReifyRankedShapedTypeOpInterface`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.memref.alloca_to_global` (transform::MemRefAllocaToGlobalOp) [¶](#transformmemrefalloca_to_global-transformmemrefallocatoglobalop)

`transform.memref.alloca_to_global`
[¶](#transformmemrefalloca_to_global-transformmemrefallocatoglobalop)

Syntax:

```
operation ::= `transform.memref.alloca_to_global` $alloca attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.memref.alloca_to_global` $alloca attr-dict `:` functional-type(operands, results) ``

Inserts a new `memref.global` for each provided `memref.alloca` into the
nearest symbol table (e.g., a `builtin.module`) and replaces it with a
`memref.get_global`. This is useful, for example, for allocations that
should reside in the shared memory of a GPU, which have to be declared as
globals.

`memref.global`
`memref.alloca`
`builtin.module`
`memref.get_global`

---

#### Example [¶](#example-1)

[¶](#example-1)

Consider the following transform op:

```
%get_global, %global =
    transform.memref.alloca_to_global %alloca
      : (!transform.op<"memref.alloca">)
        -> (!transform.any_op, !transform.any_op)
```

```
%get_global, %global =
    transform.memref.alloca_to_global %alloca
      : (!transform.op<"memref.alloca">)
        -> (!transform.any_op, !transform.any_op)
```

`%get_global, %global =
 transform.memref.alloca_to_global %alloca
 : (!transform.op<"memref.alloca">)
 -> (!transform.any_op, !transform.any_op)`
%get\_global, %global =
%get\_global, %global =
%get\_global
,
%global
=
 transform.memref.alloca\_to\_global %alloca
 transform.memref.alloca\_to\_global %alloca
.
memref
.
%alloca
 : (!transform.op<"memref.alloca">)
 : (!transform.op<"memref.alloca">)
:
(!
.
<
"memref.alloca"
>)
 -> (!transform.any\_op, !transform.any\_op)
 -> (!transform.any\_op, !transform.any\_op)
->
(!
.
,
!
.
)

and the following input payload:

```
module {
  func.func @func() {
    %alloca = memref.alloca() : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

```
module {
  func.func @func() {
    %alloca = memref.alloca() : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

`module {
 func.func @func() {
 %alloca = memref.alloca() : memref<2x32xf32>
 // usages of %alloca...
 }
}`
module {
module {
{
 func.func @func() {
 func.func @func() {
func
.
func
@func
()
{
 %alloca = memref.alloca() : memref<2x32xf32>
 %alloca = memref.alloca() : memref<2x32xf32>
%alloca
=
memref
.
()
:
memref
<
2x32x
f32
>
 // usages of %alloca...
 // usages of %alloca...
// usages of %alloca...
 }
 }

}
}
}
}

then applying the transform op to the payload would result in the following
output IR:

```
module {
  memref.global "private" @alloc : memref<2x32xf32>
  func.func @func() {
    %alloca = memref.get_global @alloc : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

```
module {
  memref.global "private" @alloc : memref<2x32xf32>
  func.func @func() {
    %alloca = memref.get_global @alloc : memref<2x32xf32>
    // usages of %alloca...
  }
}
```

`module {
 memref.global "private" @alloc : memref<2x32xf32>
 func.func @func() {
 %alloca = memref.get_global @alloc : memref<2x32xf32>
 // usages of %alloca...
 }
}`
module {
module {
{
 memref.global "private" @alloc : memref<2x32xf32>
 memref.global "private" @alloc : memref<2x32xf32>
memref
.
"private"
@alloc
:
memref
<
2x32x
f32
>
 func.func @func() {
 func.func @func() {
func
.
func
@func
()
{
 %alloca = memref.get\_global @alloc : memref<2x32xf32>
 %alloca = memref.get\_global @alloc : memref<2x32xf32>
%alloca
=
memref
.
@alloc
:
memref
<
2x32x
f32
>
 // usages of %alloca...
 // usages of %alloca...
// usages of %alloca...
 }
 }

}
}
}
}

---

#### Return modes [¶](#return-modes-22)

[¶](#return-modes-22)

Succeeds always. The returned handles refer to the `memref.get_global` and
`memref.global` ops that were inserted by the transformation.

`memref.get_global`
`memref.global`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-62)

[¶](#operands-62)

| Operand | Description |
| --- | --- |
| `alloca` | Transform IR handle to memref.alloca operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `alloca` | Transform IR handle to memref.alloca operations |
| `alloca` | Transform IR handle to memref.alloca operations |
 `alloca` |`alloca` Transform IR handle to memref.alloca operations |

---

#### Results: [¶](#results-41)

[¶](#results-41)

| Result | Description |
| --- | --- |
| `getGlobal` | TransformHandleTypeInterface instance |
| `global` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `getGlobal` | TransformHandleTypeInterface instance |
| `global` | TransformHandleTypeInterface instance |
| `getGlobal` | TransformHandleTypeInterface instance |
 `getGlobal` |`getGlobal` TransformHandleTypeInterface instance || `global` | TransformHandleTypeInterface instance |
 `global` |`global` TransformHandleTypeInterface instance |

---

### `transform.memref.erase_dead_alloc_and_stores` (transform::MemRefEraseDeadAllocAndStoresOp) [¶](#transformmemreferase_dead_alloc_and_stores-transformmemreferasedeadallocandstoresop)

`transform.memref.erase_dead_alloc_and_stores`
[¶](#transformmemreferase_dead_alloc_and_stores-transformmemreferasedeadallocandstoresop)

Syntax:

```
operation ::= `transform.memref.erase_dead_alloc_and_stores` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.memref.erase_dead_alloc_and_stores` $target attr-dict `:` functional-type($target, results) ``

This applies memory optimization on memref. In particular it does store to
load forwarding, dead store elimination and dead alloc/alloca elimination.

---

#### Return modes [¶](#return-modes-23)

[¶](#return-modes-23)

This operation applies a set of memory optimization on the whole region of
the operand.

The transformation does not consume the target handle. It modifies the
payload. Dead allocations, loads and stores are silently dropped from all
mappings.

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-63)

[¶](#operands-63)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.memref.make_loop_independent` (transform::MemRefMakeLoopIndependentOp) [¶](#transformmemrefmake_loop_independent-transformmemrefmakeloopindependentop)

`transform.memref.make_loop_independent`
[¶](#transformmemrefmake_loop_independent-transformmemrefmakeloopindependentop)

Syntax:

```
operation ::= `transform.memref.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed)
```

`` operation ::= `transform.memref.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed) ``

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`memref.subview` inside the loop.

`num_loop`
`scf.for`
`num_loop`
`memref.subview`

Currently supported operations are:

* memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
  followed by a memref.subview.

- memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
  followed by a memref.subview.

---

#### Return modes [¶](#return-modes-24)

[¶](#return-modes-24)

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:

* memref.alloca: The returned handle points to the memref.subview op.

- memref.alloca: The returned handle points to the memref.subview op.

This transform op consumes the target handle and produces a result handle.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-47)

[¶](#attributes-47)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `num_loops` |`num_loops` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-64)

[¶](#operands-64)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-42)

[¶](#results-42)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.memref.multibuffer` (transform::MemRefMultiBufferOp) [¶](#transformmemrefmultibuffer-transformmemrefmultibufferop)

`transform.memref.multibuffer`
[¶](#transformmemrefmultibuffer-transformmemrefmultibufferop)

*Multibuffers an allocation*

*Multibuffers an allocation*

Syntax:

```
operation ::= `transform.memref.multibuffer` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.memref.multibuffer` $target attr-dict `:` functional-type(operands, results) ``

Transformation to do multi-buffering/array expansion to remove
dependencies on the temporary allocation between consecutive loop
iterations. This transform expands the size of an allocation by
a given multiplicative factor and fixes up any users of the
multibuffered allocation.
If skip analysis is not set the transformation will only apply
if it can prove that there is no data being carried across loop
iterations.

---

#### Return modes [¶](#return-modes-25)

[¶](#return-modes-25)

This operation returns the new allocation if multi-buffering
succeeds, and failure otherwise.

Traits: `FunctionalStyleTransformOpTrait`

`FunctionalStyleTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-48)

[¶](#attributes-48)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |
| `skip_analysis` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive |
 `factor` |`factor` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is positive || `skip_analysis` | ::mlir::UnitAttr | unit attribute |
 `skip_analysis` |`skip_analysis` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-65)

[¶](#operands-65)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to memref.alloc operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to memref.alloc operations |
| `target` | Transform IR handle to memref.alloc operations |
 `target` |`target` Transform IR handle to memref.alloc operations |

---

#### Results: [¶](#results-43)

[¶](#results-43)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter` (transform::MemrefToLLVMTypeConverterOp) [¶](#transformapply_conversion_patternsmemrefmemref_to_llvm_type_converter-transformmemreftollvmtypeconverterop)

`transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter`
[¶](#transformapply_conversion_patternsmemrefmemref_to_llvm_type_converter-transformmemreftollvmtypeconverterop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter` attr-dict ``

This operation provides an “LLVMTypeConverter” that lowers memref types to
LLVM types.

The type converter can be customized as follows:

* `use_aligned_alloc`: Use aligned\_alloc in place of malloc for heap
  allocations.
* `index_bitwidth`: Bitwidth of the index type, “0” indicates the size of a
  machine word.
* `use_generic_functions`: Use generic allocation and deallocation functions
  instead of the classic “malloc”, “aligned\_alloc” and “free” functions.
  // TODO: the following two options don’t really make sense for
  // memref\_to\_llvm\_type\_converter specifically.
  // We should have a single to\_llvm\_type\_converter.
* `use_bare_ptr_call_conv`: Replace FuncOp’s MemRef arguments with bare
  pointers to the MemRef element types.
* `data-layout`: String description (LLVM format) of the data layout that is
  expected on the produced module.

- `use_aligned_alloc`: Use aligned\_alloc in place of malloc for heap
  allocations.
`use_aligned_alloc`- `index_bitwidth`: Bitwidth of the index type, “0” indicates the size of a
  machine word.
`index_bitwidth`- `use_generic_functions`: Use generic allocation and deallocation functions
  instead of the classic “malloc”, “aligned\_alloc” and “free” functions.
  // TODO: the following two options don’t really make sense for
  // memref\_to\_llvm\_type\_converter specifically.
  // We should have a single to\_llvm\_type\_converter.
`use_generic_functions`- `use_bare_ptr_call_conv`: Replace FuncOp’s MemRef arguments with bare
  pointers to the MemRef element types.
`use_bare_ptr_call_conv`- `data-layout`: String description (LLVM format) of the data layout that is
  expected on the produced module.
`data-layout`

Interfaces: `TypeConverterBuilderOpInterface`

`TypeConverterBuilderOpInterface`

---

#### Attributes: [¶](#attributes-49)

[¶](#attributes-49)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `use_aligned_alloc` | ::mlir::BoolAttr | bool attribute |
| `index_bitwidth` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `use_generic_functions` | ::mlir::BoolAttr | bool attribute |
| `use_bare_ptr_call_conv` | ::mlir::BoolAttr | bool attribute |
| `data_layout` | ::mlir::StringAttr | string attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `use_aligned_alloc` | ::mlir::BoolAttr | bool attribute |
 `use_aligned_alloc` |`use_aligned_alloc` ::mlir::BoolAttr | bool attribute || `index_bitwidth` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `index_bitwidth` |`index_bitwidth` ::mlir::IntegerAttr | 64-bit signless integer attribute || `use_generic_functions` | ::mlir::BoolAttr | bool attribute |
 `use_generic_functions` |`use_generic_functions` ::mlir::BoolAttr | bool attribute || `use_bare_ptr_call_conv` | ::mlir::BoolAttr | bool attribute |
 `use_bare_ptr_call_conv` |`use_bare_ptr_call_conv` ::mlir::BoolAttr | bool attribute || `data_layout` | ::mlir::StringAttr | string attribute |
 `data_layout` |`data_layout` ::mlir::StringAttr | string attribute |

---

### `transform.pdl_match` (transform::PDLMatchOp) [¶](#transformpdl_match-transformpdlmatchop)

`transform.pdl_match`
[¶](#transformpdl_match-transformpdlmatchop)

*Finds ops that match the named PDL pattern*

*Finds ops that match the named PDL pattern*

Syntax:

```
operation ::= `transform.pdl_match` $pattern_name `in` $root attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.pdl_match` $pattern_name `in` $root attr-dict `:` functional-type(operands, results) ``

Find Payload IR ops nested within the Payload IR op associated with the
operand that match the PDL pattern identified by its name. The pattern is
expected to be defined in the closest surrounding `WithPDLPatternsOp`.

`WithPDLPatternsOp`

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

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-50)

[¶](#attributes-50)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `pattern_name` | ::mlir::SymbolRefAttr | symbol reference attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `pattern_name` | ::mlir::SymbolRefAttr | symbol reference attribute |
 `pattern_name` |`pattern_name` ::mlir::SymbolRefAttr | symbol reference attribute |

---

#### Operands: [¶](#operands-66)

[¶](#operands-66)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-44)

[¶](#results-44)

| Result | Description |
| --- | --- |
| `matched` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `matched` | TransformHandleTypeInterface instance |
| `matched` | TransformHandleTypeInterface instance |
 `matched` |`matched` TransformHandleTypeInterface instance |

---

### `transform.with_pdl_patterns` (transform::WithPDLPatternsOp) [¶](#transformwith_pdl_patterns-transformwithpdlpatternsop)

`transform.with_pdl_patterns`
[¶](#transformwith_pdl_patterns-transformwithpdlpatternsop)

*Contains PDL patterns available for use in transforms*

*Contains PDL patterns available for use in transforms*

Syntax:

```
operation ::= `transform.with_pdl_patterns` ($root^ `:` type($root))? attr-dict-with-keyword regions
```

`` operation ::= `transform.with_pdl_patterns` ($root^ `:` type($root))? attr-dict-with-keyword regions ``

This op contains a set of named PDL patterns that are available for the
Transform dialect operations to be used for pattern matching. For example,
PDLMatchOp can be used to produce a Transform IR value associated with all
Payload IR operations that match the pattern as follows:

```
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

```
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

`transform.with_pdl_patterns {
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
}`
transform.with\_pdl\_patterns {
transform.with\_pdl\_patterns {
.
{
^bb0(%arg0: !transform.any\_op):
^bb0(%arg0: !transform.any\_op):
^bb0
(
%arg0
:
!
.
):
 pdl.pattern @my\_pattern : benefit(1) {
 pdl.pattern @my\_pattern : benefit(1) {
.
@my\_pattern
:
(
1
)
{
 %0 = pdl.operation //...
 %0 = pdl.operation //...
%0
=
.
//...
 // Regular PDL goes here.
 // Regular PDL goes here.

// Regular PDL goes here.
 pdl.rewrite %0 with "transform.dialect"
 pdl.rewrite %0 with "transform.dialect"

.
%0
"transform.dialect"
 }
 }
}




 sequence %arg0 failures(propagate) {
 sequence %arg0 failures(propagate) {
%arg0
(
)
{
 ^bb0(%arg1: !transform.any\_op):
 ^bb0(%arg1: !transform.any\_op):
^bb0
(
%arg1
:
!
.
):
 %1 = pdl\_match @my\_pattern in %arg1
 %1 = pdl\_match @my\_pattern in %arg1
%1
=
@my\_pattern
%arg1
 // Use %1 as handle
 // Use %1 as handle
// Use %1 as handle
 }
 }

}
}
}
}

Note that the pattern is expected to finish with a `pdl.rewrite` terminator
that points to the custom rewriter named “transform.dialect”. The rewriter
actually does nothing, but the transform application will keep track of the
operations that matched the pattern.

`pdl.rewrite`

This op is expected to contain `pdl.pattern` operations and exactly one
another Transform dialect operation that gets executed with all patterns
available. This op is a possible top-level Transform IR op, the argument of
its entry block corresponds to either the root op of the payload IR or the
ops associated with its operand when provided.

`pdl.pattern`

Traits: `NoTerminator`, `PossibleTopLevelTransformOpTrait`, `SymbolTable`

`NoTerminator`
`PossibleTopLevelTransformOpTrait`
`SymbolTable`

Interfaces: `MemoryEffectOpInterface`, `OpAsmOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`OpAsmOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-67)

[¶](#operands-67)

| Operand | Description |
| --- | --- |
| `root` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `root` | TransformHandleTypeInterface instance |
| `root` | TransformHandleTypeInterface instance |
 `root` |`root` TransformHandleTypeInterface instance |

---

### `transform.match.structured.body` (transform::MatchStructuredBodyOp) [¶](#transformmatchstructuredbody-transformmatchstructuredbodyop)

`transform.match.structured.body`
[¶](#transformmatchstructuredbody-transformmatchstructuredbodyop)

*Checks if the body of the structured op satisfies some criteria*

*Checks if the body of the structured op satisfies some criteria*

Syntax:

```
operation ::= `transform.match.structured.body` $operand_handle attr-dict `:` type($operand_handle)
```

`` operation ::= `transform.match.structured.body` $operand_handle attr-dict `:` type($operand_handle) ``

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

- `reduction_position`: the body of the structured payload op implements
  a reduction of the `n`-th operand (`n` is the value of the attribute)
  using a single combiner operation;

`reduction_position`: the body of the structured payload op implements
a reduction of the `n`-th operand (`n` is the value of the attribute)
using a single combiner operation;

`reduction_position`
`n`
`n`- `passthrough`: the body of the structured payload op only forwards
  inputs to the outputs (copy or broadcast).

`passthrough`: the body of the structured payload op only forwards
inputs to the outputs (copy or broadcast).

`passthrough`- `elementwise`: the body of the structured payload op represents an
  elementwise operation.

`elementwise`: the body of the structured payload op represents an
elementwise operation.

`elementwise`- `contraction`: the body of the structured payload op is a contraction
  of the form `<red>(<elem>(bbarg0, bbarg1), bbarg2)` where `<elem>` and
  `<red>` are binary operations whose names are specified in the attribute
  and operands can be permuted and optionally forwarded through a chain of
  unary side effect-free operations.

`contraction`: the body of the structured payload op is a contraction
of the form `<red>(<elem>(bbarg0, bbarg1), bbarg2)` where `<elem>` and
`<red>` are binary operations whose names are specified in the attribute
and operands can be permuted and optionally forwarded through a chain of
unary side effect-free operations.

`contraction`
`<red>(<elem>(bbarg0, bbarg1), bbarg2)`
`<elem>`
`<red>`

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-26)

[¶](#return-modes-26)

Succeeds if the operation body satisfies the specified criteria, produces a
silenceable failure otherwise. Produces a definite failure if the operand is
not associated with a single payload op.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-51)

[¶](#attributes-51)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `reduction_position` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `passthrough` | ::mlir::UnitAttr | unit attribute |
| `elementwise` | ::mlir::UnitAttr | unit attribute |
| `contraction` | ::mlir::ArrayAttr | string array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `reduction_position` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `reduction_position` |`reduction_position` ::mlir::IntegerAttr | 64-bit signless integer attribute || `passthrough` | ::mlir::UnitAttr | unit attribute |
 `passthrough` |`passthrough` ::mlir::UnitAttr | unit attribute || `elementwise` | ::mlir::UnitAttr | unit attribute |
 `elementwise` |`elementwise` ::mlir::UnitAttr | unit attribute || `contraction` | ::mlir::ArrayAttr | string array attribute |
 `contraction` |`contraction` ::mlir::ArrayAttr | string array attribute |

---

#### Operands: [¶](#operands-68)

[¶](#operands-68)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

### `transform.match.structured.classify_contraction_dims` (transform::MatchStructuredClassifyContractionDimsOp) [¶](#transformmatchstructuredclassify_contraction_dims-transformmatchstructuredclassifycontractiondimsop)

`transform.match.structured.classify_contraction_dims`
[¶](#transformmatchstructuredclassify_contraction_dims-transformmatchstructuredclassifycontractiondimsop)

*Checks if an operation has contraction-like dimensions and returns them*

*Checks if an operation has contraction-like dimensions and returns them*

Syntax:

```
operation ::= `transform.match.structured.classify_contraction_dims` $operand_handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.classify_contraction_dims` $operand_handle attr-dict `:` functional-type(operands, results) ``

Checks if the structured payload op has contraction-like dimensions as
follows:

C(batch, m, n) += A(batch, m, k) \* B(batch, k, n)

That is:

* ‘batch’ are parallel dimensions used in inputs and result;
* ’m’ are parallel dimensions used in the LHS and result;
* ’n’ are parallel dimensions used in rhe RHS and result;
* ‘k’ are reduction dimensions present only in LHS and RHS.

- ‘batch’ are parallel dimensions used in inputs and result;
- ’m’ are parallel dimensions used in the LHS and result;
- ’n’ are parallel dimensions used in rhe RHS and result;
- ‘k’ are reduction dimensions present only in LHS and RHS.

Note that this doesn’t check the operation in the body.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-27)

[¶](#return-modes-27)

Succeeds if the operation has the contraction-like dimensions, produces a
silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-69)

[¶](#operands-69)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-45)

[¶](#results-45)

| Result | Description |
| --- | --- |
| `batch` | TransformParamTypeInterface instance |
| `m` | TransformParamTypeInterface instance |
| `n` | TransformParamTypeInterface instance |
| `k` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `batch` | TransformParamTypeInterface instance |
| `m` | TransformParamTypeInterface instance |
| `n` | TransformParamTypeInterface instance |
| `k` | TransformParamTypeInterface instance |
| `batch` | TransformParamTypeInterface instance |
 `batch` |`batch` TransformParamTypeInterface instance || `m` | TransformParamTypeInterface instance |
 `m` |`m` TransformParamTypeInterface instance || `n` | TransformParamTypeInterface instance |
 `n` |`n` TransformParamTypeInterface instance || `k` | TransformParamTypeInterface instance |
 `k` |`k` TransformParamTypeInterface instance |

---

### `transform.match.structured.classify_convolution_dims` (transform::MatchStructuredClassifyConvolutionDimsOp) [¶](#transformmatchstructuredclassify_convolution_dims-transformmatchstructuredclassifyconvolutiondimsop)

`transform.match.structured.classify_convolution_dims`
[¶](#transformmatchstructuredclassify_convolution_dims-transformmatchstructuredclassifyconvolutiondimsop)

*Checks if an operation has convolution-like dimensions and returns them*

*Checks if an operation has convolution-like dimensions and returns them*

Syntax:

```
operation ::= `transform.match.structured.classify_convolution_dims` $operand_handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.classify_convolution_dims` $operand_handle attr-dict `:` functional-type(operands, results) ``

Checks if the structured payload op has convolution-like dimensions as
follows:

C(batch, depth, oi, oc) += A(batch, depth, oi, ic) \* B(fl, depth, ic, oc)

That is:

* ‘batch’ are parallel dimensions used in the input and result;
* ‘output\_image’ (‘oi’) are parallel dimensions used in the input and result;
* ‘output\_channel’ (‘oc’) are parallel dimensions used in the filter and result;
* ‘filter\_loop’ (‘fl’) are reduction dimensions representing the dimensions of the sliding window;
* ‘input\_channel’ (‘ic’) are reduction dimensions present only in the input and filter.
* ‘depth’ (‘ic’) are parallel dimensions present in the input, filter, and output.

- ‘batch’ are parallel dimensions used in the input and result;
- ‘output\_image’ (‘oi’) are parallel dimensions used in the input and result;
- ‘output\_channel’ (‘oc’) are parallel dimensions used in the filter and result;
- ‘filter\_loop’ (‘fl’) are reduction dimensions representing the dimensions of the sliding window;
- ‘input\_channel’ (‘ic’) are reduction dimensions present only in the input and filter.
- ‘depth’ (‘ic’) are parallel dimensions present in the input, filter, and output.

Additionally this will match stride and dilation information for the convolution:

* ‘strides’ are the static strides per convolution window dimension;
* ‘dilations’ are the static dilations per convolution window dimension.

- ‘strides’ are the static strides per convolution window dimension;
- ‘dilations’ are the static dilations per convolution window dimension.

Note that this doesn’t check the operation in the body.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-28)

[¶](#return-modes-28)

Succeeds if the operation has the convolution-like dimensions, produces a
silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-70)

[¶](#operands-70)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-46)

[¶](#results-46)

| Result | Description |
| --- | --- |
| `batch` | TransformParamTypeInterface instance |
| `output_image` | TransformParamTypeInterface instance |
| `output_channel` | TransformParamTypeInterface instance |
| `filter_loop` | TransformParamTypeInterface instance |
| `input_channel` | TransformParamTypeInterface instance |
| `depth` | TransformParamTypeInterface instance |
| `strides` | TransformParamTypeInterface instance |
| `dilations` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `batch` | TransformParamTypeInterface instance |
| `output_image` | TransformParamTypeInterface instance |
| `output_channel` | TransformParamTypeInterface instance |
| `filter_loop` | TransformParamTypeInterface instance |
| `input_channel` | TransformParamTypeInterface instance |
| `depth` | TransformParamTypeInterface instance |
| `strides` | TransformParamTypeInterface instance |
| `dilations` | TransformParamTypeInterface instance |
| `batch` | TransformParamTypeInterface instance |
 `batch` |`batch` TransformParamTypeInterface instance || `output_image` | TransformParamTypeInterface instance |
 `output_image` |`output_image` TransformParamTypeInterface instance || `output_channel` | TransformParamTypeInterface instance |
 `output_channel` |`output_channel` TransformParamTypeInterface instance || `filter_loop` | TransformParamTypeInterface instance |
 `filter_loop` |`filter_loop` TransformParamTypeInterface instance || `input_channel` | TransformParamTypeInterface instance |
 `input_channel` |`input_channel` TransformParamTypeInterface instance || `depth` | TransformParamTypeInterface instance |
 `depth` |`depth` TransformParamTypeInterface instance || `strides` | TransformParamTypeInterface instance |
 `strides` |`strides` TransformParamTypeInterface instance || `dilations` | TransformParamTypeInterface instance |
 `dilations` |`dilations` TransformParamTypeInterface instance |

---

### `transform.match.structured.dim` (transform::MatchStructuredDimOp) [¶](#transformmatchstructureddim-transformmatchstructureddimop)

`transform.match.structured.dim`
[¶](#transformmatchstructureddim-transformmatchstructureddimop)

*Checks if the dimensions of the structured op satisfy some criteria*

*Checks if the dimensions of the structured op satisfy some criteria*

Syntax:

```
operation ::= `transform.match.structured.dim` $operand_handle `[`custom<TransformMatchDims>($raw_dim_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

`` operation ::= `transform.match.structured.dim` $operand_handle `[`custom<TransformMatchDims>($raw_dim_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result)) ``

Checks if the dimensions (loop ranges) of the structured payload op satisfy
the criteria specified as attributes. May capture the numeric value of the
dimension into a parameter that it returns.

The following dimension specifications are supported:

* `all`: all dimensions are checked and captured;
* list of integers: the listed dimensions are checked and captured;
* `except(` list of integers `)`: all dimensions except the
  specified ones are checked and captured.

- `all`: all dimensions are checked and captured;
`all`- list of integers: the listed dimensions are checked and captured;
- `except(` list of integers `)`: all dimensions except the
  specified ones are checked and captured.
`except(`
`)`

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last dimension and
`except(-1)` means all dimensions but the last. Indexes must be unique,
including after interpretation of negative ones.

`-1`
`except(-1)`

Produces a silenceable failure in case of index overflow, including backward
counting.

The following mutually exclusive conditions are available as unit
attributes:

* `parallel`: the dimension corresponds to a parallel loop;
* `reduction`: the dimension corresponds to a reduction loop.

- `parallel`: the dimension corresponds to a parallel loop;
`parallel`- `reduction`: the dimension corresponds to a reduction loop.
`reduction`

If the result type is specified, associates the parameter with the (static)
values of dimensions in the same order as listed and preserving the natural
order for `all` and `except`. Specifically, if `-1, -2` are specified, the
parameter will be associated with the value of the second-to-last dimension
followed by the last dimension. If the dimension is dynamic, the parameter
will contain a negative value corresponding to kDynamic in C++.

`all`
`except`
`-1, -2`

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-29)

[¶](#return-modes-29)

Succeeds if the specified dimensions satisfy the specified criteria,
produces a silenceable failure otherwise. Produces a definite failure if
the operand is not associated with a single payload op.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-52)

[¶](#attributes-52)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `raw_dim_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `is_inverted` | ::mlir::UnitAttr | unit attribute |
| `is_all` | ::mlir::UnitAttr | unit attribute |
| `parallel` | ::mlir::UnitAttr | unit attribute |
| `reduction` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `raw_dim_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `raw_dim_list` |`raw_dim_list` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `is_inverted` | ::mlir::UnitAttr | unit attribute |
 `is_inverted` |`is_inverted` ::mlir::UnitAttr | unit attribute || `is_all` | ::mlir::UnitAttr | unit attribute |
 `is_all` |`is_all` ::mlir::UnitAttr | unit attribute || `parallel` | ::mlir::UnitAttr | unit attribute |
 `parallel` |`parallel` ::mlir::UnitAttr | unit attribute || `reduction` | ::mlir::UnitAttr | unit attribute |
 `reduction` |`reduction` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-71)

[¶](#operands-71)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-47)

[¶](#results-47)

| Result | Description |
| --- | --- |
| `result` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformParamTypeInterface instance |
| `result` | TransformParamTypeInterface instance |
 `result` |`result` TransformParamTypeInterface instance |

---

### `transform.match.structured.elemental_bitwidth` (transform::MatchStructuredElementalBitwidthOp) [¶](#transformmatchstructuredelemental_bitwidth-transformmatchstructuredelementalbitwidthop)

`transform.match.structured.elemental_bitwidth`
[¶](#transformmatchstructuredelemental_bitwidth-transformmatchstructuredelementalbitwidthop)

*Captures the bitwidth of the value’s elemental type as a parameter*

*Captures the bitwidth of the value’s elemental type as a parameter*

Syntax:

```
operation ::= `transform.match.structured.elemental_bitwidth` $operand_handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.elemental_bitwidth` $operand_handle attr-dict `:` functional-type(operands, results) ``

Produces a transform dialect parameter associated with the bitwidth of the
elemental type of the payload value passed as the operand.
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-30)

[¶](#return-modes-30)

Succeeds if the operand is associated with exactly one payload value of
`ShapedType`. Produces a silenceable failure otherwise.

`ShapedType`

Traits: `SingleValueMatcher`

`SingleValueMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-72)

[¶](#operands-72)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformValueHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformValueHandleTypeInterface instance |
| `operand_handle` | TransformValueHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformValueHandleTypeInterface instance |

---

#### Results: [¶](#results-48)

[¶](#results-48)

| Result | Description |
| --- | --- |
| `result` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformParamTypeInterface instance |
| `result` | TransformParamTypeInterface instance |
 `result` |`result` TransformParamTypeInterface instance |

---

### `transform.match.structured.init` (transform::MatchStructuredInitOp) [¶](#transformmatchstructuredinit-transformmatchstructuredinitop)

`transform.match.structured.init`
[¶](#transformmatchstructuredinit-transformmatchstructuredinitop)

*Captures init operand(s) of a structured operation*

*Captures init operand(s) of a structured operation*

Syntax:

```
operation ::= `transform.match.structured.init` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

`` operation ::= `transform.match.structured.init` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result)) ``

Produces a transform dialect value depending on the result type:

* If the result type is a value handle, it will be associated with the init
  operand(s) of the payload operation associated with the operand handle.
* If the result type is an operation handle, it will be associated with the
  operation defining the init operand(s) of the payload operation associated
  with the operand handle.
* If the result type is an affine map parameter type, it will be associated
  with the indexing map that corresponds to the init operand(s) of the
  payload operation associated with the operand handle.

- If the result type is a value handle, it will be associated with the init
  operand(s) of the payload operation associated with the operand handle.
- If the result type is an operation handle, it will be associated with the
  operation defining the init operand(s) of the payload operation associated
  with the operand handle.
- If the result type is an affine map parameter type, it will be associated
  with the indexing map that corresponds to the init operand(s) of the
  payload operation associated with the operand handle.

For example, given the following operation:

```
%arg3 = linalg.fill
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

```
%arg3 = linalg.fill
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

`%arg3 = linalg.fill
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)`
%arg3 = linalg.fill
%arg3 = linalg.fill
%arg3
=
.
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
.
(
%arg1
,
%arg2
:
...)
(
%arg3
:
...)

in case of a successful match for init operand 0 this operation will return,
for each of the respective cases above:

* A handle to `%arg3` if the result is a value handle.
* A handle to `linalg.fill` if the result is an operation handle.
* A parameter containing the result map of the matrix multiplication, i.e.
  `affine_map<(d0, d1, d2) -> (d0, d1)>` if the result is an affine
  map parameter.

- A handle to `%arg3` if the result is a value handle.
`%arg3`- A handle to `linalg.fill` if the result is an operation handle.
`linalg.fill`- A parameter containing the result map of the matrix multiplication, i.e.
  `affine_map<(d0, d1, d2) -> (d0, d1)>` if the result is an affine
  map parameter.
`affine_map<(d0, d1, d2) -> (d0, d1)>`

The match succeeds if the conditions specified as attributes succeed.

The following init specifications are supported:

* `all`: all inits are checked and captured;
* list of integers: the listed inits are checked and captured;
* `except(` list of integers `)`: all inits except the
  specified ones are checked and captured.

- `all`: all inits are checked and captured;
`all`- list of integers: the listed inits are checked and captured;
- `except(` list of integers `)`: all inits except the
  specified ones are checked and captured.
`except(`
`)`

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last init and
`except(-1)` means all inits but the last. Indexes must be unique,
including after interpretation of negative ones.

`-1`
`except(-1)`

Produces a silenceable failure in case of index overflow, including backward
counting.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-31)

[¶](#return-modes-31)

Succeeds if all init(outs) indexes are in bounds, produces a silenceable
failure otherwise. Additionally, when the result is an operation handle,
produces a silenceable failure if the init(outs) specification defines
more than one init(outs) or if the operand is not an operation result.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-53)

[¶](#attributes-53)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `is_inverted` | ::mlir::UnitAttr | unit attribute |
| `is_all` | ::mlir::UnitAttr | unit attribute |
| `permutation` | ::mlir::UnitAttr | unit attribute |
| `projected_permutation` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `raw_position_list` |`raw_position_list` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `is_inverted` | ::mlir::UnitAttr | unit attribute |
 `is_inverted` |`is_inverted` ::mlir::UnitAttr | unit attribute || `is_all` | ::mlir::UnitAttr | unit attribute |
 `is_all` |`is_all` ::mlir::UnitAttr | unit attribute || `permutation` | ::mlir::UnitAttr | unit attribute |
 `permutation` |`permutation` ::mlir::UnitAttr | unit attribute || `projected_permutation` | ::mlir::UnitAttr | unit attribute |
 `projected_permutation` |`projected_permutation` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-73)

[¶](#operands-73)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-49)

[¶](#results-49)

| Result | Description |
| --- | --- |
| `result` | transform operation or value handle or |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | transform operation or value handle or |
| `result` | transform operation or value handle or |
 `result` |`result` transform operation or value handle or |

---

### `transform.match.structured.input` (transform::MatchStructuredInputOp) [¶](#transformmatchstructuredinput-transformmatchstructuredinputop)

`transform.match.structured.input`
[¶](#transformmatchstructuredinput-transformmatchstructuredinputop)

*Captures input operand(s) of a structured operation*

*Captures input operand(s) of a structured operation*

Syntax:

```
operation ::= `transform.match.structured.input` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result))
```

`` operation ::= `transform.match.structured.input` $operand_handle `[`custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)`]` attr-dict `:` custom<SemiFunctionType>(type($operand_handle), type($result)) ``

Produces a transform dialect value depending on the result type:

* If the result type is a value handle, it will be associated with the input
  operand(s) of the payload operation associated with the operand handle.
* If the result type is an operation handle, it will be associated with the
  operation defining the input operand(s) of the payload operation associated
  with the operand handle.
* If the result type is an affine map parameter type, it will be associated
  with the indexing map that corresponds to the input operand(s) of the
  payload operation associated with the operand handle.

- If the result type is a value handle, it will be associated with the input
  operand(s) of the payload operation associated with the operand handle.
- If the result type is an operation handle, it will be associated with the
  operation defining the input operand(s) of the payload operation associated
  with the operand handle.
- If the result type is an affine map parameter type, it will be associated
  with the indexing map that corresponds to the input operand(s) of the
  payload operation associated with the operand handle.

For example, given the following operation:

```
%arg1 = some.op
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

```
%arg1 = some.op
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

`%arg1 = some.op
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)`
%arg1 = some.op
%arg1 = some.op
%arg1
=
.
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
.
(
%arg1
,
%arg2
:
...)
(
%arg3
:
...)

in case of a successful match for operand 0 this operation will return, for
each of the respective cases above:

* A handle to `%arg1` if the result is a value handle.
* A handle to `some.op` if the result is an operation handle.
* A parameter containing the LHS map of the matrix multiplication, i.e.
  `affine_map<(d0, d1, d2) -> (d0, d2)>` if the result is an affine
  map parameter.

- A handle to `%arg1` if the result is a value handle.
`%arg1`- A handle to `some.op` if the result is an operation handle.
`some.op`- A parameter containing the LHS map of the matrix multiplication, i.e.
  `affine_map<(d0, d1, d2) -> (d0, d2)>` if the result is an affine
  map parameter.
`affine_map<(d0, d1, d2) -> (d0, d2)>`

The match succeeds if the conditions specified as attributes succeed.

The following input specifications are supported:

* `all`: all inputs are checked and captured;
* list of integers: the listed inputs are checked and captured;
* `except(` list of integers `)`: all inputs except the
  specified ones are checked and captured.

- `all`: all inputs are checked and captured;
`all`- list of integers: the listed inputs are checked and captured;
- `except(` list of integers `)`: all inputs except the
  specified ones are checked and captured.
`except(`
`)`

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last input and
`except(-1)` means all inputs but the last. Indexes must be unique,
including after interpretation of negative ones.

`-1`
`except(-1)`

Produces a silenceable failure in case of index overflow, including backward
counting.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-32)

[¶](#return-modes-32)

Succeeds if all input indexes are in bounds, produces a silenceable failure
otherwise. Additionally, when the result is an operation handle, produces a
silenceable failure if the input specification defines more than one input
or if the operand is not an operation result.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-54)

[¶](#attributes-54)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `is_inverted` | ::mlir::UnitAttr | unit attribute |
| `is_all` | ::mlir::UnitAttr | unit attribute |
| `permutation` | ::mlir::UnitAttr | unit attribute |
| `projected_permutation` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `raw_position_list` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `raw_position_list` |`raw_position_list` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `is_inverted` | ::mlir::UnitAttr | unit attribute |
 `is_inverted` |`is_inverted` ::mlir::UnitAttr | unit attribute || `is_all` | ::mlir::UnitAttr | unit attribute |
 `is_all` |`is_all` ::mlir::UnitAttr | unit attribute || `permutation` | ::mlir::UnitAttr | unit attribute |
 `permutation` |`permutation` ::mlir::UnitAttr | unit attribute || `projected_permutation` | ::mlir::UnitAttr | unit attribute |
 `projected_permutation` |`projected_permutation` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-74)

[¶](#operands-74)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-50)

[¶](#results-50)

| Result | Description |
| --- | --- |
| `result` | transform operation or value handle or |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | transform operation or value handle or |
| `result` | transform operation or value handle or |
 `result` |`result` transform operation or value handle or |

---

### `transform.match.structured.num_inits` (transform::MatchStructuredNumInitsOp) [¶](#transformmatchstructurednum_inits-transformmatchstructurednuminitsop)

`transform.match.structured.num_inits`
[¶](#transformmatchstructurednum_inits-transformmatchstructurednuminitsop)

*Captures the number of init(outs) operands of a structuredoperation as parameter*

*Captures the number of init(outs) operands of a structuredoperation as parameter*

Syntax:

```
operation ::= `transform.match.structured.num_inits` $operand_handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.num_inits` $operand_handle attr-dict `:` functional-type(operands, results) ``

Produces a transform dialect parameter value associated with an integer
attribute containing the number of init(outs) operands of the payload
operation associated with the operand handle.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-33)

[¶](#return-modes-33)

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-75)

[¶](#operands-75)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-51)

[¶](#results-51)

| Result | Description |
| --- | --- |
| `result` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformParamTypeInterface instance |
| `result` | TransformParamTypeInterface instance |
 `result` |`result` TransformParamTypeInterface instance |

---

### `transform.match.structured.num_inputs` (transform::MatchStructuredNumInputsOp) [¶](#transformmatchstructurednum_inputs-transformmatchstructurednuminputsop)

`transform.match.structured.num_inputs`
[¶](#transformmatchstructurednum_inputs-transformmatchstructurednuminputsop)

*Captures the number of input operands of a structured operation as parameter*

*Captures the number of input operands of a structured operation as parameter*

Syntax:

```
operation ::= `transform.match.structured.num_inputs` $operand_handle attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.num_inputs` $operand_handle attr-dict `:` functional-type(operands, results) ``

Produces a transform dialect parameter value associated with an integer
attribute containing the number of input operands of the payload operation
associated with the operand handle.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-34)

[¶](#return-modes-34)

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-76)

[¶](#operands-76)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-52)

[¶](#results-52)

| Result | Description |
| --- | --- |
| `result` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformParamTypeInterface instance |
| `result` | TransformParamTypeInterface instance |
 `result` |`result` TransformParamTypeInterface instance |

---

### `transform.match.structured` (transform::MatchStructuredOp) [¶](#transformmatchstructured-transformmatchstructuredop)

`transform.match.structured`
[¶](#transformmatchstructured-transformmatchstructuredop)

*Matches a structured (linalg) operation with additional conditions*

*Matches a structured (linalg) operation with additional conditions*

Syntax:

```
operation ::= `transform.match.structured` (`failures` `(` $failure_propagation_mode^ `)`)?$current `:` custom<SemiFunctionType>(type($current), type($outputs))attr-dict-with-keyword regions
```

`` operation ::= `transform.match.structured` (`failures` `(` $failure_propagation_mode^ `)`)?$current `:` custom<SemiFunctionType>(type($current), type($outputs))attr-dict-with-keyword regions ``

Checks if the payload operation associated with the operand handle is a
structured operation, that is, an operation that implements
`LinalgOpInterface`, and that all conditions listed in the body of this
operation are satisfied. Produces a silenceable failure if the payload
operation is not structured.

`LinalgOpInterface`

The transform operations nested in the body region are applied one by one.
If any of them produces a failure, silenceable or definite, the following
operations are not applied. If the failure propagation mode is “propagate”,
silenceable failures are forwarded as the result of this operation. If it is
“suppress”, they are ignored and this operation immediately succeeds.
Definite failures are always propagated immediately.

In case of success, the transform values produced by this operation are
associated with the same payload as the operands of the block terminator. If
any of the nested operations produced a silenceable failure, regardless of
the failure propagation mode, the transform values produced by this
operation that correspond to the already defined terminator operands are
associated with the same payload as the already defined terminator operands.
Other values produced by this operation are associated with empty payloads.

If the failure propagation mode is not specified, it is considered
“propagate” by default. The “suppress” mode can be used to specify optional
matches.

---

#### Return modes [¶](#return-modes-35)

[¶](#return-modes-35)

This operation only reads all operand handles and produces all resulting
handles. It succeeds in “propagate” mode if the payload operation is a
structured operation and if all the nested operations succeed. It succeeds
in “suppress” mode as long as the operand handle is associated with exactly
one payload operation. It produces a definite failure when the handle is
not associated with exactly one payload operation.

Traits: `SingleBlockImplicitTerminator<::mlir::transform::MatchStructuredYieldOp>`, `SingleBlock`, `SingleOpMatcher`

`SingleBlockImplicitTerminator<::mlir::transform::MatchStructuredYieldOp>`
`SingleBlock`
`SingleOpMatcher`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-55)

[¶](#attributes-55)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `failure_propagation_mode` | ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |
 `failure_propagation_mode` |`failure_propagation_mode` ::mlir::transform::FailurePropagationModeAttr | Silenceable error propagation policy |

---

#### Operands: [¶](#operands-77)

[¶](#operands-77)

| Operand | Description |
| --- | --- |
| `current` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `current` | TransformHandleTypeInterface instance |
| `current` | TransformHandleTypeInterface instance |
 `current` |`current` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-53)

[¶](#results-53)

| Result | Description |
| --- | --- |
| `outputs` | variadic of any transform handle or parameter |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `outputs` | variadic of any transform handle or parameter |
| `outputs` | variadic of any transform handle or parameter |
 `outputs` |`outputs` variadic of any transform handle or parameter |

---

### `transform.match.structured.rank` (transform::MatchStructuredRankOp) [¶](#transformmatchstructuredrank-transformmatchstructuredrankop)

`transform.match.structured.rank`
[¶](#transformmatchstructuredrank-transformmatchstructuredrankop)

*Captures the rank of a structured operation as parameter*

*Captures the rank of a structured operation as parameter*

Syntax:

```
operation ::= `transform.match.structured.rank` $operand_handle attr-dict `:`
              custom<SemiFunctionType>(type($operand_handle), type($rank), "false")
```

`` operation ::= `transform.match.structured.rank` $operand_handle attr-dict `:`
custom<SemiFunctionType>(type($operand_handle), type($rank), "false") ``

Produces a transform dialect parameter value associated with an integer
attribute containing the rank of the structured payload operation associated
with the operand handle.

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-36)

[¶](#return-modes-36)

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-78)

[¶](#operands-78)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-54)

[¶](#results-54)

| Result | Description |
| --- | --- |
| `rank` | TransformParamTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `rank` | TransformParamTypeInterface instance |
| `rank` | TransformParamTypeInterface instance |
 `rank` |`rank` TransformParamTypeInterface instance |

---

### `transform.match.structured.result` (transform::MatchStructuredResultOp) [¶](#transformmatchstructuredresult-transformmatchstructuredresultop)

`transform.match.structured.result`
[¶](#transformmatchstructuredresult-transformmatchstructuredresultop)

*Captures the result of a structured payload operation in an op or value handle*

*Captures the result of a structured payload operation in an op or value handle*

Syntax:

```
operation ::= `transform.match.structured.result` $operand_handle `[` $position `]` (`any` $any^)? (`single` $single^)?attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.match.structured.result` $operand_handle `[` $position `]` (`any` $any^)? (`single` $single^)?attr-dict `:` functional-type(operands, results) ``

Produces a transform dialect value handle associated with the payload value
defined as a result of the payload operation associated with the operand
handle, or an operation handle to an operation using the produced result
with additional constraints specified by the attributes as follows.

* If `any` is specified, binds the resulting handle to any operation using
  the result and succeeds.
* If `single` is specified, binds the resulting handle to the only
  operation using the result or fails if there is more than one (or no)
  such operation.

- If `any` is specified, binds the resulting handle to any operation using
  the result and succeeds.
`any`- If `single` is specified, binds the resulting handle to the only
  operation using the result or fails if there is more than one (or no)
  such operation.
`single`

The number of the result is specified as `position` attribute. It may take
positive and negative values. Negative values are interpreted as counting
results from backwards, e.g., `-1` means the last result and `-2` means the
second-to-last result. In any case, the position must be in bounds for the
given payload operation. A silenceable failure is produced for out-of-bounds
positions.

`position`
`-1`
`-2`

This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.

`transform.match.structured`

---

#### Return modes [¶](#return-modes-37)

[¶](#return-modes-37)

Succeeds if the position is in bounds and if the user operation could be
found when requested. Produces a silenceable failure otherwise.

Traits: `SingleOpMatcher`, `StructuredPredicate`

`SingleOpMatcher`
`StructuredPredicate`

Interfaces: `MatchOpInterface`, `MemoryEffectsOpInterface`, `TransformOpInterface`

`MatchOpInterface`
`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-56)

[¶](#attributes-56)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `position` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `any` | ::mlir::UnitAttr | unit attribute |
| `single` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `position` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `position` |`position` ::mlir::IntegerAttr | 64-bit signless integer attribute || `any` | ::mlir::UnitAttr | unit attribute |
 `any` |`any` ::mlir::UnitAttr | unit attribute || `single` | ::mlir::UnitAttr | unit attribute |
 `single` |`single` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-79)

[¶](#operands-79)

| Operand | Description |
| --- | --- |
| `operand_handle` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `operand_handle` | TransformHandleTypeInterface instance |
| `operand_handle` | TransformHandleTypeInterface instance |
 `operand_handle` |`operand_handle` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-55)

[¶](#results-55)

| Result | Description |
| --- | --- |
| `result` | transform operation or value handle |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | transform operation or value handle |
| `result` | transform operation or value handle |
 `result` |`result` transform operation or value handle |

---

### `transform.match.structured.yield` (transform::MatchStructuredYieldOp) [¶](#transformmatchstructuredyield-transformmatchstructuredyieldop)

`transform.match.structured.yield`
[¶](#transformmatchstructuredyield-transformmatchstructuredyieldop)

*Terminator for transform.match.structured blocks*

*Terminator for transform.match.structured blocks*

Syntax:

```
operation ::= `transform.match.structured.yield` $handles attr-dict (`:` type($handles)^)?
```

`` operation ::= `transform.match.structured.yield` $handles attr-dict (`:` type($handles)^)? ``

Forwards the payload association from the operands to the results of the
parent op. Always succeeds.

Traits: `Terminator`

`Terminator`

Interfaces: `MemoryEffectOpInterface`

`MemoryEffectOpInterface`

---

#### Operands: [¶](#operands-80)

[¶](#operands-80)

| Operand | Description |
| --- | --- |
| `handles` | variadic of any transform handle or parameter |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `handles` | variadic of any transform handle or parameter |
| `handles` | variadic of any transform handle or parameter |
 `handles` |`handles` variadic of any transform handle or parameter |

---

### `transform.apply_patterns.linalg.decompose_pack_unpack` (transform::ApplyDecomposeTensorPackUnpackPatternsOp) [¶](#transformapply_patternslinalgdecompose_pack_unpack-transformapplydecomposetensorpackunpackpatternsop)

`transform.apply_patterns.linalg.decompose_pack_unpack`
[¶](#transformapply_patternslinalgdecompose_pack_unpack-transformapplydecomposetensorpackunpackpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.decompose_pack_unpack` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.decompose_pack_unpack` attr-dict ``

Collect patterns to decompose linalg.pack and linalg.unpack into e.g.
tensor::PadOp, linalg::transposeOp Ops. Requires all outer dims to be unit.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.decompose_pad` (transform::ApplyDecomposeTensorPadPatternsOp) [¶](#transformapply_patternslinalgdecompose_pad-transformapplydecomposetensorpadpatternsop)

`transform.apply_patterns.linalg.decompose_pad`
[¶](#transformapply_patternslinalgdecompose_pad-transformapplydecomposetensorpadpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.decompose_pad` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.decompose_pad` attr-dict ``

Collect patterns to decompose tensor.pad into e.g. tensor::EmptyOp,
linalg::FillOp and tensor::InsertSliceOp.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.erase_unnecessary_inputs` (transform::ApplyEraseUnnecessaryInputsPatternsOp) [¶](#transformapply_patternslinalgerase_unnecessary_inputs-transformapplyeraseunnecessaryinputspatternsop)

`transform.apply_patterns.linalg.erase_unnecessary_inputs`
[¶](#transformapply_patternslinalgerase_unnecessary_inputs-transformapplyeraseunnecessaryinputspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.erase_unnecessary_inputs` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.erase_unnecessary_inputs` attr-dict ``

Collects patterns that promote inputs to outputs and remove unused inputs of
`linalg.generic` ops.

`linalg.generic`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.fold_add_into_dest` (transform::ApplyFoldAddIntoDestPatternsOp) [¶](#transformapply_patternslinalgfold_add_into_dest-transformapplyfoldaddintodestpatternsop)

`transform.apply_patterns.linalg.fold_add_into_dest`
[¶](#transformapply_patternslinalgfold_add_into_dest-transformapplyfoldaddintodestpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_add_into_dest` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.fold_add_into_dest` attr-dict ``

Collects patterns to replace linalg.add when destination passing suffices
for achieving the sum.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.fold_into_pack_and_unpack` (transform::ApplyFoldIntoPackAndUnpackPatternsOp) [¶](#transformapply_patternstensorfold_into_pack_and_unpack-transformapplyfoldintopackandunpackpatternsop)

`transform.apply_patterns.tensor.fold_into_pack_and_unpack`
[¶](#transformapply_patternstensorfold_into_pack_and_unpack-transformapplyfoldintopackandunpackpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_into_pack_and_unpack` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.fold_into_pack_and_unpack` attr-dict ``

Indicates that operations like tensor.pad and tensor.extract\_slice should
be folded into linalg.pack and linalg.unpack operations, respectively.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.fold_pack_unpack_into_empty` (transform::ApplyFoldPackUnpackIntoEmptyPatternsOp) [¶](#transformapply_patternslinalgfold_pack_unpack_into_empty-transformapplyfoldpackunpackintoemptypatternsop)

`transform.apply_patterns.linalg.fold_pack_unpack_into_empty`
[¶](#transformapply_patternslinalgfold_pack_unpack_into_empty-transformapplyfoldpackunpackintoemptypatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_pack_unpack_into_empty` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.fold_pack_unpack_into_empty` attr-dict ``

// TODO:

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-57)

[¶](#attributes-57)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fold_single_use_only` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `fold_single_use_only` | ::mlir::BoolAttr | bool attribute |
 `fold_single_use_only` |`fold_single_use_only` ::mlir::BoolAttr | bool attribute |

---

### `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` (transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp) [¶](#transformapply_patternslinalgfold_unit_extent_dims_via_reshapes-transformapplyfoldunitextentdimsviareshapespatternsop)

`transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes`
[¶](#transformapply_patternslinalgfold_unit_extent_dims_via_reshapes-transformapplyfoldunitextentdimsviareshapespatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` attr-dict ``

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via reassociative reshape ops.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` (transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp) [¶](#transformapply_patternslinalgfold_unit_extent_dims_via_slices-transformapplyfoldunitextentdimsviaslicespatternsop)

`transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices`
[¶](#transformapply_patternslinalgfold_unit_extent_dims_via_slices-transformapplyfoldunitextentdimsviaslicespatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` attr-dict ``

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via rank-reducing slices.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.pad_vectorization` (transform::ApplyPadVectorizationPatternsOp) [¶](#transformapply_patternslinalgpad_vectorization-transformapplypadvectorizationpatternsop)

`transform.apply_patterns.linalg.pad_vectorization`
[¶](#transformapply_patternslinalgpad_vectorization-transformapplypadvectorizationpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.pad_vectorization` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.pad_vectorization` attr-dict ``

Apply patterns that vectorize tensor.pad.

These patterns rewrite tensor.pad Ops using vector.transfer\_read and
vector.transfer\_write operations. This is done either by:

1. Folding tensor.pad with an existing vector.transfer\_read /
   vector.transfer\_write Op (generated prior to running these patterns).
2. Rewriting it (when matched together with q tensor.insert\_slice
   consumer Op) as a vector.transfer\_read + vector.transfer\_write pair.

- Folding tensor.pad with an existing vector.transfer\_read /
  vector.transfer\_write Op (generated prior to running these patterns).
- Rewriting it (when matched together with q tensor.insert\_slice
  consumer Op) as a vector.transfer\_read + vector.transfer\_write pair.

In both cases, these patterns look at producers and consumers for the
matched tensor.pad Op to find opportunities for vectorization.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.linalg.tiling_canonicalization` (transform::ApplyTilingCanonicalizationPatternsOp) [¶](#transformapply_patternslinalgtiling_canonicalization-transformapplytilingcanonicalizationpatternsop)

`transform.apply_patterns.linalg.tiling_canonicalization`
[¶](#transformapply_patternslinalgtiling_canonicalization-transformapplytilingcanonicalizationpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.linalg.tiling_canonicalization` attr-dict
```

`` operation ::= `transform.apply_patterns.linalg.tiling_canonicalization` attr-dict ``

Collects canonicalization patterns relevant to apply after tiling patterns.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.structured.bufferize_to_allocation` (transform::BufferizeToAllocationOp) [¶](#transformstructuredbufferize_to_allocation-transformbufferizetoallocationop)

`transform.structured.bufferize_to_allocation`
[¶](#transformstructuredbufferize_to_allocation-transformbufferizetoallocationop)

Syntax:

```
operation ::= `transform.structured.bufferize_to_allocation` $target attr-dict `:` type($target)
```

`` operation ::= `transform.structured.bufferize_to_allocation` $target attr-dict `:` type($target) ``

This transform bufferizes the targeted operation and materializes the
result in a new allocation. It replaces all original uses of the target
result with the newly allocated buffer, wrapped in a
`bufferization.to_tensor` op. It returns a handle to the newly allocated
buffer. Furthermore, it returns a handle that is mapped to all newly created
ops.

`bufferization.to_tensor`

Only bufferizable ops are that bufferize to a memory write or have an
aliasing OpOperand (and do not themselves bufferize to an allocation) are
supported. They are bufferized using their BufferizableOpInterface
implementation. E.g.:

```
%0 = tensor.insert %f into %dest[%pos] : tensor<10xf32>
```

`%0 = tensor.insert %f into %dest[%pos] : tensor<10xf32>`

Is bufferized to:

```
%alloc = memref.alloc() : memref<10xf32>
bufferization.materialize_in_destination %dest in %alloc
memref.store %f, %alloc[%pos] : memref<10xf32>
%0 = bufferization.to_tensor %alloc restrict writable : memref<10xf32>
```

`%alloc = memref.alloc() : memref<10xf32>
bufferization.materialize_in_destination %dest in %alloc
memref.store %f, %alloc[%pos] : memref<10xf32>
%0 = bufferization.to_tensor %alloc restrict writable : memref<10xf32>`

Selected ops that bufferize to an allocation (or need special handling) are
also supported:

* `tensor.pad` is lowered to an allocation, followed by a `linalg.fill` and
  and a buffer copy (all on memrefs).
* `vector.mask` is bufferized together with its region. The allocation is
  placed in front of the `vector.mask` op.

- `tensor.pad` is lowered to an allocation, followed by a `linalg.fill` and
  and a buffer copy (all on memrefs).
`tensor.pad`
`linalg.fill`- `vector.mask` is bufferized together with its region. The allocation is
  placed in front of the `vector.mask` op.
`vector.mask`
`vector.mask`

An optional memory space attribute can be specified for the materialized
buffer allocation.

If a memory copy is needed, a “bufferization.materialize\_in\_destination” is
used when possible. This is an op with tensor semantics that will bufferize
to a memory copy later. Which concrete op will be used for the memory copy
is up to the bufferization framework. Alternatively, a custom memcpy op can
be specified via `memcpy_op`. Currently supported are “memref.copy” and
“linalg.copy”. In that case, the source of each memcpy must not have a
custom memory space. Furthermore, because the future buffer layout unknown
for a given tensor, a fully dynamic layout is assumed for best
compatibility. Users should use “bufferization.materialize\_in\_destination”
when possible.

`memcpy_op`

“memref.alloc” is used for new buffer allocations. The buffer is deallocated
at the end of the block if the “emit\_dealloc” attribute is present. If this
attribute is not present, the allocated memory will be leaked. However,
running the `-buffer-deallocation-pipeline` after all bufferization is done
will properly insert the corresponding deallocation(s). Custom allocation
ops can be specified via `alloc_op`. Currently supported are “memref.alloc”
and “memref.alloca”. In case of a “memref.alloca”, the buffer is not
deallocated.

`-buffer-deallocation-pipeline`
`alloc_op`

If `bufferize_destination_only` is set, only the destination operands of the
op are bufferized to a new memory allocation, but not the op itself.

`bufferize_destination_only`

---

#### Return modes [¶](#return-modes-38)

[¶](#return-modes-38)

This operation consumes the `target` handle and produces the
`allocated_buffer` and `new_ops` handles. It always succeeds.

`target`
`allocated_buffer`
`new_ops`

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`InferTypeOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-58)

[¶](#attributes-58)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `memory_space` | ::mlir::Attribute | any attribute |
| `memcpy_op` | ::mlir::StringAttr | string attribute |
| `alloc_op` | ::mlir::StringAttr | string attribute |
| `bufferize_destination_only` | ::mlir::UnitAttr | unit attribute |
| `emit_dealloc` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `memory_space` | ::mlir::Attribute | any attribute |
 `memory_space` |`memory_space` ::mlir::Attribute | any attribute || `memcpy_op` | ::mlir::StringAttr | string attribute |
 `memcpy_op` |`memcpy_op` ::mlir::StringAttr | string attribute || `alloc_op` | ::mlir::StringAttr | string attribute |
 `alloc_op` |`alloc_op` ::mlir::StringAttr | string attribute || `bufferize_destination_only` | ::mlir::UnitAttr | unit attribute |
 `bufferize_destination_only` |`bufferize_destination_only` ::mlir::UnitAttr | unit attribute || `emit_dealloc` | ::mlir::UnitAttr | unit attribute |
 `emit_dealloc` |`emit_dealloc` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-81)

[¶](#operands-81)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-56)

[¶](#results-56)

| Result | Description |
| --- | --- |
| `allocated_buffer` |  |
| `new_ops` |  |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `allocated_buffer` |  |
| `new_ops` |  |
| `allocated_buffer` |  |
 `allocated_buffer` |`allocated_buffer`  || `new_ops` |  |
 `new_ops` |`new_ops`  |

---

### `transform.structured.continuous_tile_sizes` (transform::ContinuousTileSizesOp) [¶](#transformstructuredcontinuous_tile_sizes-transformcontinuoustilesizesop)

`transform.structured.continuous_tile_sizes`
[¶](#transformstructuredcontinuous_tile_sizes-transformcontinuoustilesizesop)

Syntax:

```
operation ::= `transform.structured.continuous_tile_sizes` $target attr-dict `:` custom<ContinuousTileSizeTypes>(type($target), type($tile_sizes), type($chunk_sizes))
```

`` operation ::= `transform.structured.continuous_tile_sizes` $target attr-dict `:` custom<ContinuousTileSizeTypes>(type($target), type($tile_sizes), type($chunk_sizes)) ``

This transform emits the IR computing the list of (1) exponentially
diminishing tile sizes that are powers of 2; and (2) the corresponding
chunk-sizes the target op should be split into along the given dimension.

For example, for `target_size` 9, and `dimension` 0 for the following
linalg op as target

`target_size`
`dimension`

```
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                  outs(%arg2: tensor<25x25xf32>)
```

 `%0 = linalg.matmul ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
outs(%arg2: tensor<25x25xf32>)`

the first result `tile_sizes` will be a list of diminishing tile sizes
9, 4, 2, 1; and the second result will be a list of chunk sizes
18, 4, 2, 1 that the corresponding dimension should be split into.

`tile_sizes`

After the target op has been split along the given dimension (for example
using multiway split), each chunk can be tiled with the corresponding tile
size in the `tile_sizes` list generated as a result of this op.

`tile_sizes`

Specifying the output type as !transform.param will cause `tile_sizes`
and `chunk_sizes` to be computed statically and not dynamically.

 will cause `tile_sizes`
and `chunk_sizes` to be computed statically and not dynamically.
`tile_sizes`
`chunk_sizes`

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-59)

[¶](#attributes-59)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative |
| `target_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative |
 `dimension` |`dimension` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative || `target_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative |
 `target_size` |`target_size` ::mlir::IntegerAttr | 64-bit signless integer attribute whose value is non-negative |

---

#### Operands: [¶](#operands-82)

[¶](#operands-82)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-57)

[¶](#results-57)

| Result | Description |
| --- | --- |
| `tile_sizes` | transform any param type or any handle type |
| `chunk_sizes` | transform any param type or any handle type |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `tile_sizes` | transform any param type or any handle type |
| `chunk_sizes` | transform any param type or any handle type |
| `tile_sizes` | transform any param type or any handle type |
 `tile_sizes` |`tile_sizes` transform any param type or any handle type || `chunk_sizes` | transform any param type or any handle type |
 `chunk_sizes` |`chunk_sizes` transform any param type or any handle type |

---

### `transform.structured.convert_conv2d_to_img2col` (transform::ConvertConv2DToImg2ColOp) [¶](#transformstructuredconvert_conv2d_to_img2col-transformconvertconv2dtoimg2colop)

`transform.structured.convert_conv2d_to_img2col`
[¶](#transformstructuredconvert_conv2d_to_img2col-transformconvertconv2dtoimg2colop)

Syntax:

```
operation ::= `transform.structured.convert_conv2d_to_img2col` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.convert_conv2d_to_img2col` $target attr-dict `:` functional-type($target, results) ``

Convert linalg.conv\_2d\_xxx into linalg.generic (for img2col packing)
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

`[x(0, 0) , x(0, 1) , ..., x(0, n) ]
[x(1, 0) , x(1, 1) , ..., x(1, n) ]
[. , . ,. , . ] [w(0, 0), w(0, 1)]
[. , . , . , . ] (conv) [w(1, 0), w(1, 1)]
[. , . , ., . ]
[x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]`

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

`[x(0,0), x(0,1), x(1,0), x(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)]
[ . , . , . , . ]`

In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
multiplication (Ho x Wo, Kh x Kw x C) \* (Kh x Kw x C, D) for each input in
the N input. For the case where N > 1 its a batched matrxi-matrix
multplication.

Returns two handles:

* One on the operation that produces the img2col tensor.
* One on the final operation of the sequence that replaces the original
  convolution.

- One on the operation that produces the img2col tensor.
- One on the final operation of the sequence that replaces the original
  convolution.

---

#### Return modes: [¶](#return-modes-39)

[¶](#return-modes-39)

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-83)

[¶](#operands-83)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-58)

[¶](#results-58)

| Result | Description |
| --- | --- |
| `img2col_tensor` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `img2col_tensor` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
| `img2col_tensor` | TransformHandleTypeInterface instance |
 `img2col_tensor` |`img2col_tensor` TransformHandleTypeInterface instance || `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.convert_to_loops` (transform::ConvertToLoopsOp) [¶](#transformstructuredconvert_to_loops-transformconverttoloopsop)

`transform.structured.convert_to_loops`
[¶](#transformstructuredconvert_to_loops-transformconverttoloopsop)

Syntax:

```
operation ::= `transform.structured.convert_to_loops` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.convert_to_loops` $target attr-dict `:` functional-type(operands, results) ``

For operations that implement the `TilingInterface`, and implement
the `generateScalarImplementation` method, lowers the operation to
loops. The return handle points to all generated loops.
Fails if the payload ops cannot be lowered to loops.

`TilingInterface`
`generateScalarImplementation`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-84)

[¶](#operands-84)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-59)

[¶](#results-59)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.structured.decompose_interface` (transform::DecomposeInterfaceOp) [¶](#transformstructureddecompose_interface-transformdecomposeinterfaceop)

`transform.structured.decompose_interface`
[¶](#transformstructureddecompose_interface-transformdecomposeinterfaceop)

Syntax:

```
operation ::= `transform.structured.decompose_interface` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.decompose_interface` $target attr-dict `:` functional-type(operands, results) ``

TODO

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-85)

[¶](#operands-85)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-60)

[¶](#results-60)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.decompose` (transform::DecomposeOp) [¶](#transformstructureddecompose-transformdecomposeop)

`transform.structured.decompose`
[¶](#transformstructureddecompose-transformdecomposeop)

Syntax:

```
operation ::= `transform.structured.decompose` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.decompose` $target attr-dict `:` functional-type(operands, results) ``

Decomposes named complex operations, such as higher-dimensional
(depthwise) convolutions, into combinations of lower-dimensional equivalents
when possible.

---

#### Return modes [¶](#return-modes-40)

[¶](#return-modes-40)

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle decompose
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure. The return handle points to only the subset of
successfully produced computational operations, which can be empty.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-86)

[¶](#operands-86)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-61)

[¶](#results-61)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.decompose_winograd_op` (transform::DecomposeWinogradOp) [¶](#transformstructureddecompose_winograd_op-transformdecomposewinogradop)

`transform.structured.decompose_winograd_op`
[¶](#transformstructureddecompose_winograd_op-transformdecomposewinogradop)

Syntax:

```
operation ::= `transform.structured.decompose_winograd_op` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.decompose_winograd_op` $target attr-dict `:` functional-type($target, results) ``

Decompose winograd operations. It will convert filter, input and output
transform operations into a combination of scf, tensor, and linalg
equivalent operations. Before applying this transform operations, users
need to tile winograd transform operations into supported sizes.

---

#### Return modes: [¶](#return-modes-41)

[¶](#return-modes-41)

This operation fails if `target` is unsupported. Otherwise, the operation
succeeds and returns a handle of the sequence that replaces the original
operations.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-87)

[¶](#operands-87)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-62)

[¶](#results-62)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.eliminate_empty_tensors` (transform::EliminateLinalgOpAnchoredEmptyTensorsOp) [¶](#transformstructuredeliminate_empty_tensors-transformeliminatelinalgopanchoredemptytensorsop)

`transform.structured.eliminate_empty_tensors`
[¶](#transformstructuredeliminate_empty_tensors-transformeliminatelinalgopanchoredemptytensorsop)

Syntax:

```
operation ::= `transform.structured.eliminate_empty_tensors` $target attr-dict `:` type($target)
```

`` operation ::= `transform.structured.eliminate_empty_tensors` $target attr-dict `:` type($target) ``

Try to eliminate all `tensor.empty` op uses that are anchored on a LinalgOp
within the targeted op.

`tensor.empty`

This op is similar to `bufferization.eliminate_empty_tensors`, but specific
to LinalgOps.

`bufferization.eliminate_empty_tensors`

`tensor.empty` ops cannot be bufferized. They can either be converted to
`bufferization.alloc_tensor` or replaced with another tensor (via this
transform). `tensor.empty` does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

`tensor.empty`
`bufferization.alloc_tensor`
`tensor.empty`

This transform looks for `tensor.empty` ops where the SSA use-def chain of
the result ends in a supported LinalgOp (always following the aliasing
OpOperand/OpResult chain). The following LinalgOps are supported:

`tensor.empty`

* Only parallel iterator types.
* The use-def chain ends in an input operand of the LinalgOp.
* The LinalgOp has an unused output operand with the same shape and
  indexing map.

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

`%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%0)
%2 = linalg.generic ins(%1) outs(%dest) {
^bb0(%in: f32, %out: f32):
// out not used
}`

Is rewritten with:

```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%dest)
%2 = linalg.generic ins(%0) outs(%1) {
  ^bb0(%in: f32, %out: f32):
  // Use %out instead of %in
}
```

`%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%dest)
%2 = linalg.generic ins(%0) outs(%1) {
^bb0(%in: f32, %out: f32):
// Use %out instead of %in
}`

After this transformation, the “ins” operand has no uses inside the body of
the LinalgOp and can be folded away with existing cleanup patterns.
Afterwards, the tensor::EmptyOp can also fold away, so that the example can
bufferize without an allocation (in the absence of other conflicts).

---

#### Return modes [¶](#return-modes-42)

[¶](#return-modes-42)

This transform reads the target handle and modifies the payload. It does
not produce any handle.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-88)

[¶](#operands-88)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

### `transform.structured.flatten_elementwise` (transform::FlattenElementwiseLinalgOp) [¶](#transformstructuredflatten_elementwise-transformflattenelementwiselinalgop)

`transform.structured.flatten_elementwise`
[¶](#transformstructuredflatten_elementwise-transformflattenelementwiselinalgop)

Syntax:

```
operation ::= `transform.structured.flatten_elementwise` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.flatten_elementwise` $target attr-dict `:` functional-type($target, results) ``

Flattens the iteration space and (applicable) operands of elementwise
linalg ops to a single dimension.

Returns one handle:

* Flattened linalg operation.

- Flattened linalg operation.

---

#### Return modes: [¶](#return-modes-43)

[¶](#return-modes-43)

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-89)

[¶](#operands-89)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-63)

[¶](#results-63)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.fuse_into_containing_op` (transform::FuseIntoContainingOp) [¶](#transformstructuredfuse_into_containing_op-transformfuseintocontainingop)

`transform.structured.fuse_into_containing_op`
[¶](#transformstructuredfuse_into_containing_op-transformfuseintocontainingop)

*Fuse a producer into a containing operation.*

*Fuse a producer into a containing operation.*

Syntax:

```
operation ::= `transform.structured.fuse_into_containing_op` $producer_op `into` $containing_op attr-dict  `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.fuse_into_containing_op` $producer_op `into` $containing_op attr-dict `:` functional-type(operands, results) ``

Fuses the `producer_op` into the `containing_op`.
Returns a handle to the fused ops and the `new_containing_op`.

`producer_op`
`containing_op`
`new_containing_op`

The producer is typically a slice of a tileable op (i.e., implements
TilingInterface). In that case, this transform computes the accessed
producer slice inside of the containing op (“tile and fuse”) and if required,
creates a new containing op with outputs from the fused producer. Otherwise,
the entire producer is cloned inside the containing op (“clone and fuse”).

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

---

#### Return modes [¶](#return-modes-44)

[¶](#return-modes-44)

If at least one producer could not be fused, this operation produces a
silenceable failure. This is the case when tiling fails or when no
producer op could be found among the remaining producers that has at least
one use within the containing op. I.e., “producers” that are not consumed
within the containing op are rejected by this operation.

This operation consumes the producer handle.
This operation only reads the containing op handle.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-90)

[¶](#operands-90)

| Operand | Description |
| --- | --- |
| `producer_op` | TransformHandleTypeInterface instance |
| `containing_op` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `producer_op` | TransformHandleTypeInterface instance |
| `containing_op` | TransformHandleTypeInterface instance |
| `producer_op` | TransformHandleTypeInterface instance |
 `producer_op` |`producer_op` TransformHandleTypeInterface instance || `containing_op` | TransformHandleTypeInterface instance |
 `containing_op` |`containing_op` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-64)

[¶](#results-64)

| Result | Description |
| --- | --- |
| `fused_op` | TransformHandleTypeInterface instance |
| `new_containing_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `fused_op` | TransformHandleTypeInterface instance |
| `new_containing_op` | TransformHandleTypeInterface instance |
| `fused_op` | TransformHandleTypeInterface instance |
 `fused_op` |`fused_op` TransformHandleTypeInterface instance || `new_containing_op` | TransformHandleTypeInterface instance |
 `new_containing_op` |`new_containing_op` TransformHandleTypeInterface instance |

---

### `transform.structured.fuse` (transform::FuseOp) [¶](#transformstructuredfuse-transformfuseop)

`transform.structured.fuse`
[¶](#transformstructuredfuse-transformfuseop)

Syntax:

```
operation ::= `transform.structured.fuse` $target oilist(
              `tile_sizes` custom<DynamicIndexList>($tile_sizes, $static_tile_sizes) |
              `interchange` custom<DynamicIndexList>($tile_interchange, $static_tile_interchange)
              )
              attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.fuse` $target oilist(
`tile_sizes` custom<DynamicIndexList>($tile_sizes, $static_tile_sizes) |
`interchange` custom<DynamicIndexList>($tile_interchange, $static_tile_interchange)
)
attr-dict `:` functional-type(operands, results) ``

Tiles the operations pointed to by the target handle and fuses their
producers greedily using the options provided as attributes. Tile sizes
and loop interchange permutation can be provided as either static
attributes or dynamic values (transform parameters or payload handles).

If `apply_cleanup` is true then slice canonicalization is applied between
fusion steps. If `use_forall` is true then tiling method generates a
`scf.forall` loop instead of `scf.for` loops.

`apply_cleanup`
`use_forall`
`scf.forall`
`scf.for`

Traits: `AttrSizedOperandSegments`, `ReportTrackingListenerFailuresOpTrait`

`AttrSizedOperandSegments`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-60)

[¶](#attributes-60)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_tile_interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `apply_cleanup` | ::mlir::UnitAttr | unit attribute |
| `use_forall` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_tile_sizes` |`static_tile_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `static_tile_interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_tile_interchange` |`static_tile_interchange` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `apply_cleanup` | ::mlir::UnitAttr | unit attribute |
 `apply_cleanup` |`apply_cleanup` ::mlir::UnitAttr | unit attribute || `use_forall` | ::mlir::UnitAttr | unit attribute |
 `use_forall` |`use_forall` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-91)

[¶](#operands-91)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `tile_interchange` | variadic of transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `tile_interchange` | variadic of transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `tile_sizes` | variadic of transform any param type or any handle type |
 `tile_sizes` |`tile_sizes` variadic of transform any param type or any handle type || `tile_interchange` | variadic of transform any param type or any handle type |
 `tile_interchange` |`tile_interchange` variadic of transform any param type or any handle type |

---

#### Results: [¶](#results-65)

[¶](#results-65)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance || `loops` | variadic of TransformHandleTypeInterface instance |
 `loops` |`loops` variadic of TransformHandleTypeInterface instance |

---

### `transform.structured.generalize` (transform::GeneralizeOp) [¶](#transformstructuredgeneralize-transformgeneralizeop)

`transform.structured.generalize`
[¶](#transformstructuredgeneralize-transformgeneralizeop)

Syntax:

```
operation ::= `transform.structured.generalize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

`` operation ::= `transform.structured.generalize` $target attr-dict `:`
custom<SemiFunctionType>(type($target), type($transformed), "false") ``

Transforms a named structured operation into the generic form with the
explicit attached region.

---

#### Return modes [¶](#return-modes-45)

[¶](#return-modes-45)

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure. The return handle points to only the subset of
successfully produced equivalent generic operations, which can be empty or
contain the original ops if they were already in generic form.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-92)

[¶](#operands-92)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-66)

[¶](#results-66)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.hoist_pad.build_packing_loop_nest` (transform::HoistPadBuildPackingLoopNestOp) [¶](#transformstructuredhoist_padbuild_packing_loop_nest-transformhoistpadbuildpackingloopnestop)

`transform.structured.hoist_pad.build_packing_loop_nest`
[¶](#transformstructuredhoist_padbuild_packing_loop_nest-transformhoistpadbuildpackingloopnestop)

Syntax:

```
operation ::= `transform.structured.hoist_pad.build_packing_loop_nest` $target
              `above` $loop
              (`,` `transpose` `by` $transpose^)?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.hoist_pad.build_packing_loop_nest` $target
`above` $loop
(`,` `transpose` `by` $transpose^)?
attr-dict
`:` functional-type(operands, results) ``

Helper transform used to hoist a tensor.pad target operation. This operation
creates the packing loop nest required by the hoist\_pad operation and makes
that functionality available independently.

TODO: In the future, we should consider rewriting as a linalg.pack after
hoisting since this abstraction is now available.

---

#### Return modes [¶](#return-modes-46)

[¶](#return-modes-46)

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

The return handle points to only the subset of successfully created packing
loop nests, which can be empty.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-61)

[¶](#attributes-61)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `transpose` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `transpose` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `transpose` |`transpose` ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

---

#### Operands: [¶](#operands-93)

[¶](#operands-93)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `loop` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `loop` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `loop` | TransformHandleTypeInterface instance |
 `loop` |`loop` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-67)

[¶](#results-67)

| Result | Description |
| --- | --- |
| `packing_loop` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `packing_loop` | TransformHandleTypeInterface instance |
| `packing_loop` | TransformHandleTypeInterface instance |
 `packing_loop` |`packing_loop` TransformHandleTypeInterface instance |

---

### `transform.structured.hoist_pad` (transform::HoistPadOp) [¶](#transformstructuredhoist_pad-transformhoistpadop)

`transform.structured.hoist_pad`
[¶](#transformstructuredhoist_pad-transformhoistpadop)

Syntax:

```
operation ::= `transform.structured.hoist_pad` $target
              `by` $num_loops `loops`
              (`,` `transpose` `by` $transpose^)?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.hoist_pad` $target
`by` $num_loops `loops`
(`,` `transpose` `by` $transpose^)?
attr-dict
`:` functional-type(operands, results) ``

Hoist the tensor.pad target operation by at most the given number of loops.
Optionally apply the transpose attribute to the inner dimensions.

TODO: In the future, we should consider rewriting as a linalg.pack after
hoisting since this abstraction is now available.
TODO: Maybe also return the linalg.generic transpose created at some point.

---

#### Return modes [¶](#return-modes-47)

[¶](#return-modes-47)

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

If all the operations referred to by the `target` handle padproperly, the
transform succeeds. Otherwise the transform produces a silenceable failure.

`target`

The return handle points to only the subset of successfully hoisted
tensor.pad operations, which can be empty.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-62)

[¶](#attributes-62)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `transpose` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `num_loops` |`num_loops` ::mlir::IntegerAttr | 64-bit signless integer attribute || `transpose` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `transpose` |`transpose` ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

---

#### Operands: [¶](#operands-94)

[¶](#operands-94)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-68)

[¶](#results-68)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.hoist_redundant_vector_broadcasts` (transform::HoistRedundantVectorBroadcastsOp) [¶](#transformstructuredhoist_redundant_vector_broadcasts-transformhoistredundantvectorbroadcastsop)

`transform.structured.hoist_redundant_vector_broadcasts`
[¶](#transformstructuredhoist_redundant_vector_broadcasts-transformhoistredundantvectorbroadcastsop)

Syntax:

```
operation ::= `transform.structured.hoist_redundant_vector_broadcasts` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.hoist_redundant_vector_broadcasts` $target attr-dict `:` functional-type(operands, results) ``

Hoist vector.extract / vector.broadcasts pairs out of immediately
enclosing scf::ForOp iteratively.

---

#### Return modes: [¶](#return-modes-48)

[¶](#return-modes-48)

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-95)

[¶](#operands-95)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-69)

[¶](#results-69)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.hoist_redundant_vector_transfers` (transform::HoistRedundantVectorTransfersOp) [¶](#transformstructuredhoist_redundant_vector_transfers-transformhoistredundantvectortransfersop)

`transform.structured.hoist_redundant_vector_transfers`
[¶](#transformstructuredhoist_redundant_vector_transfers-transformhoistredundantvectortransfersop)

Syntax:

```
operation ::= `transform.structured.hoist_redundant_vector_transfers` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.hoist_redundant_vector_transfers` $target attr-dict `:` functional-type(operands, results) ``

Hoist vector.transfer\_read / vector.transfer\_write pairs out of immediately
enclosing scf::ForOp iteratively, if the following conditions are true:

1. The 2 ops access the same memref with the same indices.
2. All operands are invariant under the enclosing scf::ForOp.
3. No uses of the memref either dominate the transfer\_read or are
   dominated by the transfer\_write (i.e. no aliasing between the write and
   the read across the loop)

- The 2 ops access the same memref with the same indices.
- All operands are invariant under the enclosing scf::ForOp.
- No uses of the memref either dominate the transfer\_read or are
  dominated by the transfer\_write (i.e. no aliasing between the write and
  the read across the loop)

WARNING: This hoisting does not model parallelism and is generally incorrect
when used on distributed loops with memref semantics!
TODO: obsolete and should be retired.

---

#### Return modes: [¶](#return-modes-49)

[¶](#return-modes-49)

The operation always succeeds and returns a handle to the transformed
function op.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-63)

[¶](#attributes-63)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `verify_non_zero_trip` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `verify_non_zero_trip` | ::mlir::UnitAttr | unit attribute |
 `verify_non_zero_trip` |`verify_non_zero_trip` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-96)

[¶](#operands-96)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-70)

[¶](#results-70)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.insert_slice_to_copy` (transform::InsertSliceToCopyOp) [¶](#transformstructuredinsert_slice_to_copy-transforminsertslicetocopyop)

`transform.structured.insert_slice_to_copy`
[¶](#transformstructuredinsert_slice_to_copy-transforminsertslicetocopyop)

Syntax:

```
operation ::= `transform.structured.insert_slice_to_copy` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.insert_slice_to_copy` $target attr-dict `:` functional-type(operands, results) ``

Targeted rewrite of an tensor.insert\_slice to linalg.copy.
This is useful to materialize copies explicitly before bufferization and
transform them, avoiding the need to rediscover them after bufferization.

If the insert\_slice source is already a linalg.copy, only return the source
op (i.e. do not create an additional linalg.copy op).

---

#### Return modes: [¶](#return-modes-50)

[¶](#return-modes-50)

The operation always succeeds and returns a handle to the relevant
linalg.copy op.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-97)

[¶](#operands-97)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-71)

[¶](#results-71)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.interchange` (transform::InterchangeOp) [¶](#transformstructuredinterchange-transforminterchangeop)

`transform.structured.interchange`
[¶](#transformstructuredinterchange-transforminterchangeop)

Syntax:

```
operation ::= `transform.structured.interchange` $target
              (`iterator_interchange` `=` $iterator_interchange^)? attr-dict
              `:` custom<SemiFunctionType>(type($target), type($transformed), "false")
```

`` operation ::= `transform.structured.interchange` $target
(`iterator_interchange` `=` $iterator_interchange^)? attr-dict
`:` custom<SemiFunctionType>(type($target), type($transformed), "false") ``

Interchanges the iterators of the operations pointed to by the target handle
using the iterator interchange attribute.

---

#### Return modes [¶](#return-modes-51)

[¶](#return-modes-51)

This operation ignores non-linalg::Generic ops and drops them in the return.
This operation fails if the interchange attribute is invalid.
If all the operations referred to by the `target` handle interchange
properly, the transform succeeds.
If any interchange fails, the transform produces a definite failure.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-64)

[¶](#attributes-64)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `iterator_interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute whose value is non-negative |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `iterator_interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute whose value is non-negative |
 `iterator_interchange` |`iterator_interchange` ::mlir::DenseI64ArrayAttr | i64 dense array attribute whose value is non-negative |

---

#### Operands: [¶](#operands-98)

[¶](#operands-98)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-72)

[¶](#results-72)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.linalg_copy_to_memref` (transform::LinalgCopyToMemrefOp) [¶](#transformstructuredlinalg_copy_to_memref-transformlinalgcopytomemrefop)

`transform.structured.linalg_copy_to_memref`
[¶](#transformstructuredlinalg_copy_to_memref-transformlinalgcopytomemrefop)

Syntax:

```
operation ::= `transform.structured.linalg_copy_to_memref` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.linalg_copy_to_memref` $target attr-dict `:` functional-type(operands, results) ``

Targeted rewrite of a linalg.copy on memrefs to a memref.copy.
This is useful when bufferizing copies to a linalg.copy, later applying some
transformations, and then rewriting the copy into a memref.copy.
If the element types of the source and destination differ, or if the source
is a scalar, the transform produces a silenceable failure.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-99)

[¶](#operands-99)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-73)

[¶](#results-73)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.lower_pack` (transform::LowerPackOp) [¶](#transformstructuredlower_pack-transformlowerpackop)

`transform.structured.lower_pack`
[¶](#transformstructuredlower_pack-transformlowerpackop)

Syntax:

```
operation ::= `transform.structured.lower_pack` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.lower_pack` $target attr-dict `:` functional-type(operands, results) ``

Rewrite a linalg.pack into tensor.pad + tensor.expand\_shape + linalg.transpose.

---

#### Return modes [¶](#return-modes-52)

[¶](#return-modes-52)

This operation ignores non-pack ops and drops them in the return. This
operation produces a silenceable failure if the rewrite fails for any
reason. If all the operations referred to by the `target` are rewritten,
the transform succeeds. Return handles to the newly produced pad,
expand\_shape and transpose ops.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-65)

[¶](#attributes-65)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowerPadLikeWithInsertSlice` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lowerPadLikeWithInsertSlice` | ::mlir::BoolAttr | bool attribute |
 `lowerPadLikeWithInsertSlice` |`lowerPadLikeWithInsertSlice` ::mlir::BoolAttr | bool attribute |

---

#### Operands: [¶](#operands-100)

[¶](#operands-100)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to linalg.pack operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to linalg.pack operations |
| `target` | Transform IR handle to linalg.pack operations |
 `target` |`target` Transform IR handle to linalg.pack operations |

---

#### Results: [¶](#results-74)

[¶](#results-74)

| Result | Description |
| --- | --- |
| `pad_op` | Transform IR handle to tensor.pad operations |
| `expand_shape_op` | Transform IR handle to tensor.expand\_shape operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `pad_op` | Transform IR handle to tensor.pad operations |
| `expand_shape_op` | Transform IR handle to tensor.expand\_shape operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |
| `pad_op` | Transform IR handle to tensor.pad operations |
 `pad_op` |`pad_op` Transform IR handle to tensor.pad operations || `expand_shape_op` | Transform IR handle to tensor.expand\_shape operations |
 `expand_shape_op` |`expand_shape_op` Transform IR handle to tensor.expand\_shape operations || `transpose_op` | Transform IR handle to linalg.transpose operations |
 `transpose_op` |`transpose_op` Transform IR handle to linalg.transpose operations |

---

### `transform.structured.lower_unpack` (transform::LowerUnPackOp) [¶](#transformstructuredlower_unpack-transformlowerunpackop)

`transform.structured.lower_unpack`
[¶](#transformstructuredlower_unpack-transformlowerunpackop)

Syntax:

```
operation ::= `transform.structured.lower_unpack` $target attr-dict `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.lower_unpack` $target attr-dict `:` functional-type(operands, results) ``

Lower a linalg.unpack into empty + linalg.transpose + tensor.collapse\_shape +
tensor.extract\_slice.

---

#### Return modes [¶](#return-modes-53)

[¶](#return-modes-53)

This operation ignores non-unpack ops and drops them in the return. This
operation produces a silenceable failure if the rewrite fails for any
reason. If all the operations referred to by the `target` are rewritten,
the transform succeeds. Return handles to the newly produced empty,
transpose, collapse\_shape and extract\_slice ops.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-66)

[¶](#attributes-66)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowerUnpadLikeWithExtractSlice` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lowerUnpadLikeWithExtractSlice` | ::mlir::BoolAttr | bool attribute |
 `lowerUnpadLikeWithExtractSlice` |`lowerUnpadLikeWithExtractSlice` ::mlir::BoolAttr | bool attribute |

---

#### Operands: [¶](#operands-101)

[¶](#operands-101)

| Operand | Description |
| --- | --- |
| `target` | Transform IR handle to linalg.unpack operations |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | Transform IR handle to linalg.unpack operations |
| `target` | Transform IR handle to linalg.unpack operations |
 `target` |`target` Transform IR handle to linalg.unpack operations |

---

#### Results: [¶](#results-75)

[¶](#results-75)

| Result | Description |
| --- | --- |
| `empty_op` | Transform IR handle to tensor.empty operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |
| `collapse_shape_op` | Transform IR handle to tensor.collapse\_shape operations |
| `extract_slice_op` | Transform IR handle to tensor.extract\_slice operations |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `empty_op` | Transform IR handle to tensor.empty operations |
| `transpose_op` | Transform IR handle to linalg.transpose operations |
| `collapse_shape_op` | Transform IR handle to tensor.collapse\_shape operations |
| `extract_slice_op` | Transform IR handle to tensor.extract\_slice operations |
| `empty_op` | Transform IR handle to tensor.empty operations |
 `empty_op` |`empty_op` Transform IR handle to tensor.empty operations || `transpose_op` | Transform IR handle to linalg.transpose operations |
 `transpose_op` |`transpose_op` Transform IR handle to linalg.transpose operations || `collapse_shape_op` | Transform IR handle to tensor.collapse\_shape operations |
 `collapse_shape_op` |`collapse_shape_op` Transform IR handle to tensor.collapse\_shape operations || `extract_slice_op` | Transform IR handle to tensor.extract\_slice operations |
 `extract_slice_op` |`extract_slice_op` Transform IR handle to tensor.extract\_slice operations |

---

### `transform.structured.gpu.map_copy_to_threads` (transform::MapCopyToThreadsOp) [¶](#transformstructuredgpumap_copy_to_threads-transformmapcopytothreadsop)

`transform.structured.gpu.map_copy_to_threads`
[¶](#transformstructuredgpumap_copy_to_threads-transformmapcopytothreadsop)

Syntax:

```
operation ::= `transform.structured.gpu.map_copy_to_threads` $target
              `total_num_threads` `=` $total_num_threads
              `desired_bit_alignment` `=` $desired_bit_alignment
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.gpu.map_copy_to_threads` $target
`total_num_threads` `=` $total_num_threads
`desired_bit_alignment` `=` $desired_bit_alignment
attr-dict
`:` functional-type(operands, results) ``

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

---

#### Return modes [¶](#return-modes-54)

[¶](#return-modes-54)

This operation fails definitely if there is an unsupported op (i.e., not
linalg.copy / tensor.pad) among the targeted op. Otherwise, the operation
always succeeds and returns a handle to the relevant tiled linalg.copy /
tensor.pad op and the enclosing scf.forall op.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-67)

[¶](#attributes-67)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `total_num_threads` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `desired_bit_alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `total_num_threads` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `total_num_threads` |`total_num_threads` ::mlir::IntegerAttr | 64-bit signless integer attribute || `desired_bit_alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `desired_bit_alignment` |`desired_bit_alignment` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-102)

[¶](#operands-102)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-76)

[¶](#results-76)

| Result | Description |
| --- | --- |
| `forall_op` | TransformHandleTypeInterface instance |
| `tiled_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `forall_op` | TransformHandleTypeInterface instance |
| `tiled_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |
 `forall_op` |`forall_op` TransformHandleTypeInterface instance || `tiled_op` | TransformHandleTypeInterface instance |
 `tiled_op` |`tiled_op` TransformHandleTypeInterface instance |

---

### `transform.structured.match` (transform::MatchOp) [¶](#transformstructuredmatch-transformmatchop)

`transform.structured.match`
[¶](#transformstructuredmatch-transformmatchop)

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

`` operation ::= `transform.structured.match` (`ops` `{` $ops^ `}`)?
(`interface` `{` $interface^ `}`)?
(`attributes` $op_attrs^)?
(`filter_result_type` `=` $filter_result_type^)?
(`filter_operand_types` `=` $filter_operand_types^)?
`in` $target attr-dict
`:` functional-type($target, results) ``

Match op with the specified constraints, within the target op.

The following constraints are supported:

* interface: an optional MatchInterfaceEnum specifying an enum
  representation for an interface to target.
* ops: an optional StrArrayAttr specifying the concrete name of an op.
  Multiple names can be specified. Matched ops must have one of specified
  names.
* attribute: the matched op must have all specified attributes (with their
  specified values).
* filter\_result\_type: the matched op must return exactly this one type.
* filter\_operand\_types: all the operands of the matched op must must be of
  this type. If more than a type is specified, then the length of the list
  must be equal to the number of operands in the matched op, and the match
  will succeed only if the operand types match all the types in the list
  in the order in which they are specified.

- interface: an optional MatchInterfaceEnum specifying an enum
  representation for an interface to target.
- ops: an optional StrArrayAttr specifying the concrete name of an op.
  Multiple names can be specified. Matched ops must have one of specified
  names.
- attribute: the matched op must have all specified attributes (with their
  specified values).
- filter\_result\_type: the matched op must return exactly this one type.
- filter\_operand\_types: all the operands of the matched op must must be of
  this type. If more than a type is specified, then the length of the list
  must be equal to the number of operands in the matched op, and the match
  will succeed only if the operand types match all the types in the list
  in the order in which they are specified.

Note: Only ops that satisfy all specified constraints are matched.

TODO: Extend with regions to allow a limited form of constraints.

---

#### Return modes [¶](#return-modes-55)

[¶](#return-modes-55)

This op traverses the ops nested under `target` and returns the handles to
all the operations that match the requirements.

`target`

This op fails if the target is not a handle to exactly one operation.
Otherwise it succeeds.

This operation does not consume the target handle and produces new handles:
it is a navigation op.

Traits: `NavigationTransformOpTrait`

`NavigationTransformOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-68)

[¶](#attributes-68)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ops` | ::mlir::ArrayAttr | string array attribute |
| `interface` | mlir::transform::MatchInterfaceEnumAttr | An interface to match |
| `op_attrs` | ::mlir::DictionaryAttr | dictionary of named attribute values |
| `filter_result_type` | ::mlir::TypeAttr | any type attribute |
| `filter_operand_types` | ::mlir::ArrayAttr | type array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `ops` | ::mlir::ArrayAttr | string array attribute |
 `ops` |`ops` ::mlir::ArrayAttr | string array attribute || `interface` | mlir::transform::MatchInterfaceEnumAttr | An interface to match |
 `interface` |`interface` mlir::transform::MatchInterfaceEnumAttr | An interface to match || `op_attrs` | ::mlir::DictionaryAttr | dictionary of named attribute values |
 `op_attrs` |`op_attrs` ::mlir::DictionaryAttr | dictionary of named attribute values || `filter_result_type` | ::mlir::TypeAttr | any type attribute |
 `filter_result_type` |`filter_result_type` ::mlir::TypeAttr | any type attribute || `filter_operand_types` | ::mlir::ArrayAttr | type array attribute |
 `filter_operand_types` |`filter_operand_types` ::mlir::ArrayAttr | type array attribute |

---

#### Operands: [¶](#operands-103)

[¶](#operands-103)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-77)

[¶](#results-77)

| Result | Description |
| --- | --- |
| `results` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `results` | TransformHandleTypeInterface instance |
| `results` | TransformHandleTypeInterface instance |
 `results` |`results` TransformHandleTypeInterface instance |

---

### `transform.structured.multitile_sizes` (transform::MultiTileSizesOp) [¶](#transformstructuredmultitile_sizes-transformmultitilesizesop)

`transform.structured.multitile_sizes`
[¶](#transformstructuredmultitile_sizes-transformmultitilesizesop)

Syntax:

```
operation ::= `transform.structured.multitile_sizes` $target attr-dict `:` custom<MultitileSizesTypes>(type($target), type($low_size), type($high_size), type($split_point))
```

`` operation ::= `transform.structured.multitile_sizes` $target attr-dict `:` custom<MultitileSizesTypes>(type($target), type($low_size), type($high_size), type($split_point)) ``

Emits the IR computing the tile sizes `s1` and `s2` such that:

`s1`
`s2`

* there exists a combination of `n` tiles of size `s1` and `m` tiles of
  size `s2` that covers the entirety of the iteration space `dimension` of
  the target structured op;
* `s1`, `s2` is less than or equal to `target_size`;
* `s1` and `s2` are divisible by `divisor.

- there exists a combination of `n` tiles of size `s1` and `m` tiles of
  size `s2` that covers the entirety of the iteration space `dimension` of
  the target structured op;
`n`
`s1`
`m`
`s2`
`dimension`- `s1`, `s2` is less than or equal to `target_size`;
`s1`
`s2`
`target_size`- `s1` and `s2` are divisible by `divisor.
`s1`
`s2`

For example, for a dimension of size 54 with target size 12 and divisor 2,
this can emit the IR computing the tile size 10, used for 3 tiles, and 12,
used for 2 tiles, totally 10*3 + 12*2 = 54. Note that when the divisor does
not divide the original dimension size, it is impossible to compute such
tile sizes. An assertion is emitted to guard against this in the dynamic
case.

*3 + 12*

Expects the target size and the divisor to be strictly positive. Folds the
IR as much as possible, normally obtaining constant sizes and numbers of
tiles for a statically known dimension.

This does *not* consume the target handle and produces three handles each
pointing to single-result index-typed operations (which may be arithmetic
constant operations) defining the two respective tile sizes and the product
of the first tile size with the number of tiles of that size (useful for
splitting the iteration space).

*not*

This operation composes with the regular tiling when applied per-dimension:

```
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

```
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

`%sz1, %sz2, %split = structured.multitile_sizes %target
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
// ...`
%sz1, %sz2, %split = structured.multitile\_sizes %target
%sz1, %sz2, %split = structured.multitile\_sizes %target
%sz1
,
%sz2
,
%split
=
.
%target
 { target\_size = 10, dimension = 1 }
 { target\_size = 10, dimension = 1 }
{
target\_size =
10
,
dimension =
1
}
 : !transform.any\_op, !transform.param<i64>,
 : !transform.any\_op, !transform.param<i64>,
:
!
.
,
!
.
<
i64
>,
 !transform.param<i64>, !transform.param<i64>
 !transform.param<i64>, !transform.param<i64>
!
.
<
i64
>,
!
.
<
i64
>
%handles = structured.split %target after %split { dimension = 1 }
%handles = structured.split %target after %split { dimension = 1 }
%handles
=
.
%target
%split
{
dimension =
1
}
 : !transform.any\_op, !transform.param<i64>
 : !transform.any\_op, !transform.param<i64>
:
!
.
,
!
.
<
i64
>
%low, %high = transform.split\_handle %handles : (!transform.any\_op)
%low, %high = transform.split\_handle %handles : (!transform.any\_op)
%low
,
%high
=
.
%handles
:
(!
.
)
 -> (!transform.any\_op, !transform.any\_op)
 -> (!transform.any\_op, !transform.any\_op)
->
(!
.
,
!
.
)
%tiled\_low, %loop1 = structured.tile\_using\_for %low [0, %sz1]
%tiled\_low, %loop1 = structured.tile\_using\_for %low [0, %sz1]
%tiled\_low
,
%loop1
=
.
%low
[
0
,
%sz1
]
 : (!transform.any\_op, !transform.param<i64>)
 : (!transform.any\_op, !transform.param<i64>)
:
(!
.
,
!
.
<
i64
>)
 -> (!transform.any\_op, !transform.any\_op)
 -> (!transform.any\_op, !transform.any\_op)
->
(!
.
,
!
.
)
%tiled\_high, %loop2 = structured.tile\_using\_for %high [0, %sz2]
%tiled\_high, %loop2 = structured.tile\_using\_for %high [0, %sz2]
%tiled\_high
,
%loop2
=
.
%high
[
0
,
%sz2
]
 : (!transform.any\_op, !transform.param<i64>)
 : (!transform.any\_op, !transform.param<i64>)
:
(!
.
,
!
.
<
i64
>)
 -> (!transform.any\_op, !transform.any\_op)
 -> (!transform.any\_op, !transform.any\_op)
->
(!
.
,
!
.
)
%common = merge\_handles %tiled\_low, %tiled\_high : !transform.any\_op
%common = merge\_handles %tiled\_low, %tiled\_high : !transform.any\_op
%common
=
%tiled\_low
,
%tiled\_high
:
!
.




%sz3, %sz4, %split = structured.multitile\_size %target
%sz3, %sz4, %split = structured.multitile\_size %target
%sz3
,
%sz4
,
%split
=
.
%target
 { target\_size = 42, dimension = 0 }
 { target\_size = 42, dimension = 0 }
{
target\_size =
42
,
dimension =
0
}
 : !transform.any\_op, !transform.any\_op,
 : !transform.any\_op, !transform.any\_op,
:
!
.
,
!
.
,
 !transform.any\_op, !transform.any\_op
 !transform.any\_op, !transform.any\_op
!
.
,
!
.
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
%sz3r
,
%sz4r
,
%splitr
=
(
%common
)
%sz3
,
%sz4
,
%splitr
 : !transform.any\_op, !transform.any\_op, !transform.any\_op
 : !transform.any\_op, !transform.any\_op, !transform.any\_op
:
!
.
,
!
.
,
!
.
structured.split %common after %splitr { dimension = 0 }
structured.split %common after %splitr { dimension = 0 }
.
%common
%splitr
{
dimension =
0
}
 : !transform.any\_op, !transform.any\_op
 : !transform.any\_op, !transform.any\_op
:
!
.
,
!
.
// ...
// ...
// ...

Traits: `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-69)

[¶](#attributes-69)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `target_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `divisor` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `dimension` |`dimension` ::mlir::IntegerAttr | 64-bit signless integer attribute || `target_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `target_size` |`target_size` ::mlir::IntegerAttr | 64-bit signless integer attribute || `divisor` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `divisor` |`divisor` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-104)

[¶](#operands-104)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-78)

[¶](#results-78)

| Result | Description |
| --- | --- |
| `low_size` | transform any param type or any handle type |
| `high_size` | transform any param type or any handle type |
| `split_point` | transform any param type or any handle type |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `low_size` | transform any param type or any handle type |
| `high_size` | transform any param type or any handle type |
| `split_point` | transform any param type or any handle type |
| `low_size` | transform any param type or any handle type |
 `low_size` |`low_size` transform any param type or any handle type || `high_size` | transform any param type or any handle type |
 `high_size` |`high_size` transform any param type or any handle type || `split_point` | transform any param type or any handle type |
 `split_point` |`split_point` transform any param type or any handle type |

---

### `transform.structured.pack_greedily` (transform::PackGreedilyOp) [¶](#transformstructuredpack_greedily-transformpackgreedilyop)

`transform.structured.pack_greedily`
[¶](#transformstructuredpack_greedily-transformpackgreedilyop)

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

`` operation ::= `transform.structured.pack_greedily` $target
oilist(
`matmul_packed_sizes` `=` custom<DynamicIndexList>($matmul_packed_sizes,
$static_matmul_packed_sizes)
(`matmul_padded_sizes_next_multiple_of` `=`
$matmul_padded_sizes_next_multiple_of^)?
`matmul_inner_dims_order` `=` $matmul_inner_dims_order
)
attr-dict
`:` functional-type(operands, results) ``

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

- Matmul packing: Try to infer a matmul operation embedded in the target op.
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

Matmul packing: Try to infer a matmul operation embedded in the target op.
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

`matmul_packed_sizes`
`matmul_padded_sizes_next_multiple_of`
`matmul_packed_sizes[i]`
`matmul_packed_sizes[i]`
`matmul_padded_sizes_next_multiple_of[i]`

`matmul_padded_sizes_next_multiple_of` is optional and is expected to
either be empty or of size `3`, matching the size of `matmul_packed_sizes`.
For each individual element of `matmul_packed_sizes` and
`matmul_padded_sizes_next_multiple_of`, only one of them is allowed to
be non-zero.

`matmul_padded_sizes_next_multiple_of`
`3`
`matmul_packed_sizes`
`matmul_packed_sizes`
`matmul_padded_sizes_next_multiple_of`

The ordering of the packed dimensions (mm, nn, kk) is specified by the
`matmul_inner_dims_order` attribute.

`matmul_inner_dims_order`

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

- Find the dimensions to pack according to the strategy.
- The target is converted to linalg.generic form.
- An interchange transform is applied to isolate the dimensions to pack as
  the most minor indexing dimensions of the linalg.generic. The most minor
  dimensions are themselves ordered according to `inner_dims_order`.
`inner_dims_order`- An elementwise traversal of `matmul_packed_sizes` and
  `matmul_padded_sizes_next_multiple_of` is performed and for each
  dimension `d`, either pack to `matmul_packed_sizes[d]` or pad to the
  `matmul_padded_sizes_next_multiple_of[d]`.
`matmul_packed_sizes`
`matmul_padded_sizes_next_multiple_of`
`d`
`matmul_packed_sizes[d]`
`matmul_padded_sizes_next_multiple_of[d]`- Packing/padding is performed by the amounts determined in step 4. and
  following `inner_dims_order`.
`inner_dims_order`

By normalizing the most minor dimensions to `inner_dims_order`, the transform
guarantees that packing immediately generates inner dimensions in a desirable
layout.

`inner_dims_order`

Outer dimension layout permutations are not controlled by this transform op
at the moment and can be obtained by composing with the pack\_transpose
transformation.

---

#### Return modes [¶](#return-modes-56)

[¶](#return-modes-56)

This operation ignores non-Linalg ops and drops them in the return.
It returns the list of packed Linalg ops or the original op when all available
packing strategies failed to apply.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-70)

[¶](#attributes-70)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_matmul_packed_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements |
| `matmul_padded_sizes_next_multiple_of` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with 0 or 3 elements |
| `matmul_inner_dims_order` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_matmul_packed_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements |
 `static_matmul_packed_sizes` |`static_matmul_packed_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements || `matmul_padded_sizes_next_multiple_of` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with 0 or 3 elements |
 `matmul_padded_sizes_next_multiple_of` |`matmul_padded_sizes_next_multiple_of` ::mlir::DenseI64ArrayAttr | i64 dense array attribute with 0 or 3 elements || `matmul_inner_dims_order` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements |
 `matmul_inner_dims_order` |`matmul_inner_dims_order` ::mlir::DenseI64ArrayAttr | i64 dense array attribute with exactly 3 elements |

---

#### Operands: [¶](#operands-105)

[¶](#operands-105)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `matmul_packed_sizes` | variadic of TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `matmul_packed_sizes` | variadic of TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `matmul_packed_sizes` | variadic of TransformHandleTypeInterface instance |
 `matmul_packed_sizes` |`matmul_packed_sizes` variadic of TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-79)

[¶](#results-79)

| Result | Description |
| --- | --- |
| `packed_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `packed_op` | TransformHandleTypeInterface instance |
| `packed_op` | TransformHandleTypeInterface instance |
 `packed_op` |`packed_op` TransformHandleTypeInterface instance |

---

### `transform.structured.pack` (transform::PackOp) [¶](#transformstructuredpack-transformpackop)

`transform.structured.pack`
[¶](#transformstructuredpack-transformpackop)

Syntax:

```
operation ::= `transform.structured.pack` $target
              `packed_sizes` `=` custom<DynamicIndexList>($packed_sizes,
              $static_packed_sizes)
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.pack` $target
`packed_sizes` `=` custom<DynamicIndexList>($packed_sizes,
$static_packed_sizes)
attr-dict
`:` functional-type(operands, results) ``

Pack a LinalgOp by applying a data tiling transformation on the op and
packing the operands according to the `packed_sizes` specification.

`packed_sizes`

Iterator dimensions are tiled in their canonical order in the op spec.
Operands are packed according to the same canonical order of the op iterator
dimensions.

Specifying a packed size of 0 for an iterator removes it from consideration
for packing.

`linalg.pack` (resp. `linalg.unpack`) operations are inserted for the operands
(resp. results) that need to be packed (resp. unpacked) according to the
`packed_sizes` specification.

`linalg.pack`
`linalg.unpack`
`packed_sizes`

---

#### Example [¶](#example-2)

[¶](#example-2)

Consider a `linalg.matmul` with indexing maps:

`linalg.matmul`

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

 `// M N K M K
// affine_map<(d0, d1, d2) -> (d0, d2)>
// K N
// affine_map<(d0, d1, d2) -> (d2, d1)>
// M N
// affine_map<(d0, d1, d2) -> (d0, d1)>
%0 = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
outs( %C: tensor<?x?xf32>)`

Specifying packed\_sizes [2, 3, 4] results in tiling the iterator dimensions
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

 `// M N K m n k M K m k
// affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// K N n k
// affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
// M N m n
// affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
%0 = linalg.generic_representing_some_higher_d_matmul
ins(%A, %B: tensor<?x?x2x4xf32>, tensor<?x?x4x3xf32>)
outs( %C: tensor<?x?x2x3xf32>)`

In particular, note that the second operand `B` has shape `KxNxnxk` (and not
`KxNxkxn` as one could expect by looking **only** at the operand).

`B`
`KxNxnxk`
`KxNxkxn`
**only**

Other layouts can be obtained unsurprisingly from this canonical
transformation by composing the resulting operation with a
`transform.structured.pack_transpose` op.
This composition allows separating concerns and composes better compared
to adding additional permutation attributes to this transform op.

`transform.structured.pack_transpose`

---

#### Return modes [¶](#return-modes-57)

[¶](#return-modes-57)

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definite failure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-71)

[¶](#attributes-71)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_packed_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_packed_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_packed_sizes` |`static_packed_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

---

#### Operands: [¶](#operands-106)

[¶](#operands-106)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `packed_sizes` | variadic of TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `packed_sizes` | variadic of TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `packed_sizes` | variadic of TransformHandleTypeInterface instance |
 `packed_sizes` |`packed_sizes` variadic of TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-80)

[¶](#results-80)

| Result | Description |
| --- | --- |
| `packed_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `packed_op` | TransformHandleTypeInterface instance |
| `packed_op` | TransformHandleTypeInterface instance |
 `packed_op` |`packed_op` TransformHandleTypeInterface instance |

---

### `transform.structured.pack_transpose` (transform::PackTransposeOp) [¶](#transformstructuredpack_transpose-transformpacktransposeop)

`transform.structured.pack_transpose`
[¶](#transformstructuredpack_transpose-transformpacktransposeop)

Syntax:

```
operation ::= `transform.structured.pack_transpose` $target_pack_or_un_pack_op
              `with_compute_op` `(` $target_linalg_op `)`
              (`outer_perm` `=` $outer_perm^ )?
              (`inner_perm` `=` $inner_perm^ )?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.pack_transpose` $target_pack_or_un_pack_op
`with_compute_op` `(` $target_linalg_op `)`
(`outer_perm` `=` $outer_perm^ )?
(`inner_perm` `=` $inner_perm^ )?
attr-dict
`:` functional-type(operands, results) ``

Apply a transposition to a single `linalg.pack` (resp. `linalg.unpack`) and
update the `linalg.generic` op that consumes (resp. produces) the operation.

`linalg.pack`
`linalg.unpack`
`linalg.generic`

This transform allows composing a simple `structured.pack` with additional
transpositions to e.g. match the data format required by a specific library
call or ISA instruction.

`structured.pack`

The transpose spec must specify at least one of `outer_perm` or `inner_perm`
attributes, which will act upon the `outer_dims_perm` or `inner_dims_pos` of
the specified `linalg.pack` or `linalg.unpack` op.

`outer_perm`
`inner_perm`
`outer_dims_perm`
`inner_dims_pos`
`linalg.pack`
`linalg.unpack`

If the `target` of this op is a `linalg.pack` then a new `tensor.empty` will
be created along with transposed versions of the `linalg.pack` and the
consuming `linalg.generic`, which is expected to be the sole consumer.

`target`
`linalg.pack`
`tensor.empty`
`linalg.pack`
`linalg.generic`

If the `target` of this op is a `linalg.unpack` then the whole pack / compute
/ unpack chain will be transposed and transposed clones of `linalg.pack`,
the consuming `linalg.generic` and the tail `linalg.pack` will be created.

`target`
`linalg.unpack`
`linalg.pack`
`linalg.generic`
`linalg.pack`

---

#### Return modes [¶](#return-modes-58)

[¶](#return-modes-58)

This operation targets a single `linalg.pack` / `linalg.unpack` op and a
single matching `linalg.generic` that consumes / produces the op. Otherwise,
it produces a silenceableFailure.

`linalg.pack`
`linalg.unpack`
`linalg.generic`

This operation may produce a silenceableFailure if the transpose spec is
ill-formed (i.e. `outer_perm` or `inner_perm` are not permutations of the
proper rank) or if the transposition of all involved operations fails for any
reason.

`outer_perm`
`inner_perm`

This operation returns 3 handles, one to the transformed LinalgOp, one to
the transformed `linalg.pack` and one to the transformed `linalg.unpack`.
The last handle for `linalg.unpack` is empty if `target_pack_or_unpack_op`
was not itself a `linalg.unpack`.

`linalg.pack`
`linalg.unpack`
`linalg.unpack`
`target_pack_or_unpack_op`
`linalg.unpack`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-72)

[¶](#attributes-72)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `outer_perm` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `inner_perm` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `outer_perm` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `outer_perm` |`outer_perm` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `inner_perm` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `inner_perm` |`inner_perm` ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

---

#### Operands: [¶](#operands-107)

[¶](#operands-107)

| Operand | Description |
| --- | --- |
| `target_pack_or_un_pack_op` | TransformHandleTypeInterface instance |
| `target_linalg_op` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target_pack_or_un_pack_op` | TransformHandleTypeInterface instance |
| `target_linalg_op` | TransformHandleTypeInterface instance |
| `target_pack_or_un_pack_op` | TransformHandleTypeInterface instance |
 `target_pack_or_un_pack_op` |`target_pack_or_un_pack_op` TransformHandleTypeInterface instance || `target_linalg_op` | TransformHandleTypeInterface instance |
 `target_linalg_op` |`target_linalg_op` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-81)

[¶](#results-81)

| Result | Description |
| --- | --- |
| `packed_op` | TransformHandleTypeInterface instance |
| `pack_op` | TransformHandleTypeInterface instance |
| `un_pack_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `packed_op` | TransformHandleTypeInterface instance |
| `pack_op` | TransformHandleTypeInterface instance |
| `un_pack_op` | TransformHandleTypeInterface instance |
| `packed_op` | TransformHandleTypeInterface instance |
 `packed_op` |`packed_op` TransformHandleTypeInterface instance || `pack_op` | TransformHandleTypeInterface instance |
 `pack_op` |`pack_op` TransformHandleTypeInterface instance || `un_pack_op` | TransformHandleTypeInterface instance |
 `un_pack_op` |`un_pack_op` TransformHandleTypeInterface instance |

---

### `transform.structured.pad` (transform::PadOp) [¶](#transformstructuredpad-transformpadop)

`transform.structured.pad`
[¶](#transformstructuredpad-transformpadop)

Syntax:

```
operation ::= `transform.structured.pad` $target
              (`pad_to_multiple_of` custom<DynamicIndexList>($pad_to_multiple_of, $static_pad_to_multiple_of)^)?
              (`use_prescribed_tensor_shapes` $use_prescribed_tensor_shapes^)?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.pad` $target
(`pad_to_multiple_of` custom<DynamicIndexList>($pad_to_multiple_of, $static_pad_to_multiple_of)^)?
(`use_prescribed_tensor_shapes` $use_prescribed_tensor_shapes^)?
attr-dict
`:` functional-type(operands, results) ``

Pads the operations pointed to by the target handle using the options
provides as operation attributes. The operation returns a handle to the
padded operation and to the padding operation (“tensor.pad”).

To preserve tensor SSA use-def chains, the unpadded result is copied back to
the original destination tensor of the targeted op. The op that copies back
the result can be customized with `copy_back_op`:

`copy_back_op`

* “bufferization.materialize\_in\_destination” (default)
* “linalg.copy”
* “none” (no copy back)

- “bufferization.materialize\_in\_destination” (default)
- “linalg.copy”
- “none” (no copy back)

---

#### Return modes [¶](#return-modes-59)

[¶](#return-modes-59)

This operation ignores non-Linalg ops and drops them in the return.
This operation may produce a definite failure if the padding fails for any
reason.

If all the operations referred to by the `target` handle pad
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-73)

[¶](#attributes-73)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `padding_values` | ::mlir::ArrayAttr | array attribute |
| `padding_dimensions` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `static_pad_to_multiple_of` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `nofold_flags` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `transpose_paddings` | ::mlir::ArrayAttr | array of arrays of i64 |
| `copy_back_op` | ::mlir::StringAttr | string attribute |
| `use_prescribed_tensor_shapes` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `padding_values` | ::mlir::ArrayAttr | array attribute |
 `padding_values` |`padding_values` ::mlir::ArrayAttr | array attribute || `padding_dimensions` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `padding_dimensions` |`padding_dimensions` ::mlir::ArrayAttr | 64-bit integer array attribute || `static_pad_to_multiple_of` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_pad_to_multiple_of` |`static_pad_to_multiple_of` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `nofold_flags` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `nofold_flags` |`nofold_flags` ::mlir::ArrayAttr | 64-bit integer array attribute || `transpose_paddings` | ::mlir::ArrayAttr | array of arrays of i64 |
 `transpose_paddings` |`transpose_paddings` ::mlir::ArrayAttr | array of arrays of i64 || `copy_back_op` | ::mlir::StringAttr | string attribute |
 `copy_back_op` |`copy_back_op` ::mlir::StringAttr | string attribute || `use_prescribed_tensor_shapes` | ::mlir::UnitAttr | unit attribute |
 `use_prescribed_tensor_shapes` |`use_prescribed_tensor_shapes` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-108)

[¶](#operands-108)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `pad_to_multiple_of` | variadic of transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `pad_to_multiple_of` | variadic of transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `pad_to_multiple_of` | variadic of transform any param type or any handle type |
 `pad_to_multiple_of` |`pad_to_multiple_of` variadic of transform any param type or any handle type |

---

#### Results: [¶](#results-82)

[¶](#results-82)

| Result | Description |
| --- | --- |
| `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |
| `copy` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |
| `copy` | TransformHandleTypeInterface instance |
| `padded` | TransformHandleTypeInterface instance |
 `padded` |`padded` TransformHandleTypeInterface instance || `pad` | TransformHandleTypeInterface instance |
 `pad` |`pad` TransformHandleTypeInterface instance || `copy` | TransformHandleTypeInterface instance |
 `copy` |`copy` TransformHandleTypeInterface instance |

---

### `transform.structured.pad_tiling_interface` (transform::PadTilingInterfaceOp) [¶](#transformstructuredpad_tiling_interface-transformpadtilinginterfaceop)

`transform.structured.pad_tiling_interface`
[¶](#transformstructuredpad_tiling_interface-transformpadtilinginterfaceop)

Syntax:

```
operation ::= `transform.structured.pad_tiling_interface` $target
              `to`
              (`padding_sizes` custom<DynamicIndexList>($padding_sizes, $static_padding_sizes)^)?
              (`pad_to_multiple_of` $pad_to_multiple_of^)?
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.pad_tiling_interface` $target
`to`
(`padding_sizes` custom<DynamicIndexList>($padding_sizes, $static_padding_sizes)^)?
(`pad_to_multiple_of` $pad_to_multiple_of^)?
attr-dict
`:` functional-type(operands, results) ``

Pads the **iteration domain** of the operations pointed to by the target
handle using the options provided as operation attributes. Padding the
iteration domain induces a padding of the operands that is consistent
across the op semantics and, unlike for simple elementwise ops, may not be
trivially deducible or specifiable on operands only (e.g. convolutions).
Currently, only a limited set of projected permutation maps are supported.

**iteration domain**

The specification of `padding_sizes` follows that of `tile_sizes` during
tiling: the value “0” on a particular iterator encode “no padding”. Like in
the case of tiling, an automatic completion by 0 to the operation rank
occurs.

`padding_sizes`
`tile_sizes`

This transformation returns a handle to the padded operation and to the
padding operation (“tensor.pad”).

TODO: in the future this should be moved out of a specific Linalg
implementation file and into a more general “Structured” file.

---

#### Return modes [¶](#return-modes-60)

[¶](#return-modes-60)

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

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-74)

[¶](#attributes-74)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `padding_values` | ::mlir::ArrayAttr | array attribute |
| `static_padding_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `pad_to_multiple_of` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `padding_values` | ::mlir::ArrayAttr | array attribute |
 `padding_values` |`padding_values` ::mlir::ArrayAttr | array attribute || `static_padding_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_padding_sizes` |`static_padding_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `pad_to_multiple_of` | ::mlir::UnitAttr | unit attribute |
 `pad_to_multiple_of` |`pad_to_multiple_of` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-109)

[¶](#operands-109)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `padding_sizes` | variadic of transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `padding_sizes` | variadic of transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `padding_sizes` | variadic of transform any param type or any handle type |
 `padding_sizes` |`padding_sizes` variadic of transform any param type or any handle type |

---

#### Results: [¶](#results-83)

[¶](#results-83)

| Result | Description |
| --- | --- |
| `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `padded` | TransformHandleTypeInterface instance |
| `pad` | TransformHandleTypeInterface instance |
| `padded` | TransformHandleTypeInterface instance |
 `padded` |`padded` TransformHandleTypeInterface instance || `pad` | TransformHandleTypeInterface instance |
 `pad` |`pad` TransformHandleTypeInterface instance |

---

### `transform.structured.promote` (transform::PromoteOp) [¶](#transformstructuredpromote-transformpromoteop)

`transform.structured.promote`
[¶](#transformstructuredpromote-transformpromoteop)

Syntax:

```
operation ::= `transform.structured.promote` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

`` operation ::= `transform.structured.promote` $target attr-dict `:`
custom<SemiFunctionType>(type($target), type($transformed), "false") ``

Promotes the specified operands of the target into a separate memory buffer.

At this point, this transform does not allow customizing alloc/dealloc
functions nor the behavior on copy in/out operations.

---

#### Return modes [¶](#return-modes-61)

[¶](#return-modes-61)

This operation applies to a single Linalg op that satisfies the
`promoteSubviewsPrecondition`, otherwise it fails.

`promoteSubviewsPrecondition`

If the operations referred to by the `target` handle promote
properly, the transform succeeds.

`target`

When successful, the return handle points to the $target operation that
was modified inplace.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-75)

[¶](#attributes-75)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `operands_to_promote` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `use_full_tile_buffers` | ::mlir::ArrayAttr | 1-bit boolean array attribute |
| `use_full_tiles_by_default` | ::mlir::UnitAttr | unit attribute |
| `use_original_subview_size` | ::mlir::UnitAttr | unit attribute |
| `use_alloca` | ::mlir::UnitAttr | unit attribute |
| `memory_space` | ::mlir::Attribute | any attribute |
| `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |
| `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `operands_to_promote` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `operands_to_promote` |`operands_to_promote` ::mlir::ArrayAttr | 64-bit integer array attribute || `use_full_tile_buffers` | ::mlir::ArrayAttr | 1-bit boolean array attribute |
 `use_full_tile_buffers` |`use_full_tile_buffers` ::mlir::ArrayAttr | 1-bit boolean array attribute || `use_full_tiles_by_default` | ::mlir::UnitAttr | unit attribute |
 `use_full_tiles_by_default` |`use_full_tiles_by_default` ::mlir::UnitAttr | unit attribute || `use_original_subview_size` | ::mlir::UnitAttr | unit attribute |
 `use_original_subview_size` |`use_original_subview_size` ::mlir::UnitAttr | unit attribute || `use_alloca` | ::mlir::UnitAttr | unit attribute |
 `use_alloca` |`use_alloca` ::mlir::UnitAttr | unit attribute || `memory_space` | ::mlir::Attribute | any attribute |
 `memory_space` |`memory_space` ::mlir::Attribute | any attribute || `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |
 `mapping` |`mapping` ::mlir::ArrayAttr | Device Mapping array attribute || `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `alignment` |`alignment` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-110)

[¶](#operands-110)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-84)

[¶](#results-84)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.promote_tensor` (transform::PromoteTensorOp) [¶](#transformstructuredpromote_tensor-transformpromotetensorop)

`transform.structured.promote_tensor`
[¶](#transformstructuredpromote_tensor-transformpromotetensorop)

*Request a tensor value to live in a specific memory space after bufferization*

*Request a tensor value to live in a specific memory space after bufferization*

Syntax:

```
operation ::= `transform.structured.promote_tensor` (`to` $memory_space^)? $tensor attr-dict `:` type($tensor)
```

`` operation ::= `transform.structured.promote_tensor` (`to` $memory_space^)? $tensor attr-dict `:` type($tensor) ``

Requests that a tensor value lives in a specific memory space for its
lifetime. This is achieved by allocating a new tensor in the desired
memory space with `bufferization.alloc_tensor` and optionally materializing
the source value into that allocation with
`bufferization.materialize_in_destination`. All uses of the original value
are then redirected to the promoted value.

`bufferization.alloc_tensor`
`bufferization.materialize_in_destination`

The generated code for promoting tensor value %0 resembles the following:

%1 = bufferization.alloc\_tensor(<dynamic dims of %0>)
{ memory\_space = memory\_space }
// Note: the materialization is omitted if %0 is never read and is only
// written into (i.e., it behaves as a result tensor).
%2 = bufferization.materialize\_in\_destination %0 in %1
// …
<all users of %0 now use %2 instead>

Deallocation is not handled by this transform.

Return modes:

* Produces a silenceable failure if the given handle does not point to
  tensor-typed values.
* Succeeds otherwise and returns a handle to the promoted value(s), i.e.,
  the result of materialization if present and the allocation otherwise.

- Produces a silenceable failure if the given handle does not point to
  tensor-typed values.
- Succeeds otherwise and returns a handle to the promoted value(s), i.e.,
  the result of materialization if present and the allocation otherwise.

Traits: `SameOperandsAndResultType`

`SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

`InferTypeOpInterface`
`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-76)

[¶](#attributes-76)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `memory_space` | ::mlir::Attribute | any attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `memory_space` | ::mlir::Attribute | any attribute |
 `memory_space` |`memory_space` ::mlir::Attribute | any attribute |

---

#### Operands: [¶](#operands-111)

[¶](#operands-111)

| Operand | Description |
| --- | --- |
| `tensor` | TransformValueHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `tensor` | TransformValueHandleTypeInterface instance |
| `tensor` | TransformValueHandleTypeInterface instance |
 `tensor` |`tensor` TransformValueHandleTypeInterface instance |

---

#### Results: [¶](#results-85)

[¶](#results-85)

| Result | Description |
| --- | --- |
| `promoted` | TransformValueHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `promoted` | TransformValueHandleTypeInterface instance |
| `promoted` | TransformValueHandleTypeInterface instance |
 `promoted` |`promoted` TransformValueHandleTypeInterface instance |

---

### `transform.structured.replace` (transform::ReplaceOp) [¶](#transformstructuredreplace-transformreplaceop)

`transform.structured.replace`
[¶](#transformstructuredreplace-transformreplaceop)

Syntax:

```
operation ::= `transform.structured.replace` $target attr-dict-with-keyword regions `:`
              custom<SemiFunctionType>(type($target), type($replacement), "false")
```

`` operation ::= `transform.structured.replace` $target attr-dict-with-keyword regions `:`
custom<SemiFunctionType>(type($target), type($replacement), "false") ``

Replace all `target` payload ops with the single op that is contained in
this op’s region. All targets must have zero arguments and must be isolated
from above.

`target`

This op is for debugging/experiments only.

---

#### Return modes [¶](#return-modes-62)

[¶](#return-modes-62)

This operation consumes the `target` handle.

`target`

Traits: `HasOnlyGraphRegion`, `IsolatedFromAbove`, `NoTerminator`, `ReportTrackingListenerFailuresOpTrait`, `SingleBlock`

`HasOnlyGraphRegion`
`IsolatedFromAbove`
`NoTerminator`
`ReportTrackingListenerFailuresOpTrait`
`SingleBlock`

Interfaces: `MemoryEffectOpInterface`, `RegionKindInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`RegionKindInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-112)

[¶](#operands-112)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-86)

[¶](#results-86)

| Result | Description |
| --- | --- |
| `replacement` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `replacement` | TransformHandleTypeInterface instance |
| `replacement` | TransformHandleTypeInterface instance |
 `replacement` |`replacement` TransformHandleTypeInterface instance |

---

### `transform.structured.rewrite_in_destination_passing_style` (transform::RewriteInDestinationPassingStyleOp) [¶](#transformstructuredrewrite_in_destination_passing_style-transformrewriteindestinationpassingstyleop)

`transform.structured.rewrite_in_destination_passing_style`
[¶](#transformstructuredrewrite_in_destination_passing_style-transformrewriteindestinationpassingstyleop)

Syntax:

```
operation ::= `transform.structured.rewrite_in_destination_passing_style` $target attr-dict
              `:` functional-type($target, results)
```

`` operation ::= `transform.structured.rewrite_in_destination_passing_style` $target attr-dict
`:` functional-type($target, results) ``

Rewrite a supported tensor operation that is not in destination-passing style
into a form that is in destination-passing style.
Currently supported operations are:

* tensor.pad
* tensor.generate
* tensor.from\_elements
  This dichotomy hints at a future interface, for now the implementation just
  switches between different implementation.

- tensor.pad
- tensor.generate
- tensor.from\_elements
  This dichotomy hints at a future interface, for now the implementation just
  switches between different implementation.

---

#### Return modes [¶](#return-modes-63)

[¶](#return-modes-63)

This operation ignores non-unsupported ops and drops them from the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.
The return handle points to a subset of successfully produced operations:

`target`

* `tensor.pad` case, the returned handle points to the tensor.insert\_slice.
* `tensor.generate` case, the returned handle points to the linalg.generic.
* `tensor.from_elements` case, the returned handle points to the last
  `tensor.insert`.

- `tensor.pad` case, the returned handle points to the tensor.insert\_slice.
`tensor.pad`- `tensor.generate` case, the returned handle points to the linalg.generic.
`tensor.generate`- `tensor.from_elements` case, the returned handle points to the last
  `tensor.insert`.
`tensor.from_elements`
`tensor.insert`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-113)

[¶](#operands-113)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-87)

[¶](#results-87)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.scalarize` (transform::ScalarizeOp) [¶](#transformstructuredscalarize-transformscalarizeop)

`transform.structured.scalarize`
[¶](#transformstructuredscalarize-transformscalarizeop)

Syntax:

```
operation ::= `transform.structured.scalarize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($result), "false")
```

`` operation ::= `transform.structured.scalarize` $target attr-dict `:`
custom<SemiFunctionType>(type($target), type($result), "false") ``

Indicates that ops of a specific kind in the given function should be
scalarized (i.e. their dynamic dimensions tiled by 1).

---

#### Return modes: [¶](#return-modes-64)

[¶](#return-modes-64)

This operation ignores non-Linalg ops and drops them in the return.
This operation produces definite failure if the scalarization fails for any
reason.
If all the operations referred to by the `target` handle scalarize
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure.

`target`

The return handle points to only the subset of successfully produced
tiled-by-1 operations, which can be empty.

This operation does not return handles to the tiled loop.
We make this design choice because it is hard to know ahead of time the
number of loops that will be produced (it depends on the number of dynamic
dimensions after multiple transformations have been applied).
Loops can always be recovered by navigating from the tiled operations if
needed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-114)

[¶](#operands-114)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-88)

[¶](#results-88)

| Result | Description |
| --- | --- |
| `result` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `result` | TransformHandleTypeInterface instance |
| `result` | TransformHandleTypeInterface instance |
 `result` |`result` TransformHandleTypeInterface instance |

---

### `transform.structured.specialize` (transform::SpecializeOp) [¶](#transformstructuredspecialize-transformspecializeop)

`transform.structured.specialize`
[¶](#transformstructuredspecialize-transformspecializeop)

Syntax:

```
operation ::= `transform.structured.specialize` $target attr-dict `:`
              custom<SemiFunctionType>(type($target), type($transformed), "false")
```

`` operation ::= `transform.structured.specialize` $target attr-dict `:`
custom<SemiFunctionType>(type($target), type($transformed), "false") ``

Transforms a generic operation into the equivalent named form.

---

#### Return modes [¶](#return-modes-65)

[¶](#return-modes-65)

This operation ignores non-Linalg ops and drops them in the return. If all
the operations referred to by the `target` handle specialize, the transform
succeeds; otherwise, the operation produces a silenceable failure. The return
handle points to only the subset of successfully produced equivalent named
operations, which can be empty or contain the original ops if they were already
in named form. The supported specialization to named Linalg operations are:

`target`

* linalg.copy of any rank.

- linalg.copy of any rank.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-115)

[¶](#operands-115)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-89)

[¶](#results-89)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.split` (transform::SplitOp) [¶](#transformstructuredsplit-transformsplitop)

`transform.structured.split`
[¶](#transformstructuredsplit-transformsplitop)

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

`target`
`ShapedType::kDynamic`

The operation consumes the target handle, but preserves the chunk size
handle if provided. Without the `multiway` attribute, it produces a
new handle that is a list of the two parts of the structured op after
splitting, whose lower index part corresponding to the part with lower
iteration space indices.

`multiway`

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

`multiway`
`target`
`static_chunk_sizes`
`dynamic_chunk_sizes`
`multiway`

As the result handle is most of time a list, an `transform.split_handle`
is needed to access individual handle.

`transform.split_handle`

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-77)

[¶](#attributes-77)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `static_chunk_sizes` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `multiway` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `dimension` |`dimension` ::mlir::IntegerAttr | 64-bit signless integer attribute || `static_chunk_sizes` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `static_chunk_sizes` |`static_chunk_sizes` ::mlir::IntegerAttr | 64-bit signless integer attribute || `multiway` | ::mlir::UnitAttr | unit attribute |
 `multiway` |`multiway` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-116)

[¶](#operands-116)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_chunk_sizes` | transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `dynamic_chunk_sizes` | transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `dynamic_chunk_sizes` | transform any param type or any handle type |
 `dynamic_chunk_sizes` |`dynamic_chunk_sizes` transform any param type or any handle type |

---

#### Results: [¶](#results-90)

[¶](#results-90)

| Result | Description |
| --- | --- |
| `split_list` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `split_list` | TransformHandleTypeInterface instance |
| `split_list` | TransformHandleTypeInterface instance |
 `split_list` |`split_list` TransformHandleTypeInterface instance |

---

### `transform.structured.split_reduction` (transform::SplitReductionOp) [¶](#transformstructuredsplit_reduction-transformsplitreductionop)

`transform.structured.split_reduction`
[¶](#transformstructuredsplit_reduction-transformsplitreductionop)

Syntax:

```
operation ::= `transform.structured.split_reduction` $target attr-dict `:`functional-type(operands, results)
```

`` operation ::= `transform.structured.split_reduction` $target attr-dict `:`functional-type(operands, results) ``

Indicates that the given `target` op should be transformed with the
`splitReduction` transformation and split factor provided as attribute.

`target`
`splitReduction`

The `splitReduction` transformation splits the first single linalg op
reduction into a parallel and reduction dimension.
A new `linalg.generic` op is created to perform the rest of the reduction.

`splitReduction`
`linalg.generic`

The transformation supports different configurations attributes:

* split\_factor: the factor by which to split (i.e. the size of the
  remaining reduction after splitting).
* insert\_split\_dimension: the dimension in the temporary tensor into
  which the new parallel dimension is inserted.
* inner\_parallel: specifies whether the parallel dimension is before or
  after the reduction dimension in the splitting op.
* use\_scaling\_algorithm: whether to use a scaling based formulation that
  does not create an ExpandShapeOp (default: do not use scaling)
* use\_alloc: whether to use an alloc op to allocate the temporary
  tensor (default: do not use alloc op)

- split\_factor: the factor by which to split (i.e. the size of the
  remaining reduction after splitting).
- insert\_split\_dimension: the dimension in the temporary tensor into
  which the new parallel dimension is inserted.
- inner\_parallel: specifies whether the parallel dimension is before or
  after the reduction dimension in the splitting op.
- use\_scaling\_algorithm: whether to use a scaling based formulation that
  does not create an ExpandShapeOp (default: do not use scaling)
- use\_alloc: whether to use an alloc op to allocate the temporary
  tensor (default: do not use alloc op)

---

#### Return modes [¶](#return-modes-66)

[¶](#return-modes-66)

This operation ignores non-Linalg ops and drops them in the return.
This operation produces a definite failure if the splitting fails for any
reason.

If all the operations referred to by the `target` handle split
properly, the transform succeeds. Otherwise the transform produces a
silenceable failure. The 4 returned handles points to only the subset of
successfully produced computational operations, which can all be empty.
This 4 returned handles point to:

`target`

* the init op (or tensor\_alloc op if use\_alloc = true),
* the fill op used to initialize the neutral element,
* the split op and
* the result-combining op.

- the init op (or tensor\_alloc op if use\_alloc = true),
- the fill op used to initialize the neutral element,
- the split op and
- the result-combining op.

---

#### Example (default: `use_scaling_algorithm = false, use_alloc = false`): [¶](#example-default-use_scaling_algorithm--false-use_alloc--false)

`use_scaling_algorithm = false, use_alloc = false`
[¶](#example-default-use_scaling_algorithm--false-use_alloc--false)

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

 `%r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
affine_map<(d0) -> ()>],
iterator_types = ["reduction"]}
ins(%in : tensor<32xf32>)
outs(%out : tensor<f32>) {
^bb0(%arg1: f32, %arg2: f32):
%y = arith.addf %arg1, %arg2 : f32
linalg.yield %y : f32
} -> tensor<f32>`

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

 `%cst = arith.constant 0.000000e+00 : f32
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
} -> tensor<f32>`

---

#### Example (`use_scaling_algorithm = true, use_alloc = true`): [¶](#example-use_scaling_algorithm--true-use_alloc--true)

`use_scaling_algorithm = true, use_alloc = true`
[¶](#example-use_scaling_algorithm--true-use_alloc--true)

Instead of introducing an ExpandShapeOp, this scaling-based implementation
rewrites a reduction dimension `k` into `k * split_factor + kk`.
The dimension `kk` is added as an extra parallel dimension to the
intermediate output tensor at position `insert_split_dimension`.

`k`
`k * split_factor + kk`
`kk`
`insert_split_dimension`

Consider a minimal example where `k` is reduced:
O(i, j) += I(i, j, k)
Assume i=3, j=5, k=128, split\_factor=16 and insert\_split\_dimension=0.
The compute is rewritten as:
a. O\_i(kk, i, j) += I(i, j, 16 \* k + kk)
b. O(i, j) += O\_i(kk, i, j)
The intermediate tensor O\_i is of shape (128/16)x3x5 == 8x3x5.

`k`

---

#### Example: [¶](#example-3)

[¶](#example-3)

```
 %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
   outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
```

 `%0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>`

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

 `#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
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
return %4 : tensor<16x32xf32>`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-78)

[¶](#attributes-78)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `split_factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `insert_split_dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `inner_parallel` | ::mlir::UnitAttr | unit attribute |
| `use_scaling_algorithm` | ::mlir::UnitAttr | unit attribute |
| `use_alloc` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `split_factor` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `split_factor` |`split_factor` ::mlir::IntegerAttr | 64-bit signless integer attribute || `insert_split_dimension` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `insert_split_dimension` |`insert_split_dimension` ::mlir::IntegerAttr | 64-bit signless integer attribute || `inner_parallel` | ::mlir::UnitAttr | unit attribute |
 `inner_parallel` |`inner_parallel` ::mlir::UnitAttr | unit attribute || `use_scaling_algorithm` | ::mlir::UnitAttr | unit attribute |
 `use_scaling_algorithm` |`use_scaling_algorithm` ::mlir::UnitAttr | unit attribute || `use_alloc` | ::mlir::UnitAttr | unit attribute |
 `use_alloc` |`use_alloc` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-117)

[¶](#operands-117)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-91)

[¶](#results-91)

| Result | Description |
| --- | --- |
| `init_or_alloc_op` | TransformHandleTypeInterface instance |
| `fill_op` | TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `init_or_alloc_op` | TransformHandleTypeInterface instance |
| `fill_op` | TransformHandleTypeInterface instance |
| `split_linalg_op` | TransformHandleTypeInterface instance |
| `combining_linalg_op` | TransformHandleTypeInterface instance |
| `init_or_alloc_op` | TransformHandleTypeInterface instance |
 `init_or_alloc_op` |`init_or_alloc_op` TransformHandleTypeInterface instance || `fill_op` | TransformHandleTypeInterface instance |
 `fill_op` |`fill_op` TransformHandleTypeInterface instance || `split_linalg_op` | TransformHandleTypeInterface instance |
 `split_linalg_op` |`split_linalg_op` TransformHandleTypeInterface instance || `combining_linalg_op` | TransformHandleTypeInterface instance |
 `combining_linalg_op` |`combining_linalg_op` TransformHandleTypeInterface instance |

---

### `transform.structured.tile_reduction_using_for` (transform::TileReductionUsingForOp) [¶](#transformstructuredtile_reduction_using_for-transformtilereductionusingforop)

`transform.structured.tile_reduction_using_for`
[¶](#transformstructuredtile_reduction_using_for-transformtilereductionusingforop)

Syntax:

```
operation ::= `transform.structured.tile_reduction_using_for` $target
              (`reduction_dims` `=` $reduction_dims^)?
              `by` `tile_sizes` `=` $tile_sizes
              attr-dict
              `:` functional-type(operands, results)
```

`` operation ::= `transform.structured.tile_reduction_using_for` $target
(`reduction_dims` `=` $reduction_dims^)?
`by` `tile_sizes` `=` $tile_sizes
attr-dict
`:` functional-type(operands, results) ``

Indicates that the given `target` op should be transformed with the
`tileReduction` transformation with the tile size provided as attribute.

`target`
`tileReduction`

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates nested
loops with a parallel version of `target` op inside. The parallel op
dimensions are less or equal to the tile size passed by user.
After the loop a merge operation is created to do a final reduction with the
partial reductions.
The initial tensor always uses the tile size dimension. This may overallocate
if the tile size is greater than the reduction dimension.

`target`
`target`

---

#### Return modes [¶](#return-modes-67)

[¶](#return-modes-67)

Returns 4 handles associated with (in order):

* the fill op used to initialize the neutral element,
* the parallel tiled op and
* the result-combining op,
* the parent `for` op.

- the fill op used to initialize the neutral element,
- the parallel tiled op and
- the result-combining op,
- the parent `for` op.
`for`

The `reduction_dims` can be used to specify the subset of reduction dimensions
of the operation to tile. If left unspecified, all reduction dimensions are
tiled.

`reduction_dims`

---

#### Example: [¶](#example-4)

[¶](#example-4)

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

 `%red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
affine_map<(d0, d1) -> (d0)>],
iterator_types = ["parallel", "reduction"]}
ins(%arg0 : tensor<?x?xf32>)
outs(%out : tensor<?xf32>) {
^bb0(%arg7: f32, %arg9: f32):
%1 = arith.addf %arg7, %arg9 : f32
linalg.yield %1 : f32
} -> tensor<?xf32>
return %red : tensor<?xf32>`

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

 `%0 = tensor.empty(%dim_1) : tensor<?x5xf32>
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
} -> tensor<?xf32>`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-79)

[¶](#attributes-79)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `reduction_dims` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `tile_sizes` | ::mlir::ArrayAttr | 64-bit integer array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `reduction_dims` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `reduction_dims` |`reduction_dims` ::mlir::ArrayAttr | 64-bit integer array attribute || `tile_sizes` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `tile_sizes` |`tile_sizes` ::mlir::ArrayAttr | 64-bit integer array attribute |

---

#### Operands: [¶](#operands-118)

[¶](#operands-118)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-92)

[¶](#results-92)

| Result | Description |
| --- | --- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `for_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `for_op` | TransformHandleTypeInterface instance |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
 `fill_op` |`fill_op` variadic of TransformHandleTypeInterface instance || `split_op` | TransformHandleTypeInterface instance |
 `split_op` |`split_op` TransformHandleTypeInterface instance || `combining_op` | TransformHandleTypeInterface instance |
 `combining_op` |`combining_op` TransformHandleTypeInterface instance || `for_op` | TransformHandleTypeInterface instance |
 `for_op` |`for_op` TransformHandleTypeInterface instance |

---

### `transform.structured.tile_reduction_using_forall` (transform::TileReductionUsingForallOp) [¶](#transformstructuredtile_reduction_using_forall-transformtilereductionusingforallop)

`transform.structured.tile_reduction_using_forall`
[¶](#transformstructuredtile_reduction_using_forall-transformtilereductionusingforallop)

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

`` operation ::= `transform.structured.tile_reduction_using_forall` $target
(`reduction_dims` `=` $reduction_dims^)?
`by`
(`num_threads` `=` $num_threads^)?
(`tile_sizes` `=` $tile_sizes^)?
(`mapping` `=` $mapping^)?
attr-dict
`:` functional-type(operands, results) ``

Tile a PartialReductionOpInterface op to a tiled `scf.forall` doing
partial reduction.

`scf.forall`

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates a
`scf.forall` loops with the number threads given by `num_threads`.
The op is tiled op with a size equal to `floordiv(size, num_threads)`.
All the partial reduction value is are parallel inserted to create a new
tensor. After the loop a merge operation is created to do a final reduction
with the partial reductions tensor.
If an extra `tile_sizes` parameter is passed the tiles are cyclically
distributed on the threads of the `scf.foralls` loop.

`target`
`scf.forall`
`num_threads`
`floordiv(size, num_threads)`
`tile_sizes`
`scf.foralls`

---

#### Return modes [¶](#return-modes-68)

[¶](#return-modes-68)

Returns 4 handles associated with (in order):

* the fill op used to initialize the neutral element,
* the parallel tiled op and
* the result-combining op,
* the parent `forall` op.

- the fill op used to initialize the neutral element,
- the parallel tiled op and
- the result-combining op,
- the parent `forall` op.
`forall`

---

#### Example: [¶](#example-5)

[¶](#example-5)

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

 `%red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
affine_map<(d0, d1) -> (d0)>],
iterator_types = ["parallel", "reduction"]}
ins(%arg0 : tensor<?x?xf32>)
outs(%out : tensor<?xf32>) {
^bb0(%arg7: f32, %arg9: f32):
%1 = arith.addf %arg7, %arg9 : f32
linalg.yield %1 : f32
} -> tensor<?xf32>
return %red : tensor<?xf32>`

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

 `%0 = tensor.empty(%dim_1) : tensor<?x5xf32>
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
} -> tensor<?xf32>`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-80)

[¶](#attributes-80)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `reduction_dims` | ::mlir::ArrayAttr | 64-bit integer array attribute |
| `num_threads` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `reduction_dims` | ::mlir::ArrayAttr | 64-bit integer array attribute |
 `reduction_dims` |`reduction_dims` ::mlir::ArrayAttr | 64-bit integer array attribute || `num_threads` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `num_threads` |`num_threads` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `tile_sizes` |`tile_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |
 `mapping` |`mapping` ::mlir::ArrayAttr | Device Mapping array attribute |

---

#### Operands: [¶](#operands-119)

[¶](#operands-119)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-93)

[¶](#results-93)

| Result | Description |
| --- | --- |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `fill_op` | variadic of TransformHandleTypeInterface instance |
| `split_op` | TransformHandleTypeInterface instance |
| `combining_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |
| `fill_op` | variadic of TransformHandleTypeInterface instance |
 `fill_op` |`fill_op` variadic of TransformHandleTypeInterface instance || `split_op` | TransformHandleTypeInterface instance |
 `split_op` |`split_op` TransformHandleTypeInterface instance || `combining_op` | TransformHandleTypeInterface instance |
 `combining_op` |`combining_op` TransformHandleTypeInterface instance || `forall_op` | TransformHandleTypeInterface instance |
 `forall_op` |`forall_op` TransformHandleTypeInterface instance |

---

### `transform.structured.tile_using_for` (transform::TileUsingForOp) [¶](#transformstructuredtile_using_for-transformtileusingforop)

`transform.structured.tile_using_for`
[¶](#transformstructuredtile_using_for-transformtileusingforop)

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

`` operation ::= `transform.structured.tile_using_for` $target
`tile_sizes` custom<DynamicIndexList>(
$dynamic_sizes,
$static_sizes,
$scalable_sizes)
(`interchange` `=` $interchange^)?
attr-dict
`:` functional-type(operands, results) ``

Indicates that the given `target` op should be tiled with the given sizes.
This transform generates a loop nest with a smaller (“tiled”) target
operation in its body. Currently limited to LinalgOps.

`target`

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

`static_size`
`dynamic_sizes`
`static_sizes`
`ShapedType::kDynamic`
`dynamic_sizes`
`ShapedType::kDynamic`
`static_sizes`
`0`
`0`

This op returns handles to the tiled op (in the generated loop nest) and the
generated loops. The number of loops is the number of tile sizes that are
statically known to be non-zero.

---

#### Return modes [¶](#return-modes-69)

[¶](#return-modes-69)

On success, the resulting handles are associated with co-indexed lists of
tiled operations and loops around them.

This operation only supports Linalg ops and produces a silenceable failure
if the input contains any non-Linalg ops. The ops preceding it in the list
associated with the `target` handle will have been tiled.

`target`

This operation produces a silenceable failure if the `dynamic_sizes` handles
are associated with lists of payload operations of a size different than
that of the list associated with the `target` handle.

`dynamic_sizes`
`target`

If the internal implementation of tiling for any of the operations fails,
produces a definite failure.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-81)

[¶](#attributes-81)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `scalable_sizes` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_sizes` |`static_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `interchange` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `interchange` |`interchange` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `scalable_sizes` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
 `scalable_sizes` |`scalable_sizes` ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

---

#### Operands: [¶](#operands-120)

[¶](#operands-120)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `dynamic_sizes` | variadic of transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `dynamic_sizes` | variadic of transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `dynamic_sizes` | variadic of transform any param type or any handle type |
 `dynamic_sizes` |`dynamic_sizes` variadic of transform any param type or any handle type |

---

#### Results: [¶](#results-94)

[¶](#results-94)

| Result | Description |
| --- | --- |
| `tiled_linalg_op` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `tiled_linalg_op` | TransformHandleTypeInterface instance |
| `loops` | variadic of TransformHandleTypeInterface instance |
| `tiled_linalg_op` | TransformHandleTypeInterface instance |
 `tiled_linalg_op` |`tiled_linalg_op` TransformHandleTypeInterface instance || `loops` | variadic of TransformHandleTypeInterface instance |
 `loops` |`loops` variadic of TransformHandleTypeInterface instance |

---

### `transform.structured.tile_using_forall` (transform::TileUsingForallOp) [¶](#transformstructuredtile_using_forall-transformtileusingforallop)

`transform.structured.tile_using_forall`
[¶](#transformstructuredtile_using_forall-transformtileusingforallop)

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

`` operation ::= `transform.structured.tile_using_forall` $target oilist(
`num_threads` custom<PackedOrDynamicIndexList>($packed_num_threads,
$num_threads,
$static_num_threads) |
`tile_sizes` custom<PackedOrDynamicIndexList>($packed_tile_sizes,
$tile_sizes,
$static_tile_sizes))
(`(` `mapping` `=` $mapping^ `)`)? attr-dict
`:` functional-type(operands, results) ``

Tile a TilingInterface op to a tiled `scf.forall`.

`scf.forall`

Tiling is applied by either specifying `num_threads` or `tile_size`. If
`num_threads` is specified, then the tile size for each dimension `i` is
calculated dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
`num_threads` and `tile_size` can be either static index attributes or
operation handles (or a mix thereof). Operation handles must be mapped to
exactly one op that has exactly one result of index type.

`num_threads`
`tile_size`
`num_threads`
`i`
`ceilDiv(dimSize[i], num_threads[i])`
`num_threads`
`tile_size`

Static zero tile sizes indicate that the dimension is not tiled and can be
thought of as tiling by the full size of data.

It is the user’s responsibility to ensure that `num_threads/tile_sizes` is
a valid tiling specification (i.e. that only tiles parallel dimensions,
e.g. in the Linalg case). If the dimension is not parallelizable, a warning
is issued to notify the user that the generated code is not safe to
parallelize.

`num_threads/tile_sizes`

If non-empty, the `mapping` is added as an attribute to the
resulting `scf.forall`.

`mapping`
`scf.forall`

Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
threads can be an index attribute or a transform handle that is mapped to
exactly one payload op with exactly one index result.

`tile_sizes`
`num_threads`

---

#### Return modes [¶](#return-modes-70)

[¶](#return-modes-70)

This operation ignores ops that do not implement the TilingInterface and
drops them in the return.

If all the operations referred to by the `target` handle tile
successfully, the transform succeeds.
Otherwise the transform produces a silenceable failure.

`target`

The two returned handles point to only the subset of successfully produced
tiled operations, which can all be empty.

These two returned handles point to:

* the tiled op that implements TilingInterface,
* the new scf.forall op.

- the tiled op that implements TilingInterface,
- the new scf.forall op.

---

#### Example using `num_threads` [¶](#example-using-num_threads)

`num_threads`
[¶](#example-using-num_threads)

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%3:2 = transform.structured.tile_using_forall %0 num_threads [10, 20]
   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

`%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
: (!transform.any_op) -> !transform.any_op
%3:2 = transform.structured.tile_using_forall %0 num_threads [10, 20]
: (!transform.any_op) -> (!transform.any_op, !transform.any_op)`

---

#### Example using `tile_sizes` [¶](#example-using-tile_sizes)

`tile_sizes`
[¶](#example-using-tile_sizes)

```
%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%sz = transform.structured.match ...
%3:2 = transform.structured.tile_using_forall %0 tile_sizes [0, %sz, 20]
   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
```

`%0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
: (!transform.any_op) -> !transform.any_op
%sz = transform.structured.match ...
%3:2 = transform.structured.tile_using_forall %0 tile_sizes [0, %sz, 20]
: (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)`

Traits: `AttrSizedOperandSegments`, `ReportTrackingListenerFailuresOpTrait`

`AttrSizedOperandSegments`
`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-82)

[¶](#attributes-82)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_num_threads` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_num_threads` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_num_threads` |`static_num_threads` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `static_tile_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_tile_sizes` |`static_tile_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `mapping` | ::mlir::ArrayAttr | Device Mapping array attribute |
 `mapping` |`mapping` ::mlir::ArrayAttr | Device Mapping array attribute |

---

#### Operands: [¶](#operands-121)

[¶](#operands-121)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `num_threads` | variadic of transform any param type or any handle type |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `packed_num_threads` | transform any param type or any handle type |
| `packed_tile_sizes` | transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `num_threads` | variadic of transform any param type or any handle type |
| `tile_sizes` | variadic of transform any param type or any handle type |
| `packed_num_threads` | transform any param type or any handle type |
| `packed_tile_sizes` | transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `num_threads` | variadic of transform any param type or any handle type |
 `num_threads` |`num_threads` variadic of transform any param type or any handle type || `tile_sizes` | variadic of transform any param type or any handle type |
 `tile_sizes` |`tile_sizes` variadic of transform any param type or any handle type || `packed_num_threads` | transform any param type or any handle type |
 `packed_num_threads` |`packed_num_threads` transform any param type or any handle type || `packed_tile_sizes` | transform any param type or any handle type |
 `packed_tile_sizes` |`packed_tile_sizes` transform any param type or any handle type |

---

#### Results: [¶](#results-95)

[¶](#results-95)

| Result | Description |
| --- | --- |
| `tiled_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `tiled_op` | TransformHandleTypeInterface instance |
| `forall_op` | TransformHandleTypeInterface instance |
| `tiled_op` | TransformHandleTypeInterface instance |
 `tiled_op` |`tiled_op` TransformHandleTypeInterface instance || `forall_op` | TransformHandleTypeInterface instance |
 `forall_op` |`forall_op` TransformHandleTypeInterface instance |

---

### `transform.structured.transpose_conv2d` (transform::TransposeConv2DOp) [¶](#transformstructuredtranspose_conv2d-transformtransposeconv2dop)

`transform.structured.transpose_conv2d`
[¶](#transformstructuredtranspose_conv2d-transformtransposeconv2dop)

Syntax:

```
operation ::= `transform.structured.transpose_conv2d` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.transpose_conv2d` $target attr-dict `:` functional-type($target, results) ``

Convert linalg.conv\_2d\_nhwc\_fhwc into linalg.conv\_2d\_nhwc\_hwcf by introducing
a linalg.transpose on the filter tensor/memref.

Whilst the fhwc filter channel ordering can be desirable for certain targets
and is a more direct mapping to higher level dialects such as TOSA (which only
supports this ordering) hwcf is better suited for transformations such as
img2col which can make use of optimized BLAS routines such as GEMM.

Returns one handle:

* The final operation of the sequence that replaces the original
  convolution.

- The final operation of the sequence that replaces the original
  convolution.

---

#### Return modes: [¶](#return-modes-71)

[¶](#return-modes-71)

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Operands: [¶](#operands-122)

[¶](#operands-122)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-96)

[¶](#results-96)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.transpose_matmul` (transform::TransposeMatmulOp) [¶](#transformstructuredtranspose_matmul-transformtransposematmulop)

`transform.structured.transpose_matmul`
[¶](#transformstructuredtranspose_matmul-transformtransposematmulop)

Syntax:

```
operation ::= `transform.structured.transpose_matmul` $target (`<` $inputToTranspose^ `>`)?
              attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.transpose_matmul` $target (`<` $inputToTranspose^ `>`)?
attr-dict `:` functional-type($target, results) ``

Convert Linalg matmul ops to transposed variants.

By default the LHS matrix is transposed. Specify `<rhs>` to instead
transpose RHS matrix.

`<rhs>`

---

#### Return modes: [¶](#return-modes-72)

[¶](#return-modes-72)

This operation fails if `target` is unsupported, i.e., not a
`linalg.matmul` or `linalg.batch_matmul`. Otherwise, the operation succeeds
and returns a handle to the transposed matmul op.

`target`
`linalg.matmul`
`linalg.batch_matmul`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-83)

[¶](#attributes-83)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `inputToTranspose` | mlir::transform::TransposeMatmulInputAttr | Input to transpose when converting matmul ops to transposed variants |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `inputToTranspose` | mlir::transform::TransposeMatmulInputAttr | Input to transpose when converting matmul ops to transposed variants |
 `inputToTranspose` |`inputToTranspose` mlir::transform::TransposeMatmulInputAttr | Input to transpose when converting matmul ops to transposed variants |

---

#### Operands: [¶](#operands-123)

[¶](#operands-123)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-97)

[¶](#results-97)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.vectorize_children_and_apply_patterns` (transform::VectorizeChildrenAndApplyPatternsOp) [¶](#transformstructuredvectorize_children_and_apply_patterns-transformvectorizechildrenandapplypatternsop)

`transform.structured.vectorize_children_and_apply_patterns`
[¶](#transformstructuredvectorize_children_and_apply_patterns-transformvectorizechildrenandapplypatternsop)

Syntax:

```
operation ::= `transform.structured.vectorize_children_and_apply_patterns` $target attr-dict `:`functional-type(operands, results)
```

`` operation ::= `transform.structured.vectorize_children_and_apply_patterns` $target attr-dict `:`functional-type(operands, results) ``

Vectorizes all children contained in the given `target` using the
configuration specified by the attributes of this op. This only vectorizes
structured ops that operate on shaped types and does not vectorize loops or
straight-line. Internally, it applies a set of rewrite patterns, some of
which enable vectorization and some of which clean up the results.
Therefore, it can only be applied to an op with the “isolated from above”
property. This transformation only fails if the entire pattern rewriting
failed, i.e., it does **not** fail when no ops were vectorized.

`target`
**not**

Finer granularity can be achieved either with the `VectorizeOp` for
individual ops or by outlining the target part of the payload IR into, e.g.,
a function, performing this transformation, and inlining it back.

`VectorizeOp`

Note that this transformation invalidates the handles to any payload IR
operation that is contained inside the vectorization target.

This transformation supports the following attributes:

* `fold_type_extensions_into_contract`: a `UnitAttr` to enable the folding of
  type extension operations into `vector.contract` to create a mixed precision
  operation.
* `vectorize_padding`: a `UnitAttr` to activate the vectorization of
  `tensor.pad` ops. Different pipelines may prefer to lower such ops to
  loops.
* `disable_multi_reduction_to_contract_patterns`: a `UnitAttr` to deactivate
  the rewrite of `vector.multi_reduction` to `vector.contract`. This is
  intended to be used in tests only.
* `disable_transfer_permutation_map_lowering_patterns`: a `UnitAttr` to
  deactivate the rewrite of `vector.transfer` with permutation maps into
  explicit `vector.transpose` operations. This is intended to be used in
  tests only but may be promoted to a first class attribute in the future.

- `fold_type_extensions_into_contract`: a `UnitAttr` to enable the folding of
  type extension operations into `vector.contract` to create a mixed precision
  operation.
`fold_type_extensions_into_contract`
`UnitAttr`
`vector.contract`- `vectorize_padding`: a `UnitAttr` to activate the vectorization of
  `tensor.pad` ops. Different pipelines may prefer to lower such ops to
  loops.
`vectorize_padding`
`UnitAttr`
`tensor.pad`- `disable_multi_reduction_to_contract_patterns`: a `UnitAttr` to deactivate
  the rewrite of `vector.multi_reduction` to `vector.contract`. This is
  intended to be used in tests only.
`disable_multi_reduction_to_contract_patterns`
`UnitAttr`
`vector.multi_reduction`
`vector.contract`- `disable_transfer_permutation_map_lowering_patterns`: a `UnitAttr` to
  deactivate the rewrite of `vector.transfer` with permutation maps into
  explicit `vector.transpose` operations. This is intended to be used in
  tests only but may be promoted to a first class attribute in the future.
`disable_transfer_permutation_map_lowering_patterns`
`UnitAttr`
`vector.transfer`
`vector.transpose`

---

#### Return modes: [¶](#return-modes-73)

[¶](#return-modes-73)

This operation produces a definite failure if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-84)

[¶](#attributes-84)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fold_type_extensions_into_contract` | ::mlir::UnitAttr | unit attribute |
| `vectorize_padding` | ::mlir::UnitAttr | unit attribute |
| `vectorize_nd_extract` | ::mlir::UnitAttr | unit attribute |
| `flatten_1d_depthwise_conv` | ::mlir::UnitAttr | unit attribute |
| `disable_multi_reduction_to_contract_patterns` | ::mlir::UnitAttr | unit attribute |
| `disable_transfer_permutation_map_lowering_patterns` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `fold_type_extensions_into_contract` | ::mlir::UnitAttr | unit attribute |
 `fold_type_extensions_into_contract` |`fold_type_extensions_into_contract` ::mlir::UnitAttr | unit attribute || `vectorize_padding` | ::mlir::UnitAttr | unit attribute |
 `vectorize_padding` |`vectorize_padding` ::mlir::UnitAttr | unit attribute || `vectorize_nd_extract` | ::mlir::UnitAttr | unit attribute |
 `vectorize_nd_extract` |`vectorize_nd_extract` ::mlir::UnitAttr | unit attribute || `flatten_1d_depthwise_conv` | ::mlir::UnitAttr | unit attribute |
 `flatten_1d_depthwise_conv` |`flatten_1d_depthwise_conv` ::mlir::UnitAttr | unit attribute || `disable_multi_reduction_to_contract_patterns` | ::mlir::UnitAttr | unit attribute |
 `disable_multi_reduction_to_contract_patterns` |`disable_multi_reduction_to_contract_patterns` ::mlir::UnitAttr | unit attribute || `disable_transfer_permutation_map_lowering_patterns` | ::mlir::UnitAttr | unit attribute |
 `disable_transfer_permutation_map_lowering_patterns` |`disable_transfer_permutation_map_lowering_patterns` ::mlir::UnitAttr | unit attribute |

---

#### Operands: [¶](#operands-124)

[¶](#operands-124)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-98)

[¶](#results-98)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.structured.vectorize` (transform::VectorizeOp) [¶](#transformstructuredvectorize-transformvectorizeop)

`transform.structured.vectorize`
[¶](#transformstructuredvectorize-transformvectorizeop)

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

`` operation ::= `transform.structured.vectorize` $target oilist(
`vector_sizes` custom<DynamicIndexList>(
$vector_sizes,
$static_vector_sizes,
$scalable_sizes))
attr-dict
`:` type($target)(`,`type($vector_sizes)^)? ``

Vectorize the target ops, which must be Linalg ops.

Use the optional vector sizes to specify exactly what configuration the
vectorizer should use. It will then use masked vectors of the specified
size to enforce this configuration (“masked vectorization”). If no vector
sizes are specified, the vectorizer will infer the shapes to use from the
target Linalg ops (“regular vectorization”). More specifically:

```
transform.structured.vectorize %target vector_sizes [1, 4] : !transform.any_op
# Regular vectorization - vector sizes are inferred from the target Op
transform.structured.vectorize %target : !transform.any_op
```

```
transform.structured.vectorize %target vector_sizes [1, 4] : !transform.any_op
# Regular vectorization - vector sizes are inferred from the target Op
transform.structured.vectorize %target : !transform.any_op
```

`transform.structured.vectorize %target vector_sizes [1, 4] : !transform.any_op
# Regular vectorization - vector sizes are inferred from the target Op
transform.structured.vectorize %target : !transform.any_op`
transform.structured.vectorize %target vector\_sizes [1, 4] : !transform.any\_op
transform.structured.vectorize %target vector\_sizes [1, 4] : !transform.any\_op
.
.
vector
%target
vector
[
1
,
4
]
:
!
.
# Regular vectorization - vector sizes are inferred from the target Op
# Regular vectorization - vector sizes are inferred from the target Op
#
vector
-
vector
transform.structured.vectorize %target : !transform.any\_op
transform.structured.vectorize %target : !transform.any\_op
.
.
vector
%target
:
!
.

The vector sizes can be either static or dynamic (SSA values). In case of
SSA values, the handle must be mapped to exactly one payload op with
exactly one index-typed result.

Note: The input vector sizes must be bigger than or equal to their
counterpart iteration space sizes.

Typically this operator should be applied to linalg operations that have
already been tiled to the appropriate sizes.

---

#### Return modes: [¶](#return-modes-74)

[¶](#return-modes-74)

This operation produces a silenceable failure if at least one target op is
not a Linalg op or fails to vectorize. It produces a definite failure if
the dynamic vector sizes (SSA values) do not satisfy the constraints
mentioned above.

Traits: `ReportTrackingListenerFailuresOpTrait`

`ReportTrackingListenerFailuresOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

`MemoryEffectOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-85)

[¶](#attributes-85)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `static_vector_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `vectorize_nd_extract` | ::mlir::UnitAttr | unit attribute |
| `assume_dynamic_dims_match_vec_sizes` | ::mlir::UnitAttr | unit attribute |
| `create_named_contraction` | ::mlir::UnitAttr | unit attribute |
| `scalable_sizes` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `static_vector_sizes` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
 `static_vector_sizes` |`static_vector_sizes` ::mlir::DenseI64ArrayAttr | i64 dense array attribute || `vectorize_nd_extract` | ::mlir::UnitAttr | unit attribute |
 `vectorize_nd_extract` |`vectorize_nd_extract` ::mlir::UnitAttr | unit attribute || `assume_dynamic_dims_match_vec_sizes` | ::mlir::UnitAttr | unit attribute |
 `assume_dynamic_dims_match_vec_sizes` |`assume_dynamic_dims_match_vec_sizes` ::mlir::UnitAttr | unit attribute || `create_named_contraction` | ::mlir::UnitAttr | unit attribute |
 `create_named_contraction` |`create_named_contraction` ::mlir::UnitAttr | unit attribute || `scalable_sizes` | ::mlir::DenseBoolArrayAttr | i1 dense array attribute |
 `scalable_sizes` |`scalable_sizes` ::mlir::DenseBoolArrayAttr | i1 dense array attribute |

---

#### Operands: [¶](#operands-125)

[¶](#operands-125)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |
| `vector_sizes` | variadic of transform any param type or any handle type |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `vector_sizes` | variadic of transform any param type or any handle type |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance || `vector_sizes` | variadic of transform any param type or any handle type |
 `vector_sizes` |`vector_sizes` variadic of transform any param type or any handle type |

---

### `transform.structured.winograd_conv2d` (transform::WinogradConv2DOp) [¶](#transformstructuredwinograd_conv2d-transformwinogradconv2dop)

`transform.structured.winograd_conv2d`
[¶](#transformstructuredwinograd_conv2d-transformwinogradconv2dop)

Syntax:

```
operation ::= `transform.structured.winograd_conv2d` $target attr-dict `:` functional-type($target, results)
```

`` operation ::= `transform.structured.winograd_conv2d` $target attr-dict `:` functional-type($target, results) ``

Winograd Conv2D algorithm will convert linalg Conv2D operation into batched
matrix multiply. Before the matrix multiply, it will convert filter and
input into a format suitable for batched matrix multiply. After the matrix
multiply, it will convert output to the final result tensor.

The algorithm F(m x m, r x r) is

Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

The size of output Y is m x m. The size of filter g is r x r. The size of
input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
transformation matrices.

---

#### Return modes: [¶](#return-modes-75)

[¶](#return-modes-75)

This operation produces a silenceable failure if `target` is unsupported.
Otherwise, the operation succeeds and returns a handle of the sequence that
replaces the original convolution.

`target`

Traits: `FunctionalStyleTransformOpTrait`, `ReportTrackingListenerFailuresOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`ReportTrackingListenerFailuresOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-86)

[¶](#attributes-86)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fmr` | mlir::linalg::WinogradConv2DFmrAttr | allowed 32-bit signless integer cases: 0, 1, 2 |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `fmr` | mlir::linalg::WinogradConv2DFmrAttr | allowed 32-bit signless integer cases: 0, 1, 2 |
 `fmr` |`fmr` mlir::linalg::WinogradConv2DFmrAttr | allowed 32-bit signless integer cases: 0, 1, 2 |

---

#### Operands: [¶](#operands-126)

[¶](#operands-126)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-99)

[¶](#results-99)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.apply_patterns.tensor.bubble_up_extract_slice` (transform::ApplyBubbleUpExtractSlicePatternsOp) [¶](#transformapply_patternstensorbubble_up_extract_slice-transformapplybubbleupextractslicepatternsop)

`transform.apply_patterns.tensor.bubble_up_extract_slice`
[¶](#transformapply_patternstensorbubble_up_extract_slice-transformapplybubbleupextractslicepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.bubble_up_extract_slice` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.bubble_up_extract_slice` attr-dict ``

Indicates that producers of tensor.extract\_slice should swap and operate on
the result of the slice.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.decompose_concat` (transform::ApplyDecomposeTensorConcatPatternsOp) [¶](#transformapply_patternstensordecompose_concat-transformapplydecomposetensorconcatpatternsop)

`transform.apply_patterns.tensor.decompose_concat`
[¶](#transformapply_patternstensordecompose_concat-transformapplydecomposetensorconcatpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.decompose_concat` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.decompose_concat` attr-dict ``

Indicates that tensor.concat ops should be decomposed into a chain of
tensor.insert\_slice operations inserting into a materialized destination.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion` (transform::ApplyDropRedundantInsertSliceRankExpansionPatternsOp) [¶](#transformapply_patternstensordrop_redundant_insert_slice_rank_expansion-transformapplydropredundantinsertslicerankexpansionpatternsop)

`transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion`
[¶](#transformapply_patternstensordrop_redundant_insert_slice_rank_expansion-transformapplydropredundantinsertslicerankexpansionpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion` attr-dict ``

Indicates that redundant tensor.insert\_slice rank reductions should be
dropped. E.g., cases where a tensor.extract\_slice rank reduction immediately
follows an inverse tensor.insert\_slice rank expansion.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.fold_tensor_empty` (transform::ApplyFoldTensorEmptyPatternsOp) [¶](#transformapply_patternstensorfold_tensor_empty-transformapplyfoldtensoremptypatternsop)

`transform.apply_patterns.tensor.fold_tensor_empty`
[¶](#transformapply_patternstensorfold_tensor_empty-transformapplyfoldtensoremptypatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_empty` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.fold_tensor_empty` attr-dict ``

Indicates that tensor.extract\_slice and reassociative reshapes should be
folded into tensor.empty.

If `fold_single_use_only` is set to “true”, only tensor.empty that have a
single use are folded.

`fold_single_use_only`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-87)

[¶](#attributes-87)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `fold_single_use_only` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `fold_single_use_only` | ::mlir::BoolAttr | bool attribute |
 `fold_single_use_only` |`fold_single_use_only` ::mlir::BoolAttr | bool attribute |

---

### `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers` (transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp) [¶](#transformapply_patternstensorfold_tensor_subset_ops_into_vector_transfers-transformapplyfoldtensorsubsetopsintovectortransferspatternsop)

`transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers`
[¶](#transformapply_patternstensorfold_tensor_subset_ops_into_vector_transfers-transformapplyfoldtensorsubsetopsintovectortransferspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers` attr-dict ``

Indicates that tensor.extract\_slice -> vector.transfer\_read and
vector.transfer\_write -> tensor.insert\_slice op chains should be folded into
vector tranfer read and write ops

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.fold_tensor_subset_ops` (transform::ApplyFoldTensorSubsetOpsPatternsOp) [¶](#transformapply_patternstensorfold_tensor_subset_ops-transformapplyfoldtensorsubsetopspatternsop)

`transform.apply_patterns.tensor.fold_tensor_subset_ops`
[¶](#transformapply_patternstensorfold_tensor_subset_ops-transformapplyfoldtensorsubsetopspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.fold_tensor_subset_ops` attr-dict ``

Indicates that tensor.empty should be folded with tensor.extract\_slice,
tensor.expand\_shape and tensor.collapse\_shape.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice` (transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp) [¶](#transformapply_patternstensormerge_consecutive_insert_extract_slice-transformapplymergeconsecutiveinsertextractslicepatternsop)

`transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice`
[¶](#transformapply_patternstensormerge_consecutive_insert_extract_slice-transformapplymergeconsecutiveinsertextractslicepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice` attr-dict ``

Indicates that consecutive tensor.extract\_slice/tensor.insert\_slice ops
should be merged into a single op. These patterns are not canonicalizations
because the bufferization is sensitive to IR structure.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.reassociative_reshape_folding` (transform::ApplyReassociativeReshapeFoldingPatternsOp) [¶](#transformapply_patternstensorreassociative_reshape_folding-transformapplyreassociativereshapefoldingpatternsop)

`transform.apply_patterns.tensor.reassociative_reshape_folding`
[¶](#transformapply_patternstensorreassociative_reshape_folding-transformapplyreassociativereshapefoldingpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.reassociative_reshape_folding` attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.reassociative_reshape_folding` attr-dict ``

Indicates that reassociative reshapes (tensor.collapse\_shape /
tensor.expand\_shape) should be folded with inverse rank expansions / rank
reductions (via tensor.insert\_slice / tensor.extract\_slice).

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.tensor.rewrite_as_constant` (transform::ApplyRewriteTensorOpsAsConstantPatternsOp) [¶](#transformapply_patternstensorrewrite_as_constant-transformapplyrewritetensoropsasconstantpatternsop)

`transform.apply_patterns.tensor.rewrite_as_constant`
[¶](#transformapply_patternstensorrewrite_as_constant-transformapplyrewritetensoropsasconstantpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.tensor.rewrite_as_constant` (`aggressive` $aggressive^)? attr-dict
```

`` operation ::= `transform.apply_patterns.tensor.rewrite_as_constant` (`aggressive` $aggressive^)? attr-dict ``

Indicates that tensor ops (such as tensor.generate) should be replaced with
constants (arith.constant) when possible.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-88)

[¶](#attributes-88)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `aggressive` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `aggressive` | ::mlir::UnitAttr | unit attribute |
 `aggressive` |`aggressive` ::mlir::UnitAttr | unit attribute |

---

### `transform.tensor.make_loop_independent` (transform::MakeLoopIndependentOp) [¶](#transformtensormake_loop_independent-transformmakeloopindependentop)

`transform.tensor.make_loop_independent`
[¶](#transformtensormake_loop_independent-transformmakeloopindependentop)

Syntax:

```
operation ::= `transform.tensor.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed)
```

`` operation ::= `transform.tensor.make_loop_independent` $target attr-dict `:` functional-type($target, $transformed) ``

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`tensor.extract_slice` inside the loop.

`num_loop`
`scf.for`
`num_loop`
`tensor.extract_slice`

Currently supported operations are:

* tensor.empty: Replaced with a new tensor.empty with upper bound sizes,
  followed by a tensor.extract\_slice.
* tensor.pad: Replaced by an upper bound padding, followed by a
  tensor.extract\_slice.

- tensor.empty: Replaced with a new tensor.empty with upper bound sizes,
  followed by a tensor.extract\_slice.
- tensor.pad: Replaced by an upper bound padding, followed by a
  tensor.extract\_slice.

---

#### Return modes [¶](#return-modes-76)

[¶](#return-modes-76)

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:

* tensor.empty: The returned handle points to the tensor.extract\_slice op.
* tensor.pad: The returned handle points to the tensor.extract\_slice op.

- tensor.empty: The returned handle points to the tensor.extract\_slice op.
- tensor.pad: The returned handle points to the tensor.extract\_slice op.

This transform op consumes the target handle and produces a result handle.

Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

`FunctionalStyleTransformOpTrait`
`TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

`MemoryEffectsOpInterface`
`TransformOpInterface`

---

#### Attributes: [¶](#attributes-89)

[¶](#attributes-89)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `num_loops` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `num_loops` |`num_loops` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

#### Operands: [¶](#operands-127)

[¶](#operands-127)

| Operand | Description |
| --- | --- |
| `target` | TransformHandleTypeInterface instance |

| Operand | Description |
| --- | --- |
| Operand | Description |
 Operand | Description || `target` | TransformHandleTypeInterface instance |
| `target` | TransformHandleTypeInterface instance |
 `target` |`target` TransformHandleTypeInterface instance |

---

#### Results: [¶](#results-100)

[¶](#results-100)

| Result | Description |
| --- | --- |
| `transformed` | TransformHandleTypeInterface instance |

| Result | Description |
| --- | --- |
| Result | Description |
 Result | Description || `transformed` | TransformHandleTypeInterface instance |
| `transformed` | TransformHandleTypeInterface instance |
 `transformed` |`transformed` TransformHandleTypeInterface instance |

---

### `transform.type_conversion.tensor.cast_shape_dynamic_dims` (transform::TypeConversionCastShapeDynamicDimsOp) [¶](#transformtype_conversiontensorcast_shape_dynamic_dims-transformtypeconversioncastshapedynamicdimsop)

`transform.type_conversion.tensor.cast_shape_dynamic_dims`
[¶](#transformtype_conversiontensorcast_shape_dynamic_dims-transformtypeconversioncastshapedynamicdimsop)

Syntax:

```
operation ::= `transform.type_conversion.tensor.cast_shape_dynamic_dims` (`ignore_dynamic_info` $ignore_dynamic_info^)? attr-dict
```

`` operation ::= `transform.type_conversion.tensor.cast_shape_dynamic_dims` (`ignore_dynamic_info` $ignore_dynamic_info^)? attr-dict ``

Populates a type converter with conversion materialization functions that
cast a tensor value between two cast-compatible tensors. See `tensor.cast`
for more information on cast compatibility between tensors.

`tensor.cast`

If `ignore_dynamic_info` is not set, this will set an additional constraint
that source materializations do not cast dynamic dimensions to static ones.

`ignore_dynamic_info`

Interfaces: `TypeConverterBuilderOpInterface`

`TypeConverterBuilderOpInterface`

---

#### Attributes: [¶](#attributes-90)

[¶](#attributes-90)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `ignore_dynamic_info` | ::mlir::UnitAttr | unit attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `ignore_dynamic_info` | ::mlir::UnitAttr | unit attribute |
 `ignore_dynamic_info` |`ignore_dynamic_info` ::mlir::UnitAttr | unit attribute |

---

### `transform.apply_patterns.vector.cast_away_vector_leading_one_dim` (transform::ApplyCastAwayVectorLeadingOneDimPatternsOp) [¶](#transformapply_patternsvectorcast_away_vector_leading_one_dim-transformapplycastawayvectorleadingonedimpatternsop)

`transform.apply_patterns.vector.cast_away_vector_leading_one_dim`
[¶](#transformapply_patternsvectorcast_away_vector_leading_one_dim-transformapplycastawayvectorleadingonedimpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.cast_away_vector_leading_one_dim` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.cast_away_vector_leading_one_dim` attr-dict ``

Collect a set of leading one dimension removal patterns.

These patterns insert vector.shape\_cast to remove leading one dimensions
to expose more canonical forms of read/write/insert/extract operations.
With them, there are more chances that we can cancel out extract-insert
pairs or forward write-read pairs.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops` (transform::ApplyDropInnerMostUnitDimsFromXferOpsPatternsOp) [¶](#transformapply_patternsvectordrop_inner_most_unit_dims_from_xfer_ops-transformapplydropinnermostunitdimsfromxferopspatternsop)

`transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops`
[¶](#transformapply_patternsvectordrop_inner_most_unit_dims_from_xfer_ops-transformapplydropinnermostunitdimsfromxferopspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops` attr-dict ``

Apply vector patterns to drop the inner most unit dims from
vector.transfer\_read and vector.transfer\_write Ops by taking a subview (via
memref.subview) of the original source/destination MemRef. Since it
requires the input/ouptu to be MemRefs, this Op is only helpful
past-bufferization.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast` (transform::ApplyDropUnitDimWithShapeCastPatternsOp) [¶](#transformapply_patternsvectordrop_unit_dims_with_shape_cast-transformapplydropunitdimwithshapecastpatternsop)

`transform.apply_patterns.vector.drop_unit_dims_with_shape_cast`
[¶](#transformapply_patternsvectordrop_unit_dims_with_shape_cast-transformapplydropunitdimwithshapecastpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.drop_unit_dims_with_shape_cast` attr-dict ``

Apply vector patterns to fold unit dims with vector.shape\_cast Ops:

* DropUnitDimFromElementwiseOps
* DropUnitDimsFromScfForOp
* DropUnitDimsFromTransposeOp

- DropUnitDimFromElementwiseOps
- DropUnitDimsFromScfForOp
- DropUnitDimsFromTransposeOp

Excludes patterns for vector.transfer Ops. This is complemented by
shape\_cast folding patterns.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.fold_arith_extension` (transform::ApplyFoldArithExtensionPatternsOp) [¶](#transformapply_patternsvectorfold_arith_extension-transformapplyfoldarithextensionpatternsop)

`transform.apply_patterns.vector.fold_arith_extension`
[¶](#transformapply_patternsvectorfold_arith_extension-transformapplyfoldarithextensionpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.fold_arith_extension` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.fold_arith_extension` attr-dict ``

Collect a set of patterns that fold arithmetic extension on floating point
into vector contract for the backends with native support.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.elementwise_to_vector` (transform::ApplyFoldElementwiseToVectorPatternsOp) [¶](#transformapply_patternsvectorelementwise_to_vector-transformapplyfoldelementwisetovectorpatternsop)

`transform.apply_patterns.vector.elementwise_to_vector`
[¶](#transformapply_patternsvectorelementwise_to_vector-transformapplyfoldelementwisetovectorpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.elementwise_to_vector` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.elementwise_to_vector` attr-dict ``

Collect a set of patterns that fold elementwise op on vectors to the vector
dialect.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.interleave_to_shuffle` (transform::ApplyInterleaveToShufflePatternsOp) [¶](#transformapply_patternsvectorinterleave_to_shuffle-transformapplyinterleavetoshufflepatternsop)

`transform.apply_patterns.vector.interleave_to_shuffle`
[¶](#transformapply_patternsvectorinterleave_to_shuffle-transformapplyinterleavetoshufflepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.interleave_to_shuffle` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.interleave_to_shuffle` attr-dict ``

Indicates that 1D vector interleave operations should be rewritten as
vector shuffle operations.

This is motivated by some current codegen backends not handling vector
interleave operations.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_bitcast` (transform::ApplyLowerBitCastPatternsOp) [¶](#transformapply_patternsvectorlower_bitcast-transformapplylowerbitcastpatternsop)

`transform.apply_patterns.vector.lower_bitcast`
[¶](#transformapply_patternsvectorlower_bitcast-transformapplylowerbitcastpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_bitcast` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_bitcast` attr-dict ``

Indicates that vector bitcast operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_broadcast` (transform::ApplyLowerBroadcastPatternsOp) [¶](#transformapply_patternsvectorlower_broadcast-transformapplylowerbroadcastpatternsop)

`transform.apply_patterns.vector.lower_broadcast`
[¶](#transformapply_patternsvectorlower_broadcast-transformapplylowerbroadcastpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_broadcast` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_broadcast` attr-dict ``

Indicates that vector broadcast operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_contraction` (transform::ApplyLowerContractionPatternsOp) [¶](#transformapply_patternsvectorlower_contraction-transformapplylowercontractionpatternsop)

`transform.apply_patterns.vector.lower_contraction`
[¶](#transformapply_patternsvectorlower_contraction-transformapplylowercontractionpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_contraction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_contraction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict ``

Indicates that vector contraction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-91)

[¶](#attributes-91)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowering_strategy` | ::mlir::vector::VectorContractLoweringAttr | control the lowering of `vector.contract` operations. |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lowering_strategy` | ::mlir::vector::VectorContractLoweringAttr | control the lowering of `vector.contract` operations. |
 `lowering_strategy` |`lowering_strategy` ::mlir::vector::VectorContractLoweringAttr | control the lowering of `vector.contract` operations. |

---

### `transform.apply_patterns.vector.lower_create_mask` (transform::ApplyLowerCreateMaskPatternsOp) [¶](#transformapply_patternsvectorlower_create_mask-transformapplylowercreatemaskpatternsop)

`transform.apply_patterns.vector.lower_create_mask`
[¶](#transformapply_patternsvectorlower_create_mask-transformapplylowercreatemaskpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_create_mask` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_create_mask` attr-dict ``

Indicates that vector create\_mask-like operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_gather` (transform::ApplyLowerGatherPatternsOp) [¶](#transformapply_patternsvectorlower_gather-transformapplylowergatherpatternsop)

`transform.apply_patterns.vector.lower_gather`
[¶](#transformapply_patternsvectorlower_gather-transformapplylowergatherpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_gather` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_gather` attr-dict ``

Indicates that vector.gather operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_interleave` (transform::ApplyLowerInterleavePatternsOp) [¶](#transformapply_patternsvectorlower_interleave-transformapplylowerinterleavepatternsop)

`transform.apply_patterns.vector.lower_interleave`
[¶](#transformapply_patternsvectorlower_interleave-transformapplylowerinterleavepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_interleave` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_interleave` attr-dict ``

Indicates that vector interleave operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_masked_transfers` (transform::ApplyLowerMaskedTransfersPatternsOp) [¶](#transformapply_patternsvectorlower_masked_transfers-transformapplylowermaskedtransferspatternsop)

`transform.apply_patterns.vector.lower_masked_transfers`
[¶](#transformapply_patternsvectorlower_masked_transfers-transformapplylowermaskedtransferspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_masked_transfers` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_masked_transfers` attr-dict ``

Apply opt-in patterns that lower vector.mask operations surrounding
side-effecting ops:

* MaskedTransferReadOpPattern
* MaskedTransferWriteOpPattern
* MaskedGatherOpPattern

- MaskedTransferReadOpPattern
- MaskedTransferWriteOpPattern
- MaskedGatherOpPattern

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_masks` (transform::ApplyLowerMasksPatternsOp) [¶](#transformapply_patternsvectorlower_masks-transformapplylowermaskspatternsop)

`transform.apply_patterns.vector.lower_masks`
[¶](#transformapply_patternsvectorlower_masks-transformapplylowermaskspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_masks` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_masks` attr-dict ``

Indicates that vector.create\_mask and vector.constant\_mask operations
should be lowered to finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_multi_reduction` (transform::ApplyLowerMultiReductionPatternsOp) [¶](#transformapply_patternsvectorlower_multi_reduction-transformapplylowermultireductionpatternsop)

`transform.apply_patterns.vector.lower_multi_reduction`
[¶](#transformapply_patternsvectorlower_multi_reduction-transformapplylowermultireductionpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_multi_reduction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_multi_reduction` (`lowering_strategy` `=` $lowering_strategy^)? attr-dict ``

Indicates that vector multi\_reduction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-92)

[¶](#attributes-92)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowering_strategy` | ::mlir::vector::VectorMultiReductionLoweringAttr | control the lowering of `vector.multi\_reduction`. |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lowering_strategy` | ::mlir::vector::VectorMultiReductionLoweringAttr | control the lowering of `vector.multi\_reduction`. |
 `lowering_strategy` |`lowering_strategy` ::mlir::vector::VectorMultiReductionLoweringAttr | control the lowering of `vector.multi\_reduction`. |

---

### `transform.apply_patterns.vector.lower_outerproduct` (transform::ApplyLowerOuterProductPatternsOp) [¶](#transformapply_patternsvectorlower_outerproduct-transformapplylowerouterproductpatternsop)

`transform.apply_patterns.vector.lower_outerproduct`
[¶](#transformapply_patternsvectorlower_outerproduct-transformapplylowerouterproductpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_outerproduct` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_outerproduct` attr-dict ``

Indicates that the vector outerproduct operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_scan` (transform::ApplyLowerScanPatternsOp) [¶](#transformapply_patternsvectorlower_scan-transformapplylowerscanpatternsop)

`transform.apply_patterns.vector.lower_scan`
[¶](#transformapply_patternsvectorlower_scan-transformapplylowerscanpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_scan` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_scan` attr-dict ``

Indicates that vector.scan operations should be lowered to
finer-grained vector primitives.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_shape_cast` (transform::ApplyLowerShapeCastPatternsOp) [¶](#transformapply_patternsvectorlower_shape_cast-transformapplylowershapecastpatternsop)

`transform.apply_patterns.vector.lower_shape_cast`
[¶](#transformapply_patternsvectorlower_shape_cast-transformapplylowershapecastpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_shape_cast` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_shape_cast` attr-dict ``

Indicates that vector shape\_cast operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.lower_transfer` (transform::ApplyLowerTransferPatternsOp) [¶](#transformapply_patternsvectorlower_transfer-transformapplylowertransferpatternsop)

`transform.apply_patterns.vector.lower_transfer`
[¶](#transformapply_patternsvectorlower_transfer-transformapplylowertransferpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_transfer` (`max_transfer_rank` `=` $max_transfer_rank^)? attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_transfer` (`max_transfer_rank` `=` $max_transfer_rank^)? attr-dict ``

Indicates that vector transfer operations should be lowered to finer-grained
vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-93)

[¶](#attributes-93)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `max_transfer_rank` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `max_transfer_rank` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `max_transfer_rank` |`max_transfer_rank` ::mlir::IntegerAttr | 64-bit signless integer attribute |

---

### `transform.apply_patterns.vector.lower_transpose` (transform::ApplyLowerTransposePatternsOp) [¶](#transformapply_patternsvectorlower_transpose-transformapplylowertransposepatternsop)

`transform.apply_patterns.vector.lower_transpose`
[¶](#transformapply_patternsvectorlower_transpose-transformapplylowertransposepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.lower_transpose` oilist (
              `lowering_strategy` `=` $lowering_strategy
              | `avx2_lowering_strategy` `=` $avx2_lowering_strategy
              )
              attr-dict
```

`` operation ::= `transform.apply_patterns.vector.lower_transpose` oilist (
`lowering_strategy` `=` $lowering_strategy
| `avx2_lowering_strategy` `=` $avx2_lowering_strategy
)
attr-dict ``

Indicates that vector transpose-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-94)

[¶](#attributes-94)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `lowering_strategy` | ::mlir::vector::VectorTransposeLoweringAttr | control the lowering of `vector.transpose` operations. |
| `avx2_lowering_strategy` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `lowering_strategy` | ::mlir::vector::VectorTransposeLoweringAttr | control the lowering of `vector.transpose` operations. |
 `lowering_strategy` |`lowering_strategy` ::mlir::vector::VectorTransposeLoweringAttr | control the lowering of `vector.transpose` operations. || `avx2_lowering_strategy` | ::mlir::BoolAttr | bool attribute |
 `avx2_lowering_strategy` |`avx2_lowering_strategy` ::mlir::BoolAttr | bool attribute |

---

### `transform.apply_patterns.vector.materialize_masks` (transform::ApplyMaterializeMasksPatternsOp) [¶](#transformapply_patternsvectormaterialize_masks-transformapplymaterializemaskspatternsop)

`transform.apply_patterns.vector.materialize_masks`
[¶](#transformapply_patternsvectormaterialize_masks-transformapplymaterializemaskspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.materialize_masks` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.materialize_masks` attr-dict ``

Indicates that mask operations should be lowered to fine-grained arithemtic
operations.

This is usually the last step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.rank_reducing_subview_patterns` (transform::ApplyRankReducingSubviewPatternsOp) [¶](#transformapply_patternsvectorrank_reducing_subview_patterns-transformapplyrankreducingsubviewpatternsop)

`transform.apply_patterns.vector.rank_reducing_subview_patterns`
[¶](#transformapply_patternsvectorrank_reducing_subview_patterns-transformapplyrankreducingsubviewpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.rank_reducing_subview_patterns` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.rank_reducing_subview_patterns` attr-dict ``

Apply opt-in vector transfer permutation patterns that include:

* TransferReadDropUnitDimsPattern
* TransferWriteDropUnitDimsPattern

- TransferReadDropUnitDimsPattern
- TransferWriteDropUnitDimsPattern

These patterns have the effect of rewriting a vector.transfer with unit
dimensions into a rank-reduced version thanks to subview operations.
This is complemented by shape\_cast folding patterns.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.rewrite_narrow_types` (transform::ApplyRewriteNarrowTypePatternsOp) [¶](#transformapply_patternsvectorrewrite_narrow_types-transformapplyrewritenarrowtypepatternsop)

`transform.apply_patterns.vector.rewrite_narrow_types`
[¶](#transformapply_patternsvectorrewrite_narrow_types-transformapplyrewritenarrowtypepatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.rewrite_narrow_types` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.rewrite_narrow_types` attr-dict ``

Indicates that vector narrow rewrite operations should be applied.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Warning: these patterns currently only work for little endian targets.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.sink_mem_ops` (transform::ApplySinkVectorMemPatternsOp) [¶](#transformapply_patternsvectorsink_mem_ops-transformapplysinkvectormempatternsop)

`transform.apply_patterns.vector.sink_mem_ops`
[¶](#transformapply_patternsvectorsink_mem_ops-transformapplysinkvectormempatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.sink_mem_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.sink_mem_ops` attr-dict ``

Patterns that replace redundant Vector Ops (followed by
`vector.load`/`vector.store`) with either vector.load/vector.store or
`memref.load`/`memref.store`. Currently limited to 1-element vectors.

`vector.load`
`vector.store`
`memref.load`
`memref.store`

Example:

```
vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
vector.extract %0[1] : f32 from vector<4xf32>
```

`vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
vector.extract %0[1] : f32 from vector<4xf32>`

Gets converted to:

```
%c1 = arith.constant 1 : index
%0 = arith.addi %arg1, %c1 overflow<nsw> : index
%1 = memref.load %arg0[%0] : memref<?xf32>
```

`%c1 = arith.constant 1 : index
%0 = arith.addi %arg1, %c1 overflow<nsw> : index
%1 = memref.load %arg0[%0] : memref<?xf32>`

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.sink_ops` (transform::ApplySinkVectorPatternsOp) [¶](#transformapply_patternsvectorsink_ops-transformapplysinkvectorpatternsop)

`transform.apply_patterns.vector.sink_ops`
[¶](#transformapply_patternsvectorsink_ops-transformapplysinkvectorpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.sink_ops` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.sink_ops` attr-dict ``

Patterns that remove redundant Vector Ops by re-ordering them with
e.g. elementwise Ops.

Example:

```
%at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%r = arith.addf %at, %bt : vector<2x4xf32>
```

`%at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
%r = arith.addf %at, %bt : vector<2x4xf32>`

gets converted to:

```
%0 = arith.addf %a, %b : vector<4x2xf32>
%r = vector.transpose %0, [1, 0] : vector<2x4xf32>
```

`%0 = arith.addf %a, %b : vector<4x2xf32>
%r = vector.transpose %0, [1, 0] : vector<2x4xf32>`

At the moment, these patterns are limited to vector.broadcast,
vector.transpose and vector.extract.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.split_transfer_full_partial` (transform::ApplySplitTransferFullPartialPatternsOp) [¶](#transformapply_patternsvectorsplit_transfer_full_partial-transformapplysplittransferfullpartialpatternsop)

`transform.apply_patterns.vector.split_transfer_full_partial`
[¶](#transformapply_patternsvectorsplit_transfer_full_partial-transformapplysplittransferfullpartialpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.split_transfer_full_partial` (`split_transfer_strategy` `=` $split_transfer_strategy^)? attr-dict
```

`` operation ::= `transform.apply_patterns.vector.split_transfer_full_partial` (`split_transfer_strategy` `=` $split_transfer_strategy^)? attr-dict ``

Indicates that vector transfer operations should be split to full and
partial parts.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-95)

[¶](#attributes-95)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `split_transfer_strategy` | ::mlir::vector::VectorTransferSplitAttr | control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants. |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `split_transfer_strategy` | ::mlir::vector::VectorTransferSplitAttr | control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants. |
 `split_transfer_strategy` |`split_transfer_strategy` ::mlir::vector::VectorTransferSplitAttr | control the splitting of `vector.transfer` operations into in-bounds and out-of-bounds variants. |

---

### `transform.apply_patterns.vector.transfer_permutation_patterns` (transform::ApplyTransferPermutationPatternsOp) [¶](#transformapply_patternsvectortransfer_permutation_patterns-transformapplytransferpermutationpatternsop)

`transform.apply_patterns.vector.transfer_permutation_patterns`
[¶](#transformapply_patternsvectortransfer_permutation_patterns-transformapplytransferpermutationpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.transfer_permutation_patterns` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.transfer_permutation_patterns` attr-dict ``

Apply opt-in vector transfer permutation patterns that include:

* TransferReadPermutationLowering
* TransferWritePermutationLowering
* TransferOpReduceRank
* TransferWriteNonPermutationLowering

- TransferReadPermutationLowering
- TransferWritePermutationLowering
- TransferOpReduceRank
- TransferWriteNonPermutationLowering

These patterns have the effect of rewriting a vector.transfer with an
arbitrary permutation\_map to a vector.transfer with a permutation\_map that
is a minor identity followed by a vector.transpose.

In other words, this makes the vector.transfer contiguous on the most minor
dimensions and materializes the permutation\_map as a vector.transpose.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.transfer_to_scf` (transform::ApplyTransferToScfPatternsOp) [¶](#transformapply_patternsvectortransfer_to_scf-transformapplytransfertoscfpatternsop)

`transform.apply_patterns.vector.transfer_to_scf`
[¶](#transformapply_patternsvectortransfer_to_scf-transformapplytransfertoscfpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.transfer_to_scf` oilist (
              `max_transfer_rank` `=` $max_transfer_rank
              | `full_unroll` `=` $full_unroll
              )
              attr-dict
```

`` operation ::= `transform.apply_patterns.vector.transfer_to_scf` oilist (
`max_transfer_rank` `=` $max_transfer_rank
| `full_unroll` `=` $full_unroll
)
attr-dict ``

Indicates that vector transfer operations should be rewritten with scf.for
loops over finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-96)

[¶](#attributes-96)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `max_transfer_rank` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `full_unroll` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `max_transfer_rank` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
 `max_transfer_rank` |`max_transfer_rank` ::mlir::IntegerAttr | 64-bit signless integer attribute || `full_unroll` | ::mlir::BoolAttr | bool attribute |
 `full_unroll` |`full_unroll` ::mlir::BoolAttr | bool attribute |

---

### `transform.apply_patterns.vector.unroll_from_elements` (transform::ApplyUnrollFromElementsPatternsOp) [¶](#transformapply_patternsvectorunroll_from_elements-transformapplyunrollfromelementspatternsop)

`transform.apply_patterns.vector.unroll_from_elements`
[¶](#transformapply_patternsvectorunroll_from_elements-transformapplyunrollfromelementspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.unroll_from_elements` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.unroll_from_elements` attr-dict ``

Indicates that vector from\_elements operations should be unrolled
along the outermost dimension.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.unroll_to_elements` (transform::ApplyUnrollToElementsPatternsOp) [¶](#transformapply_patternsvectorunroll_to_elements-transformapplyunrolltoelementspatternsop)

`transform.apply_patterns.vector.unroll_to_elements`
[¶](#transformapply_patternsvectorunroll_to_elements-transformapplyunrolltoelementspatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.unroll_to_elements` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.unroll_to_elements` attr-dict ``

Indicates that vector to\_elements operations should be unrolled
along the outermost dimension.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_patterns.vector.reduction_to_contract` (transform::ApplyVectorReductionToContractPatternsOp) [¶](#transformapply_patternsvectorreduction_to_contract-transformapplyvectorreductiontocontractpatternsop)

`transform.apply_patterns.vector.reduction_to_contract`
[¶](#transformapply_patternsvectorreduction_to_contract-transformapplyvectorreductiontocontractpatternsop)

Syntax:

```
operation ::= `transform.apply_patterns.vector.reduction_to_contract` attr-dict
```

`` operation ::= `transform.apply_patterns.vector.reduction_to_contract` attr-dict ``

Apply opt-in patterns that convert reductions to contract:

* MultiReduceToContract
* CombineContractBroadcast
* CombineContractABTranspose
* CombineContractResultTranspose
* ReorderElementwiseOpsOnTranspose
* ReorderElementwiseOpsOnBroadcast
* ReorderCastOpsOnBroadcast

- MultiReduceToContract
- CombineContractBroadcast
- CombineContractABTranspose
- CombineContractResultTranspose
- ReorderElementwiseOpsOnTranspose
- ReorderElementwiseOpsOnBroadcast
- ReorderCastOpsOnBroadcast

These patterns have the effect of rewriting a vector.multi\_reduce into a
vector.contract.

Interfaces: `PatternDescriptorOpInterface`

`PatternDescriptorOpInterface`

---

### `transform.apply_conversion_patterns.vector.vector_to_llvm` (transform::ApplyVectorToLLVMConversionPatternsOp) [¶](#transformapply_conversion_patternsvectorvector_to_llvm-transformapplyvectortollvmconversionpatternsop)

`transform.apply_conversion_patterns.vector.vector_to_llvm`
[¶](#transformapply_conversion_patternsvectorvector_to_llvm-transformapplyvectortollvmconversionpatternsop)

Syntax:

```
operation ::= `transform.apply_conversion_patterns.vector.vector_to_llvm` attr-dict
```

`` operation ::= `transform.apply_conversion_patterns.vector.vector_to_llvm` attr-dict ``

Collects patterns that convert vector dialect ops to LLVM dialect ops. These
patterns require an “LLVMTypeConverter”.

The patterns can be customized as follows:

* `reassociate_fp_reductions`: Allows LLVM to reassociate floating-point
  reductions for speed.
* `force_32bit_vector_indices`: Allows the compiler to assume that vector
  indices fit in 32-bit if that yields faster code.

- `reassociate_fp_reductions`: Allows LLVM to reassociate floating-point
  reductions for speed.
`reassociate_fp_reductions`- `force_32bit_vector_indices`: Allows the compiler to assume that vector
  indices fit in 32-bit if that yields faster code.
`force_32bit_vector_indices`

Interfaces: `ConversionPatternDescriptorOpInterface`

`ConversionPatternDescriptorOpInterface`

---

#### Attributes: [¶](#attributes-97)

[¶](#attributes-97)

| Attribute | MLIR Type | Description |
| --- | --- | --- |
| `reassociate_fp_reductions` | ::mlir::BoolAttr | bool attribute |
| `force_32bit_vector_indices` | ::mlir::BoolAttr | bool attribute |
| `use_vector_alignment` | ::mlir::BoolAttr | bool attribute |

| Attribute | MLIR Type | Description |
 Attribute | MLIR Type | Description || `reassociate_fp_reductions` | ::mlir::BoolAttr | bool attribute |
 `reassociate_fp_reductions` |`reassociate_fp_reductions` ::mlir::BoolAttr | bool attribute || `force_32bit_vector_indices` | ::mlir::BoolAttr | bool attribute |
 `force_32bit_vector_indices` |`force_32bit_vector_indices` ::mlir::BoolAttr | bool attribute || `use_vector_alignment` | ::mlir::BoolAttr | bool attribute |
 `use_vector_alignment` |`use_vector_alignment` ::mlir::BoolAttr | bool attribute |
