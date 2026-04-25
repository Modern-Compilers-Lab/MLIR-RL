from math import prod

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Vectorizes the tagged linalg operation by tiling to match the
    requested vector sizes and then applying transform.structured.vectorize.
    For conv2d ops, preprocessing converts to img2col first.

    Each vector/tile size is clamped to evenly divide the corresponding
    iteration-space dimension.  Non-divisible sizes produce dynamic-sized
    remainder tiles whose vectorization generates ``vector.mask`` ops
    wrapping multiple operations (mulf + addf in reductions), violating
    the single-op constraint and causing a lowering failure:
      'vector.mask' op expects only one operation to mask
    Clamping to divisors avoids this entirely with zero overhead.
    """

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes for the innermost loop dimensions.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vs = params.get("vector_sizes", [])
        if not vs or not isinstance(vs, list):
            return False
        if all(s <= 1 for s in vs):
            return False
        if any(not isinstance(s, int) or s < 1 for s in vs):
            return False
        if prod(vs) > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """If the tagged op is a conv2d, convert to img2col first."""
        if "conv_2d_nchw_fchw" not in code:
            return code

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1'
            " : (!transform.any_op) -> !transform.any_op\n"
            "    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op"
            " : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            "    %matmul = transform.get_producer_of_operand %transformed[0]"
            " : (!transform.any_op) -> !transform.any_op\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
        )
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = list(params["vector_sizes"])

        # Preprocessing: convert conv2d to img2col if needed
        code = cls.preprocess(code, params)

        # Determine how many loops the tagged op has by counting
        # iterator_types or using a heuristic from the code
        n_dims = cls._count_dims(code)
        if n_dims == 0:
            return code

        # Build full tile/vector sizes: pad with 1 for dims we don't vectorize
        # tile_size 1 creates a loop with step 1, and vector_size 1 matches it
        # Map provided sizes to the last len(vector_sizes) dims
        full_sizes = [1] * n_dims
        vs_len = min(len(vector_sizes), n_dims)
        for i in range(vs_len):
            full_sizes[n_dims - vs_len + i] = vector_sizes[i]

        # Clamp each vector size to a divisor of the corresponding dimension.
        # Non-divisible sizes produce dynamic remainder tiles whose
        # vectorization generates vector.mask ops wrapping multiple operations,
        # causing 'vector.mask op expects only one operation to mask'.
        dim_sizes = cls._get_dim_sizes(code)
        if dim_sizes:
            for i, sz in enumerate(full_sizes):
                if i < len(dim_sizes) and dim_sizes[i] > 0 and sz > 1:
                    full_sizes[i] = cls._clamp_to_divisor(dim_sizes[i], sz)

        # Check vector product limit
        if prod(s for s in full_sizes if s > 0) > VECTORIZATION_SIZE_LIMIT:
            return code

        # All tile sizes are non-zero (>=1), so tile_using_for creates
        # one loop per dimension. Count ALL of them for loop handles.
        n_tiled = n_dims
        if all(s <= 1 for s in full_sizes):
            # All sizes are 1, nothing useful to vectorize
            return code

        tile_str = str(full_sizes)
        vec_str = str(full_sizes)
        loop_handles = ", ".join(["!transform.any_op"] * n_tiled)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_tiled} = transform.structured.tile_using_for %op"
            f" tile_sizes {tile_str}"
            f" : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n"
            f"    transform.structured.vectorize %tiled_op"
            f" vector_sizes {vec_str} : !transform.any_op\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def _get_dim_sizes(cls, code: str) -> list[int]:
        """Extract iteration-space dimension sizes from the tagged operation.

        Parses tensor shapes from the operands of the tagged linalg.generic
        and uses the indexing maps to map tensor dimensions back to
        iteration-space dimensions.  Handles both inline affine maps and
        named map references (e.g. ``#map6``).

        Returns a list of dimension sizes (one per iterator), or [] if
        extraction fails.
        """
        import re

        # Collect named affine map definitions from the full module
        # e.g. #map6 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
        named_maps: dict[str, str] = {}
        for m in re.finditer(
            r'(#map\d*)\s*=\s*affine_map<\([^)]+\)\s*->\s*\(([^)]*)\)>', code
        ):
            named_maps[m.group(1)] = m.group(2)

        parts = re.split(r'(?=linalg\.generic)', code)
        for part in parts:
            if 'tag = "operation_0"' not in part:
                continue

            # Extract iterator_types to know how many dims
            m_iter = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', part)
            if not m_iter:
                continue
            iter_str = m_iter.group(1)
            n_dims = iter_str.count('"parallel"') + iter_str.count('"reduction"')

            # Extract indexing_maps
            m_maps = re.search(r'indexing_maps\s*=\s*\[([^\]]+)\]', part)
            if not m_maps:
                continue
            maps_str = m_maps.group(1)

            # Try inline affine maps first
            affine_maps = re.findall(
                r'affine_map<\([^)]+\)\s*->\s*\(([^)]*)\)>', maps_str
            )

            # If no inline maps, resolve named references
            if not affine_maps:
                map_refs = re.findall(r'#map\d*', maps_str)
                affine_maps = [
                    named_maps[ref] for ref in map_refs if ref in named_maps
                ]

            # Extract tensor shapes from ins(...) and outs(...)
            tensor_shapes = re.findall(r'tensor<([^>]+)>', part)

            if not affine_maps or not tensor_shapes:
                continue

            dim_sizes = [0] * n_dims
            for map_idx, amap in enumerate(affine_maps):
                if map_idx >= len(tensor_shapes):
                    break
                # Parse map dims: "d0, d2" -> [0, 2]
                map_dims = [
                    int(d.strip()[1:])
                    for d in amap.split(',')
                    if d.strip().startswith('d')
                ]
                # Parse shape: "128x256xf32" -> [128, 256]
                shape_parts = tensor_shapes[map_idx].split('x')
                shape = []
                for s in shape_parts:
                    try:
                        shape.append(int(s))
                    except ValueError:
                        continue  # skip the dtype part (e.g. "f32")
                # Map shape dims to iteration-space dims
                for dim_pos, map_dim in enumerate(map_dims):
                    if dim_pos < len(shape) and map_dim < n_dims:
                        if dim_sizes[map_dim] == 0:
                            dim_sizes[map_dim] = shape[dim_pos]

            if any(d > 0 for d in dim_sizes):
                return dim_sizes

        # Fallback for named ops: extract from tensor types in the function signature
        # matmul: tensor<MxKxT> @ tensor<KxNxT> -> tensor<MxNxT> => dims [M, N, K]
        if "linalg.matmul" in code and "batch" not in code:
            shapes = re.findall(r'tensor<(\d+x\d+x\w+)>', code)
            if len(shapes) >= 3:
                a_shape = [int(x) for x in shapes[0].split('x')[:-1]]
                b_shape = [int(x) for x in shapes[1].split('x')[:-1]]
                if len(a_shape) == 2 and len(b_shape) == 2:
                    return [a_shape[0], b_shape[1], a_shape[1]]  # M, N, K

        return []

    @classmethod
    def _clamp_to_divisor(cls, dim_size: int, candidate: int) -> int:
        """Return the largest value from VOCAB that divides dim_size and is <= candidate.

        If no VOCAB value > 1 divides dim_size, returns 1.
        """
        best = 1
        for v in cls.VOCAB:
            if v > candidate:
                break
            if dim_size % v == 0:
                best = v
        return best

    @classmethod
    def _count_dims(cls, code: str) -> int:
        """Count the number of iterator dimensions of the tagged op."""
        import re

        # Find the linalg.generic block containing the tag
        # Split at each linalg.generic and find the one with the tag
        parts = re.split(r'(?=linalg\.generic)', code)
        for part in parts:
            if 'tag = "operation_0"' in part:
                m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', part)
                if m:
                    types = m.group(1)
                    return types.count('"parallel"') + types.count('"reduction"')

        # Fallback: look for any iterator_types with the tag nearby
        m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\].*?tag\s*=\s*"operation_0"', code, re.DOTALL)
        if m:
            types = m.group(1)
            return types.count('"parallel"') + types.count('"reduction"')

        # For named linalg ops, infer from the op name
        if "conv_2d_nchw_fchw" in code:
            return 7
        if "linalg.matmul" in code:
            return 3
        if "linalg.batch_matmul" in code:
            return 4

        return 0

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        # Check that vectorization actually produced vector types
        if "vector<" not in after and "vector." not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp total product to VECTORIZATION_SIZE_LIMIT
        while sizes and prod(sizes) > VECTORIZATION_SIZE_LIMIT:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] = max(1, sizes[max_idx] // 2)
        return {"vector_sizes": sizes}
