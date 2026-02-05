from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class VectorizationAction(ActionBase):

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_size": {
                "description": "SIMD vector width for innermost dimension vectorization",
                "type": "int",
                "default": 4,
                "values": [2, 4, 8, 16]
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Check that the target tag exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Check vector_size is valid
        vector_size = params.get("vector_size", 4)
        if vector_size < 2 or vector_size > 64:
            return False
        
        # Check it's not a no-op
        if vector_size == 1:
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_size = params.get("vector_size", 4)
        
        # Strategy: Tile to small sizes, then vectorize
        # For matmul (3D): tile to [1, vector_size, 1]
        # For conv (7D): tile last dimensions
        # For generic: tile last dimension to vector_size
        
        # Try to determine the operation type from the code
        if "linalg.matmul" in code:
            tile_sizes = [1, vector_size, 1]
            n_loops = 3
        elif "linalg.conv_2d" in code:
            # Conv has many dimensions, vectorize output width
            tile_sizes = [1, 1, 1, vector_size]
            n_loops = 4
        elif "linalg.generic" in code:
            # For generic, we need to count the number of dimensions
            # Look for iterator_types
            match = re.search(r'iterator_types\s*=\s*\[(.*?)\]', code, re.DOTALL)
            if match:
                iter_types = match.group(1)
                n_dims = len(re.findall(r'"parallel"|"reduction"', iter_types))
                # Tile only the last dimension
                tile_sizes = [0] * (n_dims - 1) + [vector_size]
                n_loops = 1
            else:
                return code  # Can't determine structure
        else:
            return code  # Unknown operation type
        
        # Build the transform IR
        r_loops = ', '.join(['!transform.any_op'] * n_loops)
        
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op_operation_0 = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op_operation_0'
            f' tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {r_loops})\n'
            f'    transform.structured.vectorize %tiled_op : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        
        try:
            return run_transform_code(code, transform_code)
        except:
            return code  # If transform fails, return original

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        # No-op check
        if after.strip() == before.strip():
            return False
        
        # Basic sanity checks
        if not after or "func.func" not in after:
            return False
        
        # Check for excessive vector sizes
        vector_pattern = r'vector<([\dx]+)xf\d+>'
        for match in re.finditer(vector_pattern, after):
            dims_str = match.group(1)
            dims = [int(d) for d in dims_str.split('x')]
            total_elements = 1
            for d in dims:
                total_elements *= d
            
            # Check against element type bounds
            if 'f64' in match.group(0) or 'i64' in match.group(0):
                if total_elements > 16:
                    return False
            elif 'f32' in match.group(0) or 'i32' in match.group(0):
                if total_elements > 32:
                    return False
            elif 'f16' in match.group(0) or 'bf16' in match.group(0) or 'i16' in match.group(0):
                if total_elements > 64:
                    return False
            elif 'i8' in match.group(0):
                if total_elements > 128:
                    return False
            
            # Check rank
            if len(dims) > 3:
                return False
        
        return True