from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re

class VectorizationAction(ActionBase):
    
    @classmethod
    def parameters(cls) -> dict:
        return {
            "vectorize": {
                "type": "bool",
                "default": True,
                "description": "Whether to apply vectorization (must be True for action to apply)"
            }
        }
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """Check if vectorization should be attempted."""
        # Must be enabled
        if not params.get("vectorize", True):
            return False
        
        # Must have the required tag
        if 'tag = "operation_0"' not in code:
            return False
        
        # Basic sanity: must contain a linalg operation
        if "linalg." not in code:
            return False
        
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """No preprocessing needed."""
        return code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """Apply vectorization using Transform dialect."""
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %matched = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.vectorize %matched : !transform.any_op\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            # If transform fails, return original code (postcondition will catch it)
            return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """Verify vectorization succeeded and produced safe vectors."""
        # Must have changed the IR
        if after.strip() == before.strip():
            return False
        
        # Basic sanity checks
        if not after or "func.func" not in after:
            return False
        
        # Vectorization safety check: scan for vector types and validate sizes
        vector_pattern = r'vector<([^>]+)>'
        vectors = re.findall(vector_pattern, after)
        
        for vec_spec in vectors:
            # Parse vector dimensions and element type
            # Format: "4x8xf32" or "16xi64" etc.
            parts = vec_spec.replace('x', ' ').split()
            if not parts:
                continue
            
            # Last part is the element type
            elem_type = parts[-1]
            
            # Earlier parts are dimensions (if any)
            dims = []
            for p in parts[:-1]:
                if p.isdigit():
                    dims.append(int(p))
            
            # Calculate total element count
            if dims:
                total_elements = 1
                for d in dims:
                    total_elements *= d
                
                # Determine limit based on element type
                if 'f64' in elem_type or 'i64' in elem_type:
                    limit = 16
                elif 'f32' in elem_type or 'i32' in elem_type:
                    limit = 32
                elif 'f16' in elem_type or 'bf16' in elem_type or 'i16' in elem_type:
                    limit = 64
                elif 'i8' in elem_type:
                    limit = 128
                else:
                    # Unknown type, use conservative limit
                    limit = 16
                
                # Check if exceeds limit
                if total_elements > limit:
                    return False
                
                # Check rank (prefer rank-1, allow small rank-2/3, reject rank>=4)
                rank = len(dims)
                if rank >= 4:
                    return False
                
                # For rank-3, be extra conservative
                if rank == 3 and total_elements > limit // 2:
                    return False
        
        return True