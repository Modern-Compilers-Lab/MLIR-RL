import re
from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class VectorizationAction(ActionBase):
    
    @classmethod
    def parameters(cls) -> dict:
        return {}
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Check that the target tag exists
        if 'tag = "operation_0"' not in code:
            return False
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1'
            ' : (!transform.any_op) -> !transform.any_op\n'
            '    transform.structured.vectorize %op : !transform.any_op\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        
        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            # If transform fails, return original code
            return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        # Check that the IR actually changed
        if after.strip() == before.strip():
            return False
        
        # Extract all vector types and validate sizes
        vector_pattern = r'vector<([0-9x]+)x([a-z0-9]+)>'
        vectors = re.findall(vector_pattern, after)
        
        for vec_dims, elem_type in vectors:
            # Calculate total vector size
            dims = [int(d) for d in vec_dims.split('x') if d.isdigit()]
            if not dims:
                continue
                
            total_size = 1
            for d in dims:
                total_size *= d
            
            # Check rank
            rank = len(dims)
            if rank >= 4:
                return False
            
            # Check size limits based on element type
            if elem_type in ['f64', 'i64']:
                if total_size > 16:
                    return False
            elif elem_type in ['f32', 'i32']:
                if total_size > 32:
                    return False
            elif elem_type in ['f16', 'bf16', 'i16']:
                if total_size > 64:
                    return False
            elif elem_type == 'i8':
                if total_size > 128:
                    return False
        
        # Check that the result is still valid MLIR
        if 'func.func' not in after:
            return False
        
        return True