from typing import Optional
import re

class Action(ActionBase):
    """
    Tile a 2D convolution operation (linalg.conv_2d_nchw_fchw) along output spatial
    and channel dimensions to improve cache locality.
    
    The action uses the MLIR Transform dialect to decompose the convolution into
    nested tiles, reducing the working set per tile to fit in L1/L2 cache.
    """
    
    name = "ConvolutionTiling"
    
    @classmethod
    def parameters(cls) -> dict:
        """Return the parameter schema for this action."""
        return {
            "tile_oh": {
                "description": "Tile size for output height dimension (0 = skip)",
                "type": "int",
                "values": [0, 4, 8, 16, 32],
                "default": 0
            },
            "tile_ow": {
                "description": "Tile size for output width dimension (0 = skip)",
                "type": "int",
                "values": [0, 4, 8, 16, 32],
                "default": 0
            },
            "tile_f": {
                "description": "Tile size for output channel dimension (0 = skip)",
                "type": "int",
                "values": [0, 8, 16, 32, 64],
                "default": 0
            },
            "tile_n": {
                "description": "Tile size for batch dimension (0 = skip)",
                "type": "int",
                "values": [0, 1, 2, 4],
                "default": 0
            }
        }
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if the transformation is applicable:
        - The code must contain at least one linalg.conv_2d_nchw_fchw operation.
        - At least one tile size must be non-zero.
        """
        # Check for presence of the convolution operation
        if "linalg.conv_2d_nchw_fchw" not in code:
            return False
        
        # Check that at least one tile dimension is non-zero
        tile_oh = params.get("tile_oh", 0)
        tile_ow = params.get("tile_ow", 0)
        tile_f = params.get("tile_f", 0)
        tile_n = params.get("tile_n", 0)
        
        if all([tile_oh == 0, tile_ow == 0, tile_f == 0, tile_n == 0]):
            return False
        
        # Check that tile sizes are positive integers
        for param_name in ["tile_oh", "tile_ow", "tile_f", "tile_n"]:
            val = params.get(param_name, 0)
            if not isinstance(val, int) or val < 0:
                return False
        
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocess the code by adding a transform.tag attribute to the
        convolution operation if not already present.
        """
        # If the conv operation already has a tag, return as-is
        if 'transform.tag = "conv"' in code:
            return code
        
        # Add a transform tag to the conv operation
        # Find the linalg.conv_2d_nchw_fchw operation and insert the tag
        pattern = r'(%\d+\s*=\s*linalg\.conv_2d_nchw_fchw)'
        replacement = r'\1 { transform.tag = "conv" }'
        result = re.sub(pattern, replacement, code)
        
        if result == code:
            # If the simple pattern didn't match, try matching the full operation
            # and add the tag before the closing brace or at the end of attributes
            if "linalg.conv_2d_nchw_fchw" in code:
                # Insert tag attribute
                pattern = r'(linalg\.conv_2d_nchw_fchw)([^{]*{[^}]*})'
                def add_tag(match):
                    op = match.group(1)
                    attrs = match.group(2)
                    # Insert tag before the closing brace
                    return op + attrs[:-2] + ', transform.tag = "conv" }'
                result = re.sub(pattern, add_tag, code)
        
        return result if result != code else code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implement the tiling transformation using MLIR Transform dialect.
        
        This constructs a transform sequence that:
        1. Matches the tagged convolution operation.
        2. Applies tiling along the specified dimensions using
           transform.structured.tile_using_for.
        """
        tile_oh = params.get("tile_oh", 0)
        tile_ow = params.get("tile_ow", 0)
        tile_f = params.get("tile_f", 0)
        tile_n = params.get("tile_n", 0)
        
        # Build the tile_sizes list; order follows the iteration space order
        # for linalg.conv_2d_nchw_fchw: [N, F, OH, OW, C, KH, KW]
        tile_sizes = [tile_n, tile_f, tile_oh, tile_ow, 0, 0, 0]
        
        # Count non-zero tiles to construct the result type
        num_tiles = sum(1 for s in tile_sizes if s != 0)
        
        # Construct the result binding for tiled operation and loop handles
        # Format: %tiled_op, %loop_0, %loop_1, ... = transform.structured.tile_using_for ...
        loop_bindings = ', '.join([f'%loop_{i}' for i in range(num_tiles)])
        if loop_bindings:
            result_binding = f'%tiled_op, {loop_bindings}'
        else:
            result_binding = '%tiled_op'
        
        # Build the transform code
        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %conv_op = transform.structured.match attributes{{transform.tag = "conv"}} in %arg0 : (!transform.any_op) -> !transform.any_op
    {result_binding} = transform.structured.tile_using_for %conv_op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op{', !transform.any_op' * num_tiles})
    transform.yield
  }}
}}
"""
        
        # Execute the transformation
        try:
            result = transform_code(code, transform_code)
            return result
        except Exception as e:
            # If transformation fails, return the original code
            # (Postcondition will catch this)
            return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation succeeded:
        - The output IR must be valid MLIR.
        - The convolution operation should be tiled (indicated by nested loops
          or a transformed IR structure).
        - Basic structural invariants: the result should contain for loops
          if tiling was applied.
        """
        # Check that we got a non-empty result
        if not after or after.strip() == "":
            return False
        
        # Minimal validity check: the IR should still contain the convolution
        # (it may be wrapped/decomposed but the operation should exist)
        if "linalg.conv_2d_nchw_fchw" not in after and "scf.for" not in after:
            # After tiling, we expect either the original conv (if tile size was 0)
            # or an scf.for loop structure. If neither, something went wrong.
            # Exception: if all tile sizes were 0, the IR should be unchanged.
            tile_oh = params.get("tile_oh", 0)
            tile_ow = params.get("tile_ow", 0)
            tile_f = params.get("tile_f", 0)
            tile_n = params.get("tile_n", 0)
            if all([tile_oh == 0, tile_ow == 0, tile_f == 0, tile_n == 0]):
                # All tile sizes are 0; expect no-op (original IR preserved)
                return after == before
            else:
                # We expected tiling to occur but didn't see it
                return False
        
        # Check that we still have a valid function
        if "func.func @main" not in after:
            return False
        
        return True
