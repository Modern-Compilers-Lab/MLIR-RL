def transform_data_layout_reorder(code, permutation):
    # permutation: list of loop indices to reorder
    # e.g., for matmul: [0, 2, 1] means i, k, j instead of i, j, k
    # This changes the effective memory access pattern