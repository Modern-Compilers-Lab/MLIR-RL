from random import randint, choice, shuffle, random
from tqdm import trange
import json
import numpy as np
import yaml
import argparse
import re
import os
import sys
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager


def remove_duplicate_args(args, shapes):
    args_shapes = list(zip(args, shapes))
    seen = set()
    result = []
    for item in args_shapes:
        if item not in seen:
            seen.add(item)
            result.append(item)

    args = [x for (x, _) in result]
    shapes = [x for (_, x) in result]
    return args, shapes


pass_pipeline = """builtin.module(
    loop-invariant-code-motion,
    canonicalize,
    convert-vector-to-scf,
    convert-linalg-to-loops,
    buffer-deallocation-pipeline,
    scf-forall-to-parallel,
    convert-scf-to-openmp,
    expand-strided-metadata,
    finalize-memref-to-llvm,
    convert-scf-to-cf,
    lower-affine,

    convert-openmp-to-llvm,
    convert-vector-to-llvm,
    convert-math-to-llvm,
    convert-func-to-llvm,
    convert-index-to-llvm,
    convert-arith-to-llvm,
    convert-cf-to-llvm,

    reconcile-unrealized-casts,
    canonicalize,
    cse
)"""

tmp_file = 'temp.mlir'

BATCH_SIZES = []
SIZES = []
HEIGHTS = []
CHANNELS = []
KERNELS = []
DILATIONS = []
STRIDES = []


def choice_topped(choices, max_value):
    trials_left = 50
    n = choice(choices)
    while not (n <= max_value) and trials_left != 0:
        n = choice(choices)
        trials_left -= 1

    if trials_left == 0:
        return None
    return n


def add():
    # SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 3))])
    matmul_size = 10 ** 9
    while matmul_size >= (10 ** 9):
        shape_int = [choice(HEIGHTS) for _ in range(4)]
        matmul_size = shape_int[0] * shape_int[1] * shape_int[2] * shape_int[3]
    SHAPE = "x".join(map(str, shape_int))
    shape_str = SHAPE.replace('x', '_')
    bench_name = f"add_{shape_str}"
    return f"linalg.add ins(%arg0, %arg1: memref<{SHAPE}xf64>, memref<{SHAPE}xf64>) outs(%arg2: memref<{SHAPE}xf64>)", bench_name


def add_nn():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    operation = f"""
    linalg.generic {{indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]}} ins(%44, %10 : tensor<{B}x{N}xf32>, tensor<{N}xf32>) outs(%42 : tensor<{B}x{N}xf32>) {{
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %46 = arith.addf %in, %in_1 : f32
      linalg.yield %46 : f32
    }}
    """.strip()
    return operation


def sub():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.sub ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def max():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.max ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def mul():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.mul ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def abs():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.abs ins(%arg0: tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def ceil():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.ceil ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def copy_():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.copy ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def fill():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.fill ins(%arg0 : f32) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def transpose():
    L = randint(1, 5)

    permutation = list(range(L))
    shuffle(permutation)

    SHAPE1 = [choice(HEIGHTS) for _ in range(L)]

    SHAPE2 = []
    for i in range(L):
        SHAPE2.append(SHAPE1[permutation[i]])

    SHAPE1 = "x".join(map(str, SHAPE1))
    SHAPE2 = "x".join(map(str, SHAPE2))

    return f"linalg.transpose ins(%input:tensor<{SHAPE1}xf32>) outs(%init:tensor<{SHAPE2}xf32>) permutation = {permutation}"


def batch_matmul():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2 : tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_a():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_a ins(%arg0, %arg1: tensor<{B}x{K}x{N}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_b():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{M}x{K}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_reduce_matmul():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul():
    matmul_size = 10 ** 9
    while matmul_size >= (10 ** 9):
        N = choice(SIZES)
        K = choice(SIZES)
        M = choice(SIZES)
        matmul_size = N * K * M
    bench_name = f"matmul_{N}_{K}_{M}"
    return f"linalg.matmul ins(%arg0, %arg1 : memref<{N}x{K}xf64>, memref<{K}x{M}xf64>) outs(%arg2 : memref<{N}x{M}xf64>)", bench_name


def matmul_transpose_a():
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_a ins(%arg0, %arg1: tensor<{K}x{N}xf32>, tensor<{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_b():
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{M}x{K}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def conv_1d():
    N = choice(HEIGHTS)
    F = choice_topped(KERNELS, N)
    N_ = N - F + 1
    return f"linalg.conv_1d ins(%input, %filter : tensor<{N}xf32>, tensor<{F}xf32>) outs(%output : tensor<{N_}xf32>) -> tensor<{N_}xf32>"


def conv_1d_ncw_fcw():
    # INPUT: NCW1
    # KERNL: FCW2
    # OUTPUT: (N, F, W1-W2+1)

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_ncw_fcw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W1}xf32>, tensor<{F}x{C}x{W2}xf32>) outs (%init: tensor<{N}x{F}x{W3}xf32>) -> tensor<{N}x{F}x{W3}xf32>"


def conv_1d_nwc_wcf():
    # INPUT: NWC
    # KERNL: WCF
    # OUTPUT: (N, W1-W2+1, F)

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_nwc_wcf {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W1}x{C}xf32>, tensor<{W2}x{C}x{F}xf32>) outs (%init: tensor<{N}x{W3}x{F}xf32>) -> tensor<{N}x{W3}x{F}xf32>"


def conv_2d():
    H, W = choice(HEIGHTS), choice(HEIGHTS)

    F1 = F2 = choice_topped(KERNELS, min(H - 2, W - 2))

    H_ = H - F1 + 1
    W_ = W - F2 + 1

    return f"linalg.conv_2d ins(%input, %filter: tensor<{H}x{W}xi32>, tensor<{F1}x{F2}xi32>) outs(%output: tensor<{H_}x{W_}xi32>) -> tensor<{H_}x{W_}xi32>"


def conv_2d_nchw_fchw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    matmul_size = 10 ** 9
    while matmul_size >= (10 ** 9):
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        # W = choice(HEIGHTS)
        W = H

        dilation = choice(DILATIONS)
        stride = choice(STRIDES)
        padding = 0

        F = choice(CHANNELS)
        KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

        H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
        W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1

        matmul_size = N * F * H_ * W_ * C * KH * KW

    bench_name = f"conv_2d_nchw_fchw_{N}_{C}_{H}_{W}_{F}_{KH}_{KW}_{H_}_{W_}"

    return f"linalg.conv_2d_nchw_fchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf64>, tensor<{F}x{C}x{KH}x{KW}xf64>) outs (%init: tensor<{N}x{F}x{H_}x{W_}xf64>) -> tensor<{N}x{F}x{H_}x{W_}xf64>", bench_name


def conv_2d_ngchw_fgchw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    N = choice(BATCH_SIZES)
    G = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_ngchw_fgchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{G}x{C}x{H}x{W}xf32>, tensor<{G}x{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{G}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{G}x{F}x{H_}x{W_}xf32>"


def conv_2d_nhwc_fhwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_fhwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{F}x{KH}x{KW}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_2d_nhwc_hwcf():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_hwcf {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{KH}x{KW}x{C}x{F}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_3d():
    H, W, D = choice(HEIGHTS), choice(HEIGHTS), choice(HEIGHTS)

    F = choice_topped(KERNELS, min(H, W, D) - 2)

    H_ = H - F + 1
    W_ = W - F + 1
    D_ = D - F + 1

    return f"linalg.conv_3d ins(%input, %filter: tensor<{H}x{W}x{D}xf32>, tensor<{F}x{F}x{F}xf32>) outs(%output: tensor<{H_}x{W_}x{D_}xf32>) -> tensor<{H_}x{W_}x{D_}xf32>"


def conv_3d_ncdhw_fcdhw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = KD = choice_topped(
        KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1
    )

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (KD - 1) - 1) // stride) + 1

    return f"linalg.conv_3d_ncdhw_fcdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}x{D}xf32>, tensor<{F}x{C}x{KH}x{KW}x{KD}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>"


def depthwise_conv_1d_ncw_cw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_ncw_cw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{C}x{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def depthwise_conv_1d_nwc_wc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wc {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def depthwise_conv_1d_nwc_wcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wcm {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{W_}x{C}x{M}xf32>"


def depthwise_conv_2d_nchw_chw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nchw_chw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{C}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def depthwise_conv_2d_nhwc_hwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_2d_nhwc_hwcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwcm {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{H_}x{W_}x{C}x{M}xf32>"


def depthwise_conv_3d_ncdhw_cdhw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ncdhw_cdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{D}x{H}x{W}xf32>, tensor<{C}x{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>"


def depthwise_conv_3d_ndhwc_dhwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwc {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_3d_ndhwc_dhwcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwcm {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>"


def pooling_nchw_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    bench_name = f"pooling_nchw_max_{N}_{C}_{H}_{W}_{K}_{H_}_{W_}"

    return f"linalg.pooling_nchw_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf64>, tensor<{K}x{K}xf64>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf64>) -> tensor<{N}x{C}x{H_}x{W_}xf64>", bench_name


def pooling_nchw_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nchw_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def pooling_ncw_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ncw_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ndhwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_max {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_min():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_min {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_sum {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_min():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_min {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def pooling_nwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def relu():

    matmul_size = 10 ** 9
    while matmul_size >= (10 ** 9):
        if random() < 0.25:

            N = choice(BATCH_SIZES)
            S = choice(SIZES)
            SHAPE = f"{N}x{S}"

            relu_operation = (
                f"linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [\"parallel\", \"parallel\"]}} ins(%arg0: memref<{SHAPE}xf64>) outs(%arg1: memref<{SHAPE}xf64>) {{\n"
                f"  ^bb0(%in: f64, %out: f64):\n"
                f"  %cst_1 = arith.constant 0.0: f64\n"
                f"  %46 = arith.cmpf ugt, %in, %cst_1 : f64\n"
                f"  %47 = arith.select %46, %in, %cst_1 : f64\n"
                f"  linalg.yield %47 : f64\n"
                f"}}\n"
            )

            matmul_size = N * S

        else:

            N = choice(BATCH_SIZES)
            C = choice(CHANNELS)
            W = choice(HEIGHTS)

            SHAPE = f"{N}x{C}x{W}x{W}"

            relu_operation = (
                f"linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}} ins(%arg0 : memref<{SHAPE}xf64>) outs(%arg1 : memref<{SHAPE}xf64>) {{\n"
                f"  ^bb0(%in: f64, %out: f64):\n"
                f"  %cst_1 = arith.constant 0.0 : f64\n"
                f"  %90 = arith.cmpf ugt, %in, %cst_1 : f64\n"
                f"  %91 = arith.select %90, %in, %cst_1 : f64\n"
                f"  linalg.yield %91 : f64\n"
                f"}}\n"
            )

            matmul_size = N * C * W * W

    shape_str = SHAPE.replace('x', '_')
    bench_name = f"relu_{shape_str}"

    return relu_operation, bench_name


LINALG_OPERATION_GENERATORS = {
    "add": add,
    "add_nn": add_nn,
    "sub": sub,
    "max": max,
    "mul": mul,
    "abs": abs,
    "ceil": ceil,
    "copy": copy_,
    "fill": fill,
    "transpose": transpose,
    "batch_matmul": batch_matmul,
    "batch_matmul_transpose_a": batch_matmul_transpose_a,
    "batch_matmul_transpose_b": batch_matmul_transpose_b,
    "batch_reduce_matmul": batch_reduce_matmul,
    "matmul": matmul,
    "matmul_transpose_a": matmul_transpose_a,
    "matmul_transpose_b": matmul_transpose_b,
    "conv_1d": conv_1d,
    "conv_1d_ncw_fcw": conv_1d_ncw_fcw,
    "conv_1d_nwc_wcf": conv_1d_nwc_wcf,
    "conv_2d": conv_2d,
    "conv_2d_nchw_fchw": conv_2d_nchw_fchw,
    "conv_2d_ngchw_fgchw": conv_2d_ngchw_fgchw,
    "conv_2d_nhwc_fhwc": conv_2d_nhwc_fhwc,
    "conv_2d_nhwc_hwcf": conv_2d_nhwc_hwcf,
    "conv_3d": conv_3d,
    "conv_3d_ncdhw_fcdhw": conv_3d_ncdhw_fcdhw,
    "depthwise_conv_1d_ncw_cw": depthwise_conv_1d_ncw_cw,
    "depthwise_conv_1d_nwc_wc": depthwise_conv_1d_nwc_wc,
    "depthwise_conv_1d_nwc_wcm": depthwise_conv_1d_nwc_wcm,
    "depthwise_conv_2d_nchw_chw": depthwise_conv_2d_nchw_chw,
    "depthwise_conv_2d_nhwc_hwc": depthwise_conv_2d_nhwc_hwc,
    "depthwise_conv_2d_nhwc_hwcm": depthwise_conv_2d_nhwc_hwcm,
    "depthwise_conv_3d_ncdhw_cdhw": depthwise_conv_3d_ncdhw_cdhw,
    "depthwise_conv_3d_ndhwc_dhwc": depthwise_conv_3d_ndhwc_dhwc,
    "depthwise_conv_3d_ndhwc_dhwcm": depthwise_conv_3d_ndhwc_dhwcm,
    "pooling_nchw_max": pooling_nchw_max,
    "pooling_nchw_sum": pooling_nchw_sum,
    "pooling_ncw_max": pooling_ncw_max,
    "pooling_ncw_sum": pooling_ncw_sum,
    "pooling_ndhwc_max": pooling_ndhwc_max,
    "pooling_ndhwc_min": pooling_ndhwc_min,
    "pooling_ndhwc_sum": pooling_ndhwc_sum,
    "pooling_nhwc_max": pooling_nhwc_max,
    "pooling_nhwc_min": pooling_nhwc_min,
    "pooling_nhwc_sum": pooling_nhwc_sum,
    "pooling_nwc_max": pooling_nwc_max,
    "pooling_nwc_sum": pooling_nwc_sum,
    "relu": relu,
}


def extract_args(operation) -> tuple[list[str], list[str]]:
    ins_outs_pattern = r"(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip()[1:] for arg in args]
    shapes = [shape.strip() for shape in shapes]

    return remove_duplicate_args(args, shapes)


def extract_main_args(code) -> tuple[list[str], list[str]]:
    main_pattern = r"func.func @main\(([^)]+)\)"
    main_params = re.search(main_pattern, code).group(1)
    main_params = main_params.split(',')
    main_args = [arg.split(':')[0].strip() for arg in main_params]
    main_shapes = [arg.split(':')[1].strip() for arg in main_params]

    return remove_duplicate_args(main_args, main_shapes)


def transform_img2col(code: str):
    code = code.strip()

    transform_dilaect_code = (
        "module attributes {transform.with_named_sequence} {\n"
        "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
        "    %op_operation = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op\n"
        "    %a, %b = transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )

    code = code + transform_dilaect_code

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -finalizing-bufferize -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process an input file and save to an output file.')
    parser.add_argument('--input_file', type=str, help='The input file to be processed.')

    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        config = yaml.safe_load(file)

    # Set the shapes of the operations
    BATCH_SIZES.extend(config['SHAPES']['BATCH_SIZES'])  # Used by all the operations
    HEIGHTS.extend(config['SHAPES']['HEIGHTS'])  # Used by operations on images
    CHANNELS.extend(config['SHAPES']['CHANNELS'])  # Used by operations on images
    KERNELS.extend(config['SHAPES']['KERNELS'])  # Used by operations on images
    DILATIONS.extend(config['SHAPES']['DILATIONS'])  # Used by operations on images
    STRIDES.extend(config['SHAPES']['STRIDES'])  # Used by operations on images
    SIZES.extend(config['SHAPES']['SIZES'])  # Used on other operations like matmul, add, etc...

    operations_config = {
        operation_name: (LINALG_OPERATION_GENERATORS[operation_name], amount) for operation_name, amount in config['OPERATIONS'].items() if amount > 0
    }

    print("Expected Bench Count:", sum(amount for _, (_, amount) in operations_config.items()))

    with open('execution_times.json', 'r') as file:
        execution_times: dict[str, int] = json.load(file)
    total_count = 0
    for operation_name, (generator, amount) in operations_config.items():
        # Iterate the specified number of times ('amount') for the current operation
        count = 0
        tqdm_iter = trange(amount, desc=operation_name, file=sys.stdout)
        for i in tqdm_iter:
            raw_operation, bench_name = generator()  # Generate the raw operation using the provided generator function
            bench_output = f'../{bench_name}.mlir'
            if bench_name in execution_times:
                print('Already exists, Bench:', bench_name, file=sys.stderr)
                continue

            args, shapes = extract_args(raw_operation)
            is_memref = 'memref' in shapes[-1]
            main_params = [f"%{arg}: {shape}" for arg, shape in zip(args, shapes)]

            code = (
                f'func.func private @nanoTime() -> i64 attributes {{ llvm.emit_c_interface }}\n'
                f'func.func @main({", ".join(main_params)}) -> i64 attributes {{ llvm.emit_c_interface }} {{\n'
                f'  %t0 = func.call @nanoTime() : () -> i64\n'
                f"  {'' if is_memref else '%0 = '}{raw_operation}\n"
                f'  %t1 = func.call @nanoTime() : () -> i64\n'
                f'  %t2 = arith.subi %t1, %t0 : i64\n'
                f'  return %t2 : i64\n'
                f'}}\n'
            )
            if not is_memref:
                code = transform_img2col(code)
                args, shapes = extract_main_args(code)
                assert all('memref' in shape for shape in shapes)

            with Context():
                module = Module.parse(code)
                pm = PassManager.parse(pass_pipeline)
                pm.run(module.operation)
            execution_engine = ExecutionEngine(
                module,
                shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
            )

            args_count = len(args)
            inputs: dict[str, np.ndarray] = {}
            for i, (arg, shape) in enumerate(zip(args, shapes)):
                *np_shape, dtype = shape.replace('memref<', '').replace('>', '').split('x')
                assert dtype == 'f64', f'expects f64, got {dtype}'
                np_shape = list(map(int, np_shape))
                inputs[arg] = np.zeros(np_shape)

            c_args = []
            for arg_name in args:
                c_args.append(ctypes.pointer(ctypes.pointer(
                    get_ranked_memref_descriptor(inputs[arg_name])
                )))
            delta_arg = (ctypes.c_int64 * 1)(0)
            c_args.append(delta_arg)

            execution_engine.invoke("main", *c_args)
            execution_engine.invoke("main", *c_args)

            exec_time = delta_arg[0]

            with open(bench_output, 'w') as file:
                file.write(code)

            execution_times[bench_name] = exec_time
            with open('execution_times.json', 'w') as file:
                json.dump(execution_times, file, indent=4)

            tqdm_iter.set_postfix({'shape': str(shape), 'time': exec_time})
