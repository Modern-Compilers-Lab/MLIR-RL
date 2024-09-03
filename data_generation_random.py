from torch import NoneType
from utils.observation_utils import (
    function_wrapper,
    lower_linalg_to_loops,
    get_nested_loops_data,
    transform_wrapper
)
from utils.transforms import evaluate_code_with_timeout
from random import randint, choice, shuffle, random
from tqdm import tqdm
import json
from copy import copy
import numpy as np
import yaml
import argparse



tmp_file = 'tmp_files/temp_mlir.mlir'

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
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(4)])
    return f"linalg.add ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


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
    N = choice(SIZES)
    K = choice(SIZES)
    M = choice(SIZES)
    return f"linalg.matmul ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%arg2 : tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


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

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nchw_fchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}xf32>"


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

    return f"linalg.pooling_nchw_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


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
    
    if random() < 0.25:
        
        N = choice(BATCH_SIZES)
        S = choice(SIZES)
        SHAPE = f"{N}x{S}"
        
        relu_maps = """
        #map2 = affine_map<(d0, d1) -> (d0, d1)>
        """.strip()

        relu_operation = """
        linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
            ^bb0(%in: f32, %out: f32):
            %cst_1 = arith.constant 0.000000e+00 : f32
            %46 = arith.cmpf ugt, %in, %cst_1 : f32
            %47 = arith.select %46, %in, %cst_1 : f32
            linalg.yield %47 : f32
        } -> tensor<SHAPExf32>
        """.strip().replace('SHAPE', SHAPE)
        
    else:
        
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)
        
        SHAPE = f"{N}x{C}x{W}x{W}"
    
        relu_maps = """
        #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
        """.strip()

        relu_operation = """
        linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<SHAPExf32>) outs(%25 : tensor<SHAPExf32>) {
            ^bb0(%in: f32, %out: f32):
            %cst_1 = arith.constant 0.000000e+00 : f32
            %90 = arith.cmpf ugt, %in, %cst_1 : f32
            %91 = arith.select %90, %in, %cst_1 : f32
            linalg.yield %91 : f32
        } -> tensor<SHAPExf32>
        """.strip().replace('SHAPE', SHAPE)
        
    
    return relu_operation, relu_maps


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



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process an input file and save to an output file.')
    parser.add_argument('--input_file', type=str, help='The input file to be processed.')
    parser.add_argument('--output_file', type=str, help='The file where the processed content will be saved.')

    args = parser.parse_args()
    
    with open(args.input_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set the shapes of the operations 
    BATCH_SIZES.extend(config['SHAPES']['BATCH_SIZES']) # Used by all the operations
    HEIGHTS.extend(config['SHAPES']['HEIGHTS']) # Used by operations on images
    CHANNELS.extend(config['SHAPES']['CHANNELS']) # Used by operations on images
    KERNELS.extend(config['SHAPES']['KERNELS']) # Used by operations on images
    DILATIONS.extend(config['SHAPES']['DILATIONS']) # Used by operations on images
    STRIDES.extend(config['SHAPES']['STRIDES']) # Used by operations on images
    SIZES.extend(config['SHAPES']['SIZES']) # Used on other operations like matmul, add, etc...
    

    operations_config = {
        operation_name: (LINALG_OPERATION_GENERATORS[operation_name], amount) for operation_name, amount in config['OPERATIONS'].items() if amount > 0
    }
        
    print( sum( amount for operation_name, (generator, amount) in operations_config.items() ) )

    all_operations = {}

    for operation_name, (generator, amount) in tqdm(operations_config.items(), desc="linalg operations"):

        # Iterate the specified number of times ('amount') for the current operation
        for i in tqdm(range(amount), desc=operation_name):
            
            exec_time = None  # Initialize execution time as None to enter the loop
            
            # Loop until a valid execution time is obtained
            while exec_time is None:
                maps = None
                res = generator()  # Generate the raw operation using the provided generator function

                if isinstance(res, tuple):
                    raw_operation, maps = res
                else:
                    raw_operation = res 
            
                # Wrap the raw operation with additional functionality or settings
                wrapped_operation = function_wrapper(raw_operation, maps=maps)
                
                # Lower the wrapped operation into loops and retrieve loop data
                loops = lower_linalg_to_loops(wrapped_operation, tmp_file)            
                loops_data = get_nested_loops_data(loops)
                
                transform_wrapped_operation = transform_wrapper(raw_operation, maps=maps)

                # Evaluate the execution time of the transformed operation with a timeout of 300 seconds
                exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file)

                # If the execution time is valid and below a certain threshold, calculate a more stable median execution time
                if exec_time and exec_time < 1000000:
                    exec_time = np.median([exec_time] + [evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file) for _ in range(2)])
            
            
            
            # If a valid execution time was obtained, store the operation details
            if exec_time:
                all_operations[f"{raw_operation}"] = {
                    "operation": raw_operation,  # The raw operation
                    "wrapped_operation": wrapped_operation,  # The wrapped version of the operation
                    "lowered_operation": loops,  # The lowered loops of the operation
                    "transform_wrapped_operation": transform_wrapped_operation,  # The transformed wrapped operation
                    "loops_data": loops_data,  # Data related to the loops in the operation
                    "execution_time": exec_time,  # The median execution time
                }
            else:
                continue  # If no valid execution time, skip to the next iteration
        
        # Write all the collected operation data to the output file in JSON format
        with open(args.output_file, "w") as file:
            json.dump(all_operations, file)
