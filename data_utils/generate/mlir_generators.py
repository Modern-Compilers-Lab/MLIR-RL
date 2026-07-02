import traceback
from random import randint, choice, shuffle, random, seed

import re
import string

def _remove_duplicate_args(args: list[str], shapes: list[str]):
    """Remove duplicate (arg, shape) pairs while preserving order."""
    seen = set()
    result = []
    for pair in zip(args, shapes):
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    args = [a for a, _ in result]
    shapes = [s for _, s in result]
    return args, shapes

def choice_topped(choices, max_value):
    trials_left = 50
    n = choice(choices)
    while not (n <= max_value) and trials_left != 0:
        n = choice(choices)
        trials_left -= 1

    if trials_left == 0:
        return None
    return n

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SIZES = [64, 128, 256, 512, 768, 1024]
HEIGHTS = [32, 64, 112, 128, 224, 256]
CHANNELS = [3, 16, 32, 64, 128, 256, 512, 1024]
KERNELS = [1, 3, 5, 7]
DILATIONS = [1, 2, 3]
STRIDES = [1, 2]

def add(*args):
    if args:
        SHAPE = "x".join(list(map(str,args[0])))
    else:  
    # SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 3))])
        SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(4)])
    return f"linalg.add ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def add_nn(*args):
    if args:
        if len(args[0])==2:
            B,N = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:             
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


def sub(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.sub ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def linalg_max(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.max ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def mul(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.mul ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def linalg_abs(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.abs ins(%arg0: tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def ceil(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.ceil ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def copy_(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.copy ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def fill(*args):
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.fill ins(%arg0 : f32) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def transpose(*args):
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


def batch_matmul(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)
    
    M = choice(HEIGHTS)

    return f"linalg.batch_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2 : tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_a(*args):
    if args:
        if len(args[0]) == 3:
            B,K,N = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)
        
    M = choice(HEIGHTS)

    return f"linalg.batch_matmul_transpose_a ins(%arg0, %arg1: tensor<{B}x{K}x{N}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_b(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)

    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{M}x{K}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_reduce_matmul(*args):
    if args:
        if len(args[0]) == 3:
            B,N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        B = choice(BATCH_SIZES)
        N = choice(HEIGHTS)
        K = choice(HEIGHTS)

    M = choice(HEIGHTS)
    return f"linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul(*args):
    if args:
        if len(args[0]) == 2:
            N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)
        
    M = choice(SIZES)

    return f"linalg.matmul ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%arg2 : tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_a(*args):
    if args:
        if len(args[0]) == 2:
            K,N = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)

    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_a ins(%arg0, %arg1: tensor<{K}x{N}xf32>, tensor<{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_b(*args):
    if args:
        if len(args[0]) == 2:
            N,K = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
        N = choice(SIZES)
        K = choice(SIZES)

    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{M}x{K}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def conv_1d(*args):
    if args:
        if len(args[0]) == 1:
            N = args[0][0]
        else:
            raise Exception("Skipped")
    else:
        N = choice(HEIGHTS)

    F = choice_topped(KERNELS, N)
    N_ = N - F + 1
    return f"linalg.conv_1d ins(%input, %filter : tensor<{N}xf32>, tensor<{F}xf32>) outs(%output : tensor<{N_}xf32>) -> tensor<{N_}xf32>"


def conv_1d_ncw_fcw(*args):
    # INPUT: NCW1
    # KERNL: FCW2
    # OUTPUT: (N, F, W1-W2+1)

    if args:
        if len(args[0]) == 3:
            N,C,W1 = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
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


def conv_1d_nwc_wcf(*args):
    # INPUT: NWC
    # KERNL: WCF
    # OUTPUT: (N, W1-W2+1, F)

    if args:
        if len(args[0]) == 3:
            N,W1,C = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
            
    else:
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


def conv_2d(*args):
    if args:
        if len(args[0]) == 2:
            H,W = tuple(args[0])
        else:
            raise Exception("given shape is not accepted")
    else:
        H, W = choice(HEIGHTS), choice(HEIGHTS)

    F1 = F2 = choice_topped(KERNELS, min(H - 2, W - 2))

    H_ = H - F1 + 1
    W_ = W - F2 + 1

    return f"linalg.conv_2d ins(%input, %filter: tensor<{H}x{W}xi32>, tensor<{F1}x{F2}xi32>) outs(%output: tensor<{H_}x{W_}xi32>) -> tensor<{H_}x{W_}xi32>"


def conv_2d_nchw_fchw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
        
        if H != W:            
            raise Exception("given shape is not accepted")
            
    else:
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


def conv_2d_ngchw_fgchw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,G,C,H,W = tuple(args[0])
            
    else:
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


def conv_2d_nhwc_fhwc(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
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


def conv_2d_nhwc_hwcf(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
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


def conv_3d(*args):
    if args:
        if len(args[0]) == 3:
            H,W,D = tuple(args[0])
        else:
            raise Exception("skipped")
    else:
        H, W, D = choice(HEIGHTS), choice(HEIGHTS), choice(HEIGHTS)

    F = choice_topped(KERNELS, min(H, W, D) - 2)

    H_ = H - F + 1
    W_ = W - F + 1
    D_ = D - F + 1

    return f"linalg.conv_3d ins(%input, %filter: tensor<{H}x{W}x{D}xf32>, tensor<{F}x{F}x{F}xf32>) outs(%output: tensor<{H_}x{W_}x{D_}xf32>) -> tensor<{H_}x{W_}x{D_}xf32>"


def conv_3d_ncdhw_fcdhw(*args):
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,C,H,W,D = tuple(args[0])
            
    else:
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


def depthwise_conv_1d_ncw_cw(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W= tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_ncw_cw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{C}x{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def depthwise_conv_1d_nwc_wc(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C= tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wc {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def depthwise_conv_1d_nwc_wcm(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)
    
    M = choice(CHANNELS)
    

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wcm {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{W_}x{C}x{M}xf32>"


def depthwise_conv_2d_nchw_chw(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
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


def depthwise_conv_2d_nhwc_hwc(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
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


def depthwise_conv_2d_nhwc_hwcm(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        H = choice(HEIGHTS)
        W = choice(HEIGHTS)

    M = choice(CHANNELS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwcm {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{H_}x{W_}x{C}x{M}xf32>"


def depthwise_conv_3d_ncdhw_cdhw(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,C,D,H,W = tuple(args[0])
            
    else:
    
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


def depthwise_conv_3d_ndhwc_dhwc(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,D,H,W,C = tuple(args[0])
            
    else:    
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


def depthwise_conv_3d_ndhwc_dhwcm(*args):
    if args:
        if len(args[0]) != 5:
            raise Exception("given shape is not accepted")
            
        N,D,H,W,C = tuple(args[0])
            
    else:
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)
        H = choice(HEIGHTS)
        D = choice(HEIGHTS)
    
    M = choice(CHANNELS)
    
    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwcm {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>"


def pooling_nchw_max(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
    
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


def pooling_nchw_sum(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,C,H,W = tuple(args[0])
            
    else:
    
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


def pooling_ncw_max(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ncw_sum(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,C,W = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ndhwc_max(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
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


def pooling_ndhwc_min(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
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


def pooling_ndhwc_sum(*args):
    if args:
        if len(args[0]) == 5:
            N,D,H,W,C = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
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


def pooling_nhwc_max(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
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


def pooling_nhwc_min(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
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


def pooling_nhwc_sum(*args):
    if args:
        if len(args[0]) != 4:
            raise Exception("given shape is not accepted")
            
        N,H,W,C = tuple(args[0])
            
    else:
    
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


def pooling_nwc_max(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def pooling_nwc_sum(*args):
    if args:
        if len(args[0]) != 3:
            raise Exception("given shape is not accepted")
            
        N,W,C = tuple(args[0])
            
    else:
    
        N = choice(BATCH_SIZES)
        C = choice(CHANNELS)
        W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"




def relu(*args):
    if args:
        if len(args[0]) not in [2,4]:
            raise Exception("Skipped")

        if len(args[0]) == 4:   
            N,C,W,W_ = tuple(args[0])
            dim = 4

            if W != W_:
                raise Exception("Skipped")

        elif len(args[0]) == 2:
            N,S = tuple(args[0])
            dim = 2
        
        else:
            raise Exception("Skipped")
    else: 
        if random() < 0.25:
            N = choice(BATCH_SIZES)
            S = choice(CHANNELS)

            dim = 2

        else:
            N = choice(BATCH_SIZES)
            C = choice(CHANNELS)
            W = choice(HEIGHTS)
        
            dim = 4

    if dim == 2:
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


def sigmoid(*args):
    # Always 2D tensor

    if args:
        if len(args[0])==2:
            N,S = tuple(args[0])
        else:
            raise Exception("Skipped")
    else:
        N = choice(BATCH_SIZES)
        S = choice(SIZES)
    
    SHAPE = f"{N}x{S}"

    sigmoid_maps = """
    #map2 = affine_map<(d0, d1) -> (d0, d1)>
    """.strip()

    sigmoid_operation = """
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} 
    ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
        ^bb0(%in: f32, %out: f32):
        %cst_1 = arith.constant 1.000000e+00 : f32
        %neg = arith.negf %in : f32
        %exp = math.exp %neg : f32
        %denom = arith.addf %cst_1, %exp : f32
        %sigmoid = arith.divf %cst_1, %denom : f32
        linalg.yield %sigmoid : f32
    } -> tensor<SHAPExf32>
    """.strip().replace('SHAPE', SHAPE)

    return sigmoid_operation, sigmoid_maps

def softmax(*args,dim=3):
    if not args:
        SIZE = [str(choice(SIZES)) for _ in range(dim)] 
    else:
        SIZE = list(map(str,args[0]))
        dim = len(SIZE)

    # Define the tensor shape
    SHAPE = f"{'x'.join(SIZE)}xf32"

    Fill_SHAPE = f"{'x'.join(SIZE[:-1])}xf32" if dim!=1 else "f32" 

    maps = f"""
    #map1 = affine_map<({",".join([f"d{i}" for i in range(dim)])}) -> ({",".join([f"d{i}" for i in range(dim)])})>
    #map2 = affine_map<({",".join([f"d{i}" for i in range(dim)])}) -> ({", ".join([f"d{i}" for i in range(dim - 1)])})>
"""

    additional_function = (f"""
    func.func private @softmax(%input: tensor<{SHAPE}>, %output: tensor<{SHAPE}>) -> tensor<{SHAPE}> attributes {{ "func.inline" = unit }} {{
    %zero = arith.constant 0.00000e+00 : f32
    // Allocate temporary tensors for max and sum computations\n"""
    f"""%filled = bufferization.alloc_tensor() : tensor<{Fill_SHAPE}>\n""" 
    f"""
    // Inline compute_max functionality
    %max = linalg.reduce ins(%input: tensor<{SHAPE}>)
                            outs(%filled: tensor<{Fill_SHAPE}>) 
                            dimensions = [{dim - 1}]
                            (%in: f32, %acc: f32) {{
      %max = arith.maximumf %in, %acc : f32
      linalg.yield %max : f32
    }}
    // Inline compute_exp_sum functionality\n"""
    f"""%result = linalg.generic {{
      indexing_maps = [#map1, #map2, #map1],
      iterator_types = [{", ".join(['"parallel"' for _ in range(dim)])}],
      library_call = "none"
    }} ins(%input, %max : tensor<{SHAPE}>, tensor<{Fill_SHAPE}>)
      outs(%output : tensor<{SHAPE}>) {{
      ^bb0(%in: f32, %max_t: f32, %out: f32):
        %diff = arith.subf %in, %max_t : f32
        %exp = math.exp %diff : f32
        linalg.yield %exp : f32
    }} -> tensor<{SHAPE}>

    return %result : tensor<{SHAPE}>
}}
""")
    return (
        f"""func.call @softmax(%arg0, %dst) : (tensor<{SHAPE}>, tensor<{SHAPE}>) -> tensor<{SHAPE}>"""
        ,(maps,additional_function)
    )

def getShapes_Args(operation):
    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    if fields == []:
        # TODO: Add shape extraction so that allocation snippet could be replicated
        fields = re.findall("(?:\(([^(]+)\))(?:\s*\->\s*([^(]+))", operation)[0]
        
        args,shapes = [],[]
        for f in fields[0].split(", "):
            shapes.append(f)
        # shapes.append(fields[1])

        args = re.findall("(?:@\w+\(([^)]+))",operation)[0].split(',')

        args = [arg.strip() for arg in args]
        shapes = [shape.strip() for shape in shapes]

    else:
        args, shapes = [], []
        for field in fields:
            args_field, shapes_field = field.split(':')
            args   += args_field.split(',')
            shapes += shapes_field.split(',')

        args = [arg.strip() for arg in args]
        shapes = [shape.strip() for shape in shapes]

        args, shapes = _remove_duplicate_args(args, shapes)

    return args,shapes

# TODO: clean the code and refactor it
def randomSubGraph(verbose=False):
    
    params = []
    shapes = []
    return_vars = []
    return_shapes = []
    core = ""


    total_maps = ""
    total_additional_function = ""

    iterations = list(range(5))
    iterations_end = 5
    for _ in iterations:        
        
        operation_name = choice(list(LINALG_OPERATION_GENERATORS.keys())) # TODO: Restriction on operators

        if verbose:
            print(f"\033[91m{operation_name=}\033[0m")

        if return_shapes and return_shapes[-1]:
            shape = list(map(int,return_shapes[-1][len("tensor<"):-1].split('x')[:-1]))
            if verbose:
                print(f'\033[33m{shape}\033[0m')
            
            try:
                res = LINALG_OPERATION_GENERATORS[operation_name](shape)
            except:
                if verbose:
                    print(f"\033[33mskipped\033[0m")
                if iterations_end > 10:
                    break
                
                iterations.append(iterations_end+1)
                iterations_end += 1
                
                continue
        else:
            res = LINALG_OPERATION_GENERATORS[operation_name]()
        
        if verbose:
            print(f"\033[92m{res}\033[0m")

        maps = ""
        additional_function = ""

        if isinstance(res, tuple):
            raw_operation, additional_tuple = res
            if isinstance(additional_tuple, tuple):
                maps, additional_function = additional_tuple
                
            else:
                maps = additional_tuple

        else:
            raw_operation = res

        # Handling maps with the same name from different generators
        maps_identifiers = re.findall(r"#(\w+)[^\w]",maps)
        for map_id in maps_identifiers:
            new_map = f"map{''.join([choice(string.digits) for _ in range(5)])}"
            
            
            maps = re.sub(rf'\b{map_id}\b', new_map, maps)
            additional_function = re.sub(rf"\b{map_id}\b",new_map, additional_function)
            raw_operation = re.sub(rf"\b{map_id}\b",new_map,raw_operation)

        # Handling additional functions with the same name (same generator called twice or user negligence)
        functions_identifiers = re.findall(r"@(\w+)[^\w]", additional_function)
        for func_id in functions_identifiers:
            new_func = f"{func_id}{''.join([choice(string.digits) for _ in range(5)])}"
            
            additional_function = re.sub(rf"\b{func_id}\b",new_func, additional_function)
            raw_operation = re.sub(rf"\b{func_id}\b",new_func,raw_operation)    


        total_maps += "\n" + maps
        total_additional_function += "\n" + additional_function

        args,args_shape = getShapes_Args(raw_operation)

        # change the input shape
        if return_vars != []:
            old_shape = args_shape[0]

            # if any([x in raw_operation for x in ["matmul", "conv"]]):
                # raw_operation = raw_operation.replace(old_shape, return_shapes[-1], 1)
                # args_shape[0] = return_shapes[-1]

            if all([x not in raw_operation for x in ["generic", "func.call"]]) and \
                not any([x in raw_operation for x in ["matmul", "conv","pool"]]):
                if verbose:
                    print("\033[91m general shape change executed \033[0m")

                raw_operation = raw_operation.replace(old_shape, return_shapes[-1])
                args_shape = [return_shapes[-1] for _ in range(len(args_shape))]


        # dealing with arguments with the same name from different generators
        new_args = []
        for i,arg in enumerate(args):
            if i == 0 and return_vars != []:
                new_arg = return_vars[-1]
                args_shape.pop(0)

            else:
                new_arg = f"{arg}{''.join([choice(string.digits) for _ in range(5)])}"
                new_args.append(new_arg)

            raw_operation = raw_operation.replace(arg,new_arg)

        
        if params == []:
            params.extend(new_args)
            shapes.extend(args_shape)

        else:
            for arg,shape in zip(new_args,args_shape):
                if "tensor" in shape:
                    core += f"{arg} = bufferization.alloc_tensor() : {shape}\n"
                else:
                    core += f"{arg} = arith.constant 1.00000e+00 : f32\n"

        return_var = f"%var{''.join([choice(string.digits) for _ in range(5)])}" # TODO: prod-cons links
        return_vars.append(return_var)
        
        return_shape = args_shape[-1]
        
        if verbose:
            print(f"\033[90m {return_shape=}\033[0m")
        return_shapes.append(return_shape)

        core += f"""{return_var} = {raw_operation} \n"""

    core += f"""return {return_vars[-1]} : {return_shapes[-1]}\n"""

    total_additional_function += f"""\nfunc.func private @myFunction({", ".join([f"{p}:{s}" for p,s in zip(params,shapes)])}) -> {return_shapes[-1]} {{        
        {core}
    }}"""

    if verbose:
        print(f'\033[94m{total_additional_function=}\033[0m')

    final_operation = f"""func.call @myFunction({",".join(params)}) : ({",".join(shapes)}) -> {return_shapes[-1]}"""

    return final_operation,(total_maps,total_additional_function)

# TODO: refactor
def generate_resnet_block(
    input_tensor_name = "%arg60",
    block_id=0,
    bn_weight1_name="%arg0",
    bn_bias1_name="%arg1", 
    bn_mean1_name="%arg3",
    bn_weight2_name="%arg4"
):
    """
    Generate MLIR code for a parameterized ResNet block.
    
    Args:
        input_tensor_name: Name of the input tensor variable
        batch_size: Batch dimension
        in_channels: Input channels
        input_height: Input height
        input_width: Input width
        out_channels: Output channels (default 64)
        block_id: Unique identifier for this block's variables
        bn_weight1_name: First batch norm weight tensor name
        bn_bias1_name: First batch norm bias tensor name
        bn_mean1_name: First batch norm mean tensor name
        bn_weight2_name: Second batch norm weight tensor name
    
    Returns:
        str: MLIR code as f-string
    """

    batch_size = choice(BATCH_SIZES)
    in_channels = choice(CHANNELS) 
    input_height = choice(HEIGHTS)
    input_width = choice(HEIGHTS)

    out_channels = choice(CHANNELS)
    out_channels = 64 if out_channels == in_channels else out_channels
    # batch_size, in_channels, input_height, input_width = 256,256,150,56
    
    # Calculate output dimensions
    conv1_height = input_height // 2  # stride=2 in first conv
    conv1_width = input_width // 2
    pool_height = conv1_height // 2   # stride=2 in max pool
    pool_width = conv1_width // 2
    padded_height = input_height + 6  # padding=3 on each side
    padded_width = input_width + 6
    
    return (f"""func.call @Resnet({input_tensor_name}) : (tensor<{batch_size}x{in_channels}x{input_height}x{input_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>""", ('',f""" func.func private @Resnet({input_tensor_name} : tensor<{batch_size}x{in_channels}x{input_height}x{input_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32> {{
  
  // ResNet Block {block_id} - Input: {batch_size}x{in_channels}x{input_height}x{input_width}
  // Constants
  %cst_0_{block_id} = arith.constant 0.000000e+00 : f32
  %cst_1_{block_id} = arith.constant 1.000000e+00 : f32
  %cst_2_{block_id} = arith.constant 0xFF800000 : f32
  %cst_3_{block_id} = arith.constant 1.000000e-05 : f64

  // Tensor allocations for block {block_id}
  %conv1_tensor_{block_id} = bufferization.alloc_tensor() : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>
  %pool_tensor_{block_id} = bufferization.alloc_tensor() : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>
  %bn_temp_{block_id} = bufferization.alloc_tensor() : tensor<{out_channels}xf32>
  
  // Create and initialize kernels
  %kernel1_{block_id}_tmp = bufferization.alloc_tensor() : tensor<{out_channels}x{in_channels}x7x7xf32>
  %kernel1_{block_id} = linalg.fill ins(%cst_0_{block_id} : f32) outs(%kernel1_{block_id}_tmp : tensor<{out_channels}x{in_channels}x7x7xf32>) -> tensor<{out_channels}x{in_channels}x7x7xf32>
  
  %kernel2_{block_id}_tmp = bufferization.alloc_tensor() : tensor<{out_channels}x{out_channels}x3x3xf32>
  %kernel2_{block_id} = linalg.fill ins(%cst_0_{block_id} : f32) outs(%kernel2_{block_id}_tmp : tensor<{out_channels}x{out_channels}x3x3xf32>) -> tensor<{out_channels}x{out_channels}x3x3xf32>

  // Pad input tensor (3 pixels on each side)
  %padded_{block_id} = tensor.pad {input_tensor_name} low[0, 0, 3, 3] high[0, 0, 3, 3] {{
  ^bb0(%arg61: index, %arg62: index, %arg63: index, %arg64: index):
    tensor.yield %cst_0_{block_id} : f32
  }} : tensor<{batch_size}x{in_channels}x{input_height}x{input_width}xf32> to tensor<{batch_size}x{in_channels}x{padded_height}x{padded_width}xf32>

  
  %arg0_tmp = bufferization.alloc_tensor() : tensor<{out_channels}xf32>
  %arg0 = linalg.fill ins(%cst_0_{block_id} : f32) outs(%arg0_tmp : tensor<{out_channels}xf32>) -> tensor<{out_channels }xf32>

  %arg1_tmp = bufferization.alloc_tensor() : tensor<{out_channels}xf32>
  %arg1 = linalg.fill ins(%cst_0_{block_id} : f32) outs(%arg1_tmp : tensor<{out_channels}xf32>) -> tensor<{out_channels }xf32>

  %arg3_tmp = bufferization.alloc_tensor() : tensor<{out_channels}xf32>
  %arg3 = linalg.fill ins(%cst_0_{block_id} : f32) outs(%arg3_tmp : tensor<{out_channels}xf32>) -> tensor<{out_channels }xf32>

  %arg4_tmp = bufferization.alloc_tensor() : tensor<{out_channels}xf32>
  %arg4 = linalg.fill ins(%cst_0_{block_id} : f32) outs(%arg4_tmp : tensor<{out_channels}xf32>) -> tensor<{out_channels }xf32>
  
  // Expand batch norm parameters
  %expanded_{block_id} = tensor.expand_shape {bn_weight1_name} [[0, 1, 2]] output_shape [{out_channels}, 1, 1] : tensor<{out_channels}xf32> into tensor<{out_channels}x1x1xf32>
  %expanded_mean_{block_id} = tensor.expand_shape {bn_mean1_name} [[0, 1, 2]] output_shape [{out_channels}, 1, 1] : tensor<{out_channels}xf32> into tensor<{out_channels}x1x1xf32>

  // Initialize output tensor
  %conv1_init_{block_id} = linalg.fill ins(%cst_0_{block_id} : f32) outs(%conv1_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>) -> tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>

  // First convolution (7x7, stride=2)
  %conv1_{block_id} = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}} ins(%padded_{block_id}, %kernel1_{block_id} : tensor<{batch_size}x{in_channels}x{padded_height}x{padded_width}xf32>, tensor<{out_channels}x{in_channels}x7x7xf32>) outs(%conv1_init_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>) -> tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>

  // First batch normalization
  %bn1_1_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins({bn_bias1_name} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %eps = arith.truncf %cst_3_{block_id} : f64 to f32
    %result = arith.addf %in, %eps : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %bn1_2_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins(%bn1_1_{block_id} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %result = math.sqrt %in : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %bn1_3_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins(%bn1_2_{block_id} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %check = arith.cmpf one, %in, %cst_0_{block_id} : f32
    cf.assert %check, "unimplemented: tensor with zero element"
    %result = arith.divf %cst_1_{block_id}, %in : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %expanded_scale_{block_id} = tensor.expand_shape %bn1_3_{block_id} [[0, 1, 2]] output_shape [{out_channels}, 1, 1] : tensor<{out_channels}xf32> into tensor<{out_channels}x1x1xf32>

  // Apply batch normalization to conv output
  %bn_applied1_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%conv1_{block_id}, %expanded_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>, tensor<{out_channels}x1x1xf32>) outs(%conv1_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>)  {{
  ^bb0(%in: f32, %in_161: f32, %out: f32):
    %result = arith.subf %in, %in_161 : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>

  %bn_applied2_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%bn_applied1_{block_id}, %expanded_scale_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>, tensor<{out_channels}x1x1xf32>) outs(%conv1_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>)  {{
  ^bb0(%in: f32, %in_161: f32, %out: f32):
    %result = arith.mulf %in, %in_161 : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>

  // ReLU activation
  %relu1_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%bn_applied2_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>) outs(%conv1_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %is_positive = arith.cmpf ugt, %in, %cst_0_{block_id} : f32
    %result = arith.select %is_positive, %in, %cst_0_{block_id} : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32>

  // Pad for max pooling
  %padded_pool_{block_id} = tensor.pad %relu1_{block_id} low[0, 0, 1, 1] high[0, 0, 1, 1] {{
  ^bb0(%arg61: index, %arg62: index, %arg63: index, %arg64: index):
    tensor.yield %cst_2_{block_id} : f32
  }} : tensor<{batch_size}x{out_channels}x{conv1_height}x{conv1_width}xf32> to tensor<{batch_size}x{out_channels}x{conv1_height + 2}x{conv1_width + 2}xf32>

  // Max pooling
  %pool_init_{block_id} = linalg.fill ins(%cst_2_{block_id} : f32) outs(%pool_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>
  %pool_kernel_{block_id} = bufferization.alloc_tensor() : tensor<3x3xf32>
  
  %maxpool_{block_id} = linalg.pooling_nchw_max {{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}} ins(%padded_pool_{block_id}, %pool_kernel_{block_id} : tensor<{batch_size}x{out_channels}x{conv1_height + 2}x{conv1_width + 2}xf32>, tensor<3x3xf32>) outs(%pool_init_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  // Pad for second conv
  %padded_conv2_{block_id} = tensor.pad %maxpool_{block_id} low[0, 0, 1, 1] high[0, 0, 1, 1] {{
  ^bb0(%arg61: index, %arg62: index, %arg63: index, %arg64: index):
    tensor.yield %cst_0_{block_id} : f32
  }} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32> to tensor<{batch_size}x{out_channels}x{pool_height + 2}x{pool_width + 2}xf32>

  %conv2_init_{block_id} = linalg.fill ins(%cst_0_{block_id} : f32) outs(%pool_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  %conv2_{block_id} = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}} ins(%padded_conv2_{block_id}, %kernel2_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height + 2}x{pool_width + 2}xf32>, tensor<{out_channels}x{out_channels}x3x3xf32>) outs(%conv2_init_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>) -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  // Second batch normalization
  %bn2_1_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins({bn_weight2_name} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %eps = arith.truncf %cst_3_{block_id} : f64 to f32
    %result = arith.addf %in, %eps : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %bn2_2_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins(%bn2_1_{block_id} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %result = math.sqrt %in : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %bn2_3_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins(%bn2_2_{block_id} : tensor<{out_channels}xf32>) outs(%bn_temp_{block_id} : tensor<{out_channels}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %check = arith.cmpf one, %in, %cst_0_{block_id} : f32
    cf.assert %check, "unimplemented: tensor with zero element"
    %result = arith.divf %cst_1_{block_id}, %in : f32
    linalg.yield %result : f32
  }} -> tensor<{out_channels}xf32>

  %expanded_scale2_{block_id} = tensor.expand_shape %bn2_3_{block_id} [[0, 1, 2]] output_shape [{out_channels}, 1, 1] : tensor<{out_channels}xf32> into tensor<{out_channels}x1x1xf32>

  // Apply second batch normalization
  %final_bn1_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%conv2_{block_id}, %expanded_mean_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>, tensor<{out_channels}x1x1xf32>) outs(%pool_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>)  {{
  ^bb0(%in: f32, %in_161: f32, %out: f32):
    %result = arith.subf %in, %in_161 : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  %final_bn2_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%final_bn1_{block_id}, %expanded_scale2_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>, tensor<{out_channels}x1x1xf32>) outs(%pool_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>)  {{
  ^bb0(%in: f32, %in_161: f32, %out: f32):
    %result = arith.mulf %in, %in_161 : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  // Final ReLU - Output tensor: %final_relu_{block_id}
  %final_relu_{block_id} = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%final_bn2_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>) outs(%pool_tensor_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>)  {{
  ^bb0(%in: f32, %out: f32):
    %is_positive = arith.cmpf ugt, %in, %cst_0_{block_id} : f32
    %result = arith.select %is_positive, %in, %cst_0_{block_id} : f32
    linalg.yield %result : f32
  }} -> tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  return %final_relu_{block_id} : tensor<{batch_size}x{out_channels}x{pool_height}x{pool_width}xf32>

  // End of ResNet Block {block_id} - Output: {batch_size}x{out_channels}x{pool_height}x{pool_width}
}}"""))

def generate_residual_block_mlir(block_name="residual_block"):
    """
    Generate MLIR code for a residual block that takes input of shape tensor<{N}x{K}xf32>
    
    Args:
        N (int): First dimension size
        K (int): Second dimension size  
        block_name (str): Name of the function block
    
    Returns:
        str: MLIR code for the residual block where F(input) = ReLU(input @ W)
    """

    N,K = choice(SIZES),choice(SIZES)

    
    mlir_code = f'''func.func private @{block_name}(%input: tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32> {{
  // F(input) = ReLU(input @ W)
  %weights = arith.constant dense<1.0> : tensor<{K}x{K}xf32>
  %zero_tensor = arith.constant dense<0.0> : tensor<{N}x{K}xf32>
  
  // Linear transformation: input @ weights
  %matmul = linalg.matmul ins(%input, %weights : tensor<{N}x{K}xf32>, tensor<{K}x{K}xf32>) 
                          outs(%zero_tensor : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>
  
  // ReLU activation on the matmul result
  %zero = arith.constant 0.0 : f32
  %relu = linalg.generic {{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }} ins(%matmul : tensor<{N}x{K}xf32>) outs(%zero_tensor : tensor<{N}x{K}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    %90 = arith.cmpf ugt, %in, %zero: f32
    %max = arith.select %90, %in, %zero : f32
    linalg.yield %max : f32
  }} -> tensor<{N}x{K}xf32>
  
  // Residual connection: output = input + F(input) = input + ReLU(input @ W)
  %residual_output = linalg.generic {{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }} ins(%input, %relu : tensor<{N}x{K}xf32>, tensor<{N}x{K}xf32>) outs(%zero_tensor : tensor<{N}x{K}xf32>) {{
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  }} -> tensor<{N}x{K}xf32>
  
  return %residual_output : tensor<{N}x{K}xf32>}}'''

    return (f"func.call @{block_name}(%input) : (tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>",('',mlir_code))

def vgg():
    # Input shape
    N = choice(BATCH_SIZES)
    C = 192
    H = W = choice(HEIGHTS)  # Assuming square input

    # Constants from the code
    padding = 1
    stride = 1
    dilation = 1

    # First convolution parameters
    F1 = 64  # Output channels for first conv
    K1 = 3   # Kernel size for first conv

    # Shape after first padding
    H_padded1 = H + 2 * padding
    W_padded1 = W + 2 * padding

    # Shape after first convolution (same padding)
    H_conv1 = H
    W_conv1 = W

    # Second convolution parameters  
    F2 = 64  # Output channels for second conv
    K2 = 3   # Kernel size for second conv

    # Shape after second padding
    H_padded2 = H_conv1 + 2 * padding
    W_padded2 = W_conv1 + 2 * padding

    # Shape after second convolution (same padding)
    H_conv2 = H_conv1
    W_conv2 = W_conv1

    # Max pooling parameters
    pool_kernel = 2
    pool_stride = 2

    # Shape after max pooling
    H_pool = (H_conv2 - pool_kernel) // pool_stride + 1
    W_pool = (W_conv2 - pool_kernel) // pool_stride + 1

    # Third convolution parameters
    F3 = 128  # Output channels for third conv
    K3 = 3    # Kernel size for third conv

    # Shape after third padding
    H_padded3 = H_pool + 2 * padding
    W_padded3 = W_pool + 2 * padding

    # Shape after third convolution (same padding)
    H_conv3 = H_pool
    W_conv3 = W_pool
    
    mlir_code = f"""func.func @vgg(%arg0: tensor<{N}x{C}x{H}x{W}xf32>) -> tensor<{N}x{F3}x{H_pool}x{W_pool}xf32> {{
%cst_0 = arith.constant 0.0 : f32
%cst2 = arith.constant 2.0 : f32

%cst_tensor = tensor.empty(): tensor<64xf32>
%cst = linalg.fill ins(%cst2: f32) outs(%cst_tensor: tensor<64xf32>) -> tensor<64xf32>

%cst_2_tensor = tensor.empty(): tensor<64x192x3x3xf32>
%cst_2 = linalg.fill ins(%cst2: f32) outs(%cst_2_tensor: tensor<64x192x3x3xf32>) -> tensor<64x192x3x3xf32>

%cst_4_tensor = tensor.empty(): tensor<64xf32>
%cst_4 = linalg.fill ins(%cst2: f32) outs(%cst_4_tensor: tensor<64xf32>) -> tensor<64xf32>

%cst_3_tensor = tensor.empty(): tensor<64x64x3x3xf32>
%cst_3 = linalg.fill ins(%cst2: f32) outs(%cst_3_tensor: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>

%cst_1 = arith.constant 1.0 : f32

%cst_6_tensor = tensor.empty(): tensor<128xf32>
%cst_6 = linalg.fill ins(%cst2: f32) outs(%cst_6_tensor: tensor<128xf32>) -> tensor<128xf32>

%cst_5_tensor = tensor.empty(): tensor<128x64x3x3xf32>
%cst_5 = linalg.fill ins(%cst2: f32) outs(%cst_5_tensor: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>

%padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {{
^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
  tensor.yield %cst_0 : f32
}} : tensor<{N}x{C}x{H}x{W}xf32> to tensor<{N}x{C}x{H+2}x{W+2}xf32>

%0 = tensor.empty() : tensor<{N}x{F1}x{H}x{W}xf32>
%broadcasted = linalg.broadcast ins(%cst : tensor<{F1}xf32>) outs(%0 : tensor<{N}x{F1}x{H}x{W}xf32>) dimensions = [0, 2, 3]

%1 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}} 
     ins(%padded, %cst_2 : tensor<{N}x{C}x{H+2}x{W+2}xf32>, tensor<{F1}x{C}x{K1}x{K1}xf32>) 
     outs(%broadcasted : tensor<{N}x{F1}x{H}x{W}xf32>) -> tensor<{N}x{F1}x{H}x{W}xf32>

%2 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%1 : tensor<{N}x{F1}x{H}x{W}xf32>) outs(%0 : tensor<{N}x{F1}x{H}x{W}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    %65 = arith.cmpf ugt, %in, %cst_0 : f32
    %66 = arith.select %65, %in, %cst_0 : f32
    linalg.yield %66 : f32
}} -> tensor<{N}x{F1}x{H}x{W}xf32>

%padded_33 = tensor.pad %2 low[0, 0, 1, 1] high[0, 0, 1, 1] {{
^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
  tensor.yield %cst_0 : f32
}} : tensor<{N}x{F1}x{H}x{W}xf32> to tensor<{N}x{F1}x{H+2}x{W+2}xf32>

%broadcasted_34 = linalg.broadcast ins(%cst_4 : tensor<{F2}xf32>) outs(%0 : tensor<{N}x{F1}x{H}x{W}xf32>) dimensions = [0, 2, 3]

%3 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}} 
     ins(%padded_33, %cst_3 : tensor<{N}x{F1}x{H+2}x{W+2}xf32>, tensor<{F2}x{F1}x{K2}x{K2}xf32>) 
     outs(%broadcasted_34 : tensor<{N}x{F1}x{H}x{W}xf32>) -> tensor<{N}x{F1}x{H}x{W}xf32>

%4 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%3 : tensor<{N}x{F1}x{H}x{W}xf32>) outs(%0 : tensor<{N}x{F1}x{H}x{W}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    %65 = arith.cmpf ugt, %in, %cst_0 : f32
    %66 = arith.select %65, %in, %cst_0 : f32
    linalg.yield %66 : f32
}} -> tensor<{N}x{F1}x{H}x{W}xf32>

%5 = tensor.empty() : tensor<{N}x{F2}x{H_pool}x{W_pool}xf32>
%6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<{N}x{F2}x{H_pool}x{W_pool}xf32>) -> tensor<{N}x{F2}x{H_pool}x{W_pool}xf32>
%7 = tensor.empty() : tensor<2x2xf32>

%8 = linalg.pooling_nchw_max {{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}} 
     ins(%4, %7 : tensor<{N}x{F1}x{H}x{W}xf32>, tensor<2x2xf32>) 
     outs(%6 : tensor<{N}x{F2}x{H_pool}x{W_pool}xf32>) -> tensor<{N}x{F2}x{H_pool}x{W_pool}xf32>

%padded_35 = tensor.pad %8 low[0, 0, 1, 1] high[0, 0, 1, 1] {{
^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
  tensor.yield %cst_0 : f32
}} : tensor<{N}x{F2}x{H_pool}x{W_pool}xf32> to tensor<{N}x{F2}x{H_pool+2}x{W_pool+2}xf32>

%9 = tensor.empty() : tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>
%broadcasted_36 = linalg.broadcast ins(%cst_6 : tensor<{F3}xf32>) outs(%9 : tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>) dimensions = [0, 2, 3]

%10 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}}
      ins(%padded_35, %cst_5 : tensor<{N}x{F2}x{H_pool+2}x{W_pool+2}xf32>, tensor<{F3}x{F2}x{K3}x{K3}xf32>) 
      outs(%broadcasted_36 : tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>) -> tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>

%11 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
      ins(%10 : tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>) 
      outs(%9 : tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>) {{
  ^bb0(%in: f32, %out: f32):
    %65 = arith.cmpf ugt, %in, %cst_0 : f32
    %66 = arith.select %65, %in, %cst_0 : f32
    linalg.yield %66 : f32
}} -> tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>
return %11: tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>
}}"""

    return f"func.call @vgg(%arg0) : (tensor<{N}x{C}x{H}x{W}xf32>) -> tensor<{N}x{F3}x{H_pool}x{W_pool}xf32>", ("""#map = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>""", mlir_code)


def bert():
    batch_size = 1
    seq_length = choice(HEIGHTS) #Max(min(choice(BATCH_SIZES), 50),16)
    head_size = choice(SIZES)
    num_attention_heads = choice(KERNELS)

    """
    Generate BERT encoding MLIR code with generalized shapes
    
    Args:
        batch_size: Batch size (N)
        seq_length: Sequence length (S)
        hidden_size: Hidden size (H)
        num_attention_heads: Number of attention heads (A)
    """
    
    # Derived dimensions
    hidden_size = head_size * num_attention_heads  # Size per attention head
    
    # Vocabulary and positional encoding sizes (from the original code)
    vocab_size = 30522
    pos_vocab_size = 512
    
    mlir_code = f""" func.func @bert(%arg0: tensor<{batch_size}x{seq_length}xi64>) -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32> {{
%cst2 = arith.constant 2.0 : f32
%cst_0 = arith.constant 0.000000e+00 : f32
%cst_1 = arith.constant 0xFF800000 : f32
%cst_2 = arith.constant 7.680000e+02 : f32
%cst_3 = arith.constant 9.99999996E-13 : f32
%c0_i64 = arith.constant 0 : i64

%cst_103 = arith.constant dense<1.000000e+00> : tensor<f32>
%cst_104 = arith.constant dense<-3.40282347E+38> : tensor<f32>
%cst_105 = arith.constant dense<1.000000e+00> : tensor<1xf32>
%cst_106 = arith.constant dense<1.41421354> : tensor<f32>
%cst_107 = arith.constant dense<5.000000e-01> : tensor<f32>
%c{vocab_size}_i64 = arith.constant {vocab_size} : i64
%cst_108 = arith.constant dense<{list(range(1,seq_length+1))}> : tensor<{seq_length}xi64>
%cst_109 = arith.constant dense<64> : tensor<1xi64>

%cst_102_tensor = tensor.empty() : tensor<{vocab_size}x{hidden_size}xf32>
%cst_102 = linalg.fill ins(%cst2: f32) outs(%cst_102_tensor: tensor<{vocab_size}x{hidden_size}xf32>) -> tensor<{vocab_size}x{hidden_size}xf32>

%cst_101_tensor = tensor.empty() : tensor<{pos_vocab_size}x{hidden_size}xf32>
%cst_101 = linalg.fill ins(%cst2: f32) outs(%cst_101_tensor: tensor<{pos_vocab_size}x{hidden_size}xf32>) -> tensor<{pos_vocab_size}x{hidden_size}xf32>

%cst_100_tensor = tensor.empty() : tensor<{hidden_size}xf32>
%cst_100 = linalg.fill ins(%cst2: f32) outs(%cst_100_tensor: tensor<{hidden_size}xf32>) -> tensor<{hidden_size}xf32>

%cst_99_tensor = tensor.empty() : tensor<{hidden_size}xf32>
%cst_99 = linalg.fill ins(%cst2: f32) outs(%cst_99_tensor: tensor<{hidden_size}xf32>) -> tensor<{hidden_size}xf32>

%arg1_tensor = tensor.empty() : tensor<{batch_size}x{seq_length}xi64>
%arg1 = linalg.fill ins(%cst2: f32) outs(%arg1_tensor: tensor<{batch_size}x{seq_length}xi64>) -> tensor<{batch_size}x{seq_length}xi64>

%cst_38_tensor = tensor.empty() : tensor<{hidden_size}x{hidden_size}xf32>
%cst_38 = linalg.fill ins(%cst2: f32) outs(%cst_38_tensor: tensor<{hidden_size}x{hidden_size}xf32>) -> tensor<{hidden_size}x{hidden_size}xf32>

%cst_98_tensor = tensor.empty() : tensor<{hidden_size}xf32>
%cst_98 = linalg.fill ins(%cst2: f32) outs(%cst_98_tensor: tensor<{hidden_size}xf32>) -> tensor<{hidden_size}xf32>

%cst_37_tensor = tensor.empty() : tensor<{hidden_size}x{hidden_size}xf32>
%cst_37 = linalg.fill ins(%cst2: f32) outs(%cst_37_tensor: tensor<{hidden_size}x{hidden_size}xf32>) -> tensor<{hidden_size}x{hidden_size}xf32>

%cst_97_tensor = tensor.empty() : tensor<{hidden_size}xf32>
%cst_97 = linalg.fill ins(%cst2: f32) outs(%cst_97_tensor: tensor<{hidden_size}xf32>) -> tensor<{hidden_size}xf32>

%cst_36_tensor = tensor.empty() : tensor<{hidden_size}x{hidden_size}xf32>
%cst_36 = linalg.fill ins(%cst2: f32) outs(%cst_36_tensor: tensor<{hidden_size}x{hidden_size}xf32>) -> tensor<{hidden_size}x{hidden_size}xf32>

%cst_96_tensor = tensor.empty() : tensor<{hidden_size}xf32>
%cst_96 = linalg.fill ins(%cst2: f32) outs(%cst_96_tensor: tensor<{hidden_size}xf32>) -> tensor<{hidden_size}xf32>

%0 = tensor.empty() : tensor<{batch_size}x{seq_length}xi1>
%1 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}} ins(%arg0 : tensor<{batch_size}x{seq_length}xi64>) outs(%0 : tensor<{batch_size}x{seq_length}xi1>) {{
^bb0(%in: i64, %out: i1):
  %472 = arith.cmpi slt, %in, %c0_i64 : i64
  linalg.yield %472 : i1
}} -> tensor<{batch_size}x{seq_length}xi1>

%2 = tensor.empty() : tensor<{batch_size}x{seq_length}xi64>
%3 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}} ins(%arg0 : tensor<{batch_size}x{seq_length}xi64>) outs(%2 : tensor<{batch_size}x{seq_length}xi64>) {{
^bb0(%in: i64, %out: i64):
  %472 = arith.addi %in, %c{vocab_size}_i64 : i64
  linalg.yield %472 : i64
}} -> tensor<{batch_size}x{seq_length}xi64>

%4 = linalg.generic {{indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "parallel"]}} ins(%1, %3, %arg0 : tensor<{batch_size}x{seq_length}xi1>, tensor<{batch_size}x{seq_length}xi64>, tensor<{batch_size}x{seq_length}xi64>) outs(%2 : tensor<{batch_size}x{seq_length}xi64>) {{
^bb0(%in: i1, %in_201: i64, %in_202: i64, %out: i64):
  %472 = arith.select %in, %in_201, %in_202 : i64
  linalg.yield %472 : i64
}} -> tensor<{batch_size}x{seq_length}xi64>

%collapsed = tensor.collapse_shape %4 [[0, 1]] : tensor<{batch_size}x{seq_length}xi64> into tensor<{batch_size * seq_length}xi64>

%5 = tensor.empty() : tensor<{batch_size * seq_length}x{hidden_size}xf32>
%6 = linalg.generic {{indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}} ins(%collapsed : tensor<{batch_size * seq_length}xi64>) outs(%5 : tensor<{batch_size * seq_length}x{hidden_size}xf32>) {{
^bb0(%in: i64, %out: f32):
  %472 = arith.index_cast %in : i64 to index
  %473 = linalg.index 1 : index
  %extracted = tensor.extract %cst_102[%472, %473] : tensor<{vocab_size}x{hidden_size}xf32>
  linalg.yield %extracted : f32
}} -> tensor<{batch_size * seq_length}x{hidden_size}xf32>

%expanded = tensor.expand_shape %6 [[0, 1], [2]] output_shape [{batch_size}, {seq_length}, {hidden_size}] : tensor<{batch_size * seq_length}x{hidden_size}xf32> into tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%c = tensor.empty() : tensor<{seq_length}x{hidden_size}xf32>
%7 = linalg.generic {{indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}} ins(%cst_108 : tensor<{seq_length}xi64>) outs(%c : tensor<{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: i64, %out: f32):
  %472 = arith.index_cast %in : i64 to index
  %473 = linalg.index 1 : index
  %extracted = tensor.extract %cst_101[%472, %473] : tensor<{pos_vocab_size}x{hidden_size}xf32>
  linalg.yield %extracted : f32
}} -> tensor<{seq_length}x{hidden_size}xf32>

%expanded_110 = tensor.expand_shape %7 [[0, 1], [2]] output_shape [{batch_size}, {seq_length}, {hidden_size}] : tensor<{batch_size * seq_length}x{hidden_size}xf32> into tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%8 = tensor.empty() : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>
%9 = linalg.generic {{indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%expanded, %expanded_110 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%10 = tensor.empty() : tensor<{batch_size}x{seq_length}x1xf32>
%11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<{batch_size}x{seq_length}x1xf32>) -> tensor<{batch_size}x{seq_length}x1xf32>

%12 = linalg.generic {{indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%9 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%11 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.addf %in, %out : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%13 = linalg.generic {{indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%12 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%10 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.divf %in, %cst_2 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%14 = linalg.generic {{indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%13 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%15 = linalg.generic {{indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%9, %14 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.subf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%16 = linalg.generic {{indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%15, %15 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.mulf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%17 = linalg.generic {{indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%16 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%11 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.addf %in, %out : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%18 = linalg.generic {{indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%17 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%10 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.divf %in, %cst_2 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%19 = linalg.generic {{indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%18 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%10 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.addf %in, %cst_3 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%20 = linalg.generic {{indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%19 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%10 : tensor<{batch_size}x{seq_length}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = math.rsqrt %in : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x1xf32>

%21 = linalg.generic {{indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%20 : tensor<{batch_size}x{seq_length}x1xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%22 = linalg.generic {{indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%15, %21 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.mulf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%23 = linalg.generic {{indexing_maps = [#map3, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%22, %cst_100 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.mulf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%24 = linalg.generic {{indexing_maps = [#map3, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%23, %cst_99 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%expanded_111 = tensor.expand_shape %arg1 [[0], [1, 2, 3]] output_shape [{batch_size}, 1, 1, {seq_length}] : tensor<{batch_size}x{seq_length}xi64> into tensor<{batch_size}x1x1x{seq_length}xi64>

%25 = tensor.empty() : tensor<{batch_size}x1x{seq_length}x{seq_length}xi64>
%26 = linalg.generic {{indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%expanded_111 : tensor<{batch_size}x1x1x{seq_length}xi64>) outs(%25 : tensor<{batch_size}x1x{seq_length}x{seq_length}xi64>) {{
^bb0(%in: i64, %out: i64):
  linalg.yield %in : i64
}} -> tensor<{batch_size}x1x{seq_length}x{seq_length}xi64>

%27 = tensor.empty() : tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>
%28 = linalg.generic {{indexing_maps = [#map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%26 : tensor<{batch_size}x1x{seq_length}x{seq_length}xi64>) outs(%27 : tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: i64, %out: f32):
  %472 = arith.sitofp %in : i64 to f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>

%29 = linalg.generic {{indexing_maps = [#map11, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%cst_103, %28 : tensor<f32>, tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) outs(%27 : tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.subf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>

%30 = tensor.empty() : tensor<{batch_size}x1x{seq_length}x{seq_length}xi1>
%31 = linalg.generic {{indexing_maps = [#map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%29 : tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) outs(%30 : tensor<{batch_size}x1x{seq_length}x{seq_length}xi1>) {{
^bb0(%in: f32, %out: i1):
  %472 = arith.cmpf une, %in, %cst_0 : f32
  linalg.yield %472 : i1
}} -> tensor<{batch_size}x1x{seq_length}x{seq_length}xi1>

%32 = linalg.generic {{indexing_maps = [#map10, #map11, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%31, %cst_104, %29 : tensor<{batch_size}x1x{seq_length}x{seq_length}xi1>, tensor<f32>, tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) outs(%27 : tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: i1, %in_201: f32, %in_202: f32, %out: f32):
  %472 = arith.select %in, %in_201, %in_202 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>

%33 = linalg.generic {{indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%24 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%34 = tensor.empty() : tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>
%35 = linalg.generic {{indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_38 : tensor<{hidden_size}x{hidden_size}xf32>) outs(%34 : tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>

%36 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%37 = linalg.batch_matmul ins(%33, %35 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) outs(%36 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%38 = linalg.generic {{indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_98, %37 : tensor<{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%expanded_112 = tensor.expand_shape %38 [[0], [1], [2, 3]] output_shape [{batch_size}, {seq_length}, {num_attention_heads}, {head_size}] : tensor<{batch_size}x{seq_length}x{hidden_size}xf32> into tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>

%39 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>
%transposed = linalg.transpose ins(%expanded_112 : tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>) outs(%39 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>) permutation = [0, 2, 1, 3]

%40 = linalg.generic {{indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_37 : tensor<{hidden_size}x{hidden_size}xf32>) outs(%34 : tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>

%41 = linalg.batch_matmul ins(%33, %40 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) outs(%36 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%42 = linalg.generic {{indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_97, %41 : tensor<{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%expanded_113 = tensor.expand_shape %42 [[0], [1], [2, 3]] output_shape [{batch_size}, {seq_length}, {num_attention_heads}, {head_size}] : tensor<{batch_size}x{seq_length}x{hidden_size}xf32> into tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>

%43 = linalg.generic {{indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_36 : tensor<{hidden_size}x{hidden_size}xf32>) outs(%34 : tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>

%44 = linalg.batch_matmul ins(%33, %43 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>, tensor<{batch_size}x{hidden_size}x{hidden_size}xf32>) outs(%36 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%45 = linalg.generic {{indexing_maps = [#map7, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%cst_96, %44 : tensor<{hidden_size}xf32>, tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) outs(%8 : tensor<{batch_size}x{seq_length}x{hidden_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{seq_length}x{hidden_size}xf32>

%expanded_114 = tensor.expand_shape %45 [[0], [1], [2, 3]] output_shape [{batch_size}, {seq_length}, {num_attention_heads}, {head_size}] : tensor<{batch_size}x{seq_length}x{hidden_size}xf32> into tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>

%transposed_115 = linalg.transpose ins(%expanded_114 : tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>) outs(%39 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>) permutation = [0, 2, 1, 3]

%46 = tensor.empty() : tensor<1xf32>
%47 = linalg.generic {{indexing_maps = [#map13, #map14], iterator_types = ["parallel"]}} ins(%cst_109 : tensor<1xi64>) outs(%46 : tensor<1xf32>) {{
^bb0(%in: i64, %out: f32):
  %472 = arith.sitofp %in : i64 to f32
  linalg.yield %472 : f32
}} -> tensor<1xf32>

%48 = linalg.generic {{indexing_maps = [#map13, #map14], iterator_types = ["parallel"]}} ins(%47 : tensor<1xf32>) outs(%46 : tensor<1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = math.sqrt %in : f32
  linalg.yield %472 : f32
}} -> tensor<1xf32>

%49 = linalg.generic {{indexing_maps = [#map13, #map13, #map14], iterator_types = ["parallel"]}} ins(%cst_105, %48 : tensor<1xf32>, tensor<1xf32>) outs(%46 : tensor<1xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.divf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<1xf32>

%50 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>
%transposed_116 = linalg.transpose ins(%expanded_113 : tensor<{batch_size}x{seq_length}x{num_attention_heads}x{head_size}xf32>) outs(%50 : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>) permutation = [0, 2, 3, 1]

%51 = linalg.generic {{indexing_maps = [#map13, #map14], iterator_types = ["parallel"]}} ins(%49 : tensor<1xf32>) outs(%46 : tensor<1xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = math.sqrt %in : f32
  linalg.yield %472 : f32
}} -> tensor<1xf32>

%52 = linalg.generic {{indexing_maps = [#map15, #map16, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%transposed, %51 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>, tensor<1xf32>) outs(%39 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.mulf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>

%53 = linalg.generic {{indexing_maps = [#map15, #map16, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%transposed_116, %51 : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>, tensor<1xf32>) outs(%50 : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.mulf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>

%54 = linalg.generic {{indexing_maps = [#map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%52 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>) outs(%39 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32>

%55 = linalg.generic {{indexing_maps = [#map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%53 : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>) outs(%50 : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32>

%collapsed_117 = tensor.collapse_shape %54 [[0, 1], [2], [3]] : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{head_size}xf32> into tensor<{batch_size * num_attention_heads}x{seq_length}x{head_size}xf32>
%collapsed_118 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<{batch_size}x{num_attention_heads}x{head_size}x{seq_length}xf32> into tensor<{batch_size * num_attention_heads}x{head_size}x{seq_length}xf32>

%56 = tensor.empty() : tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32>
%57 = linalg.fill ins(%cst_0 : f32) outs(%56 : tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32>) -> tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32>

%58 = linalg.batch_matmul ins(%collapsed_117, %collapsed_118 : tensor<{batch_size * num_attention_heads}x{seq_length}x{head_size}xf32>, tensor<{batch_size * num_attention_heads}x{head_size}x{seq_length}xf32>) outs(%57 : tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32>) -> tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32>

%expanded_119 = tensor.expand_shape %58 [[0, 1], [2], [3]] output_shape [{batch_size}, {num_attention_heads}, {seq_length}, {seq_length}] : tensor<{batch_size * num_attention_heads}x{seq_length}x{seq_length}xf32> into tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

%59 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>
%60 = linalg.generic {{indexing_maps = [#map15, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%expanded_119, %32 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>, tensor<{batch_size}x1x{seq_length}x{seq_length}xf32>) outs(%59 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.addf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

%61 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{seq_length}xi64>
%62 = linalg.fill ins(%c0_i64 : i64) outs(%61 : tensor<{batch_size}x{num_attention_heads}x{seq_length}xi64>) -> tensor<{batch_size}x{num_attention_heads}x{seq_length}xi64>

%63 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>
%64 = linalg.fill ins(%cst_1 : f32) outs(%63 : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>) -> tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>

%65:2 = linalg.generic {{indexing_maps = [#map9, #map17, #map17], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}} ins(%60 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) outs(%64, %62 : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>, tensor<{batch_size}x{num_attention_heads}x{seq_length}xi64>) {{
^bb0(%in: f32, %out: f32, %out_201: i64):
  %472 = linalg.index 3 : index
  %473 = arith.index_cast %472 : index to i64
  %474 = arith.maximumf %in, %out : f32
  %475 = arith.cmpf ogt, %in, %out : f32
  %476 = arith.select %475, %473, %out_201 : i64
  linalg.yield %474, %476 : f32, i64
}} -> (tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>, tensor<{batch_size}x{num_attention_heads}x{seq_length}xi64>)

%expanded_120 = tensor.expand_shape %65#0 [[0], [1], [2, 3]] output_shape [{batch_size}, {num_attention_heads}, {seq_length}, 1] : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32> into tensor<{batch_size}x{num_attention_heads}x{seq_length}x1xf32>

%66 = linalg.generic {{indexing_maps = [#map15, #map18, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%60, %expanded_120 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>, tensor<{batch_size}x{num_attention_heads}x{seq_length}x1xf32>) outs(%59 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.subf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

%67 = linalg.generic {{indexing_maps = [#map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%66 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) outs(%59 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = math.exp %in : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

%68 = tensor.empty() : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>
%69 = linalg.fill ins(%cst_0 : f32) outs(%68 : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>) -> tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>

%70 = linalg.generic {{indexing_maps = [#map9, #map19], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}} ins(%67 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) outs(%69 : tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>) {{
^bb0(%in: f32, %out: f32):
  %472 = arith.addf %in, %out : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>

%71 = linalg.generic {{indexing_maps = [#map15, #map17, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%67, %70 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>, tensor<{batch_size}x{num_attention_heads}x{seq_length}xf32>) outs(%59 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %in_201: f32, %out: f32):
  %472 = arith.divf %in, %in_201 : f32
  linalg.yield %472 : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

%72 = linalg.generic {{indexing_maps = [#map15, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} ins(%71 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) outs(%59 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>

return %72 : tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>
}}
"""

    return f"func.call @bert(%arg0) : (tensor<{batch_size}x{seq_length}xi64>) -> tensor<{batch_size}x{num_attention_heads}x{seq_length}x{seq_length}xf32>",("""#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map13 = affine_map<(d0) -> (0)>
#map14 = affine_map<(d0) -> (d0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map16 = affine_map<(d0, d1, d2, d3) -> (0)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map20 = affine_map<(d0, d1, d2) -> ()>""",mlir_code)


def convnext(expansion_ratio=4):
    """
    Generate ConvNeXt block MLIR code with generalized shapes
    
    Args:
        batch_size: Batch size (N)
        height: Input height (H)
        width: Input width (W)
        in_channels: Input channels (C_in)
        out_channels: Output channels (C_out)
        expansion_ratio: MLP expansion ratio (default: 4)
    """
    batch_size = choice(BATCH_SIZES)
    height = width = choice(HEIGHTS)
    in_channels = choice(CHANNELS)
    out_channels = choice(CHANNELS)
    
    # Derived dimensions
    expanded_channels = in_channels * expansion_ratio
    kernel_size = 7  # Standard ConvNeXt depthwise kernel size
    padding = kernel_size // 2  # Same padding
    stride = 4
    
    mlir_code = f"""func.func @convnext(%arg0: tensor<{batch_size}x{in_channels}x{height}x{width}xf32>) -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32> {{
    
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c7_i64 = arith.constant 7 : i64
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 9.9999999999999995E-7 : f64
    %cst_4 = arith.constant 9.600000e+01 : f32
    %cst_5 = arith.constant 1.41421354 : f32
    %cst_6 = arith.constant 1.920000e+02 : f32
    %cst_7 = arith.constant 3.840000e+02 : f32
    %cst_8 = arith.constant 7.680000e+02 : f32

    %cst2 = arith.constant 2.0 : f32 

%cst_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst = linalg.fill ins(%cst2: f32) outs(%cst_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_9_tensor = tensor.empty() : tensor<{out_channels}x{in_channels}x4x4xf32>
%cst_9 = linalg.fill ins(%cst2: f32) outs(%cst_9_tensor : tensor<{out_channels}x{in_channels}x4x4xf32>) -> tensor<{out_channels}x{in_channels}x4x4xf32>

%cst_10_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst_10 = linalg.fill ins(%cst2: f32) outs(%cst_10_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_11_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst_11 = linalg.fill ins(%cst2: f32) outs(%cst_11_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_13_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst_13 = linalg.fill ins(%cst2: f32) outs(%cst_13_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_12_tensor = tensor.empty() : tensor<{out_channels}x1x{kernel_size}x{kernel_size}xf32>
%cst_12 = linalg.fill ins(%cst2: f32) outs(%cst_12_tensor : tensor<{out_channels}x1x{kernel_size}x{kernel_size}xf32>) -> tensor<{out_channels}x1x{kernel_size}x{kernel_size}xf32>

%cst_15_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst_15 = linalg.fill ins(%cst2: f32) outs(%cst_15_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_16_tensor = tensor.empty() : tensor<{expanded_channels}x{out_channels}xf32>
%cst_16 = linalg.fill ins(%cst2: f32) outs(%cst_16_tensor : tensor<{expanded_channels}x{out_channels}xf32>) -> tensor<{expanded_channels}x{out_channels}xf32>

%cst_17_tensor = tensor.empty() :  tensor<{expanded_channels}xf32>
%cst_17 = linalg.fill ins(%cst2: f32) outs(%cst_17_tensor :  tensor<{expanded_channels}xf32>) ->  tensor<{expanded_channels}xf32>

%cst_18_tensor = tensor.empty() : tensor<{out_channels}x{expanded_channels}xf32>
%cst_18 = linalg.fill ins(%cst2: f32) outs(%cst_18_tensor : tensor<{out_channels}x{expanded_channels}xf32>) -> tensor<{out_channels}x{expanded_channels}xf32>

%cst_19_tensor = tensor.empty() : tensor<{out_channels}xf32>
%cst_19 = linalg.fill ins(%cst2: f32) outs(%cst_19_tensor : tensor<{out_channels}xf32>) -> tensor<{out_channels}xf32>

%cst_20_tensor = tensor.empty() : tensor<{out_channels}x1x1xf32>
%cst_20 = linalg.fill ins(%cst2: f32) outs(%cst_20_tensor : tensor<{out_channels}x1x1xf32>) -> tensor<{out_channels}x1x1xf32>

// ===== INITIAL CONVOLUTION (STEM) =====
%0 = tensor.empty() : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>
%broadcasted = linalg.broadcast ins(%cst : tensor<{out_channels}xf32>) outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) dimensions = [0, 2, 3]

%1 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : vector<2xi64>, strides = dense<{stride}> : vector<2xi64>}} 
     ins(%arg0, %cst_9 : tensor<{batch_size}x{in_channels}x{height}x{width}xf32>, tensor<{out_channels}x{in_channels}x4x4xf32>) 
     outs(%broadcasted : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

// ===== LAYER NORM 1 =====
%2 = tensor.empty() : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>
%3 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%1 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) 
     outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// Mean computation
%4 = tensor.empty() : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>
%5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>
%6 = linalg.generic {{indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}} 
     ins(%3 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
     outs(%5 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.addf %in, %out : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

%7 = linalg.generic {{indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%6 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
     outs(%4 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.divf %in, %cst_4 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

%8 = linalg.generic {{indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%7 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
     outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%9 = linalg.generic {{indexing_maps = [#map4, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%3, %8 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
     outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.subf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// Variance computation
%10 = linalg.generic {{indexing_maps = [#map4, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%9, %9 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.mulf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%11 = linalg.generic {{indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}} 
      ins(%10 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%5 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.addf %in, %out : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

%12 = linalg.generic {{indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%11 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
      outs(%4 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.divf %in, %cst_4 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

%13 = linalg.generic {{indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%12 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
      outs(%4 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.truncf %cst_3 : f64 to f32
  %631 = arith.addf %in, %630 : f32
  linalg.yield %631 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

%14 = linalg.generic {{indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%13 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
      outs(%4 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = math.rsqrt %in : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>

// Normalization
%15 = linalg.generic {{indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%14 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x1xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%16 = linalg.generic {{indexing_maps = [#map4, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%9, %15 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.mulf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// Scale and bias
%17 = linalg.generic {{indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%16, %cst_10 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.mulf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%18 = linalg.generic {{indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%17, %cst_11 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.addf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// Back to channel-first
%19 = linalg.generic {{indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%18 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

// ===== DEPTHWISE CONVOLUTION =====
%padded = tensor.pad %19 low[0, 0, {padding}, {padding}] high[0, 0, {padding}, {padding}] {{
^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
  tensor.yield %cst_0 : f32
}} : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32> to tensor<{batch_size}x{out_channels}x{(height//stride) + 2*padding}x{(width//stride) + 2*padding}xf32>

%broadcasted_190 = linalg.broadcast ins(%cst_13 : tensor<{out_channels}xf32>) outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) dimensions = [0, 2, 3]

%collapsed = tensor.collapse_shape %cst_12 [[0, 1], [2], [3]] : tensor<{out_channels}x1x{kernel_size}x{kernel_size}xf32> into tensor<{out_channels}x{kernel_size}x{kernel_size}xf32>

%20 = linalg.depthwise_conv_2d_nchw_chw {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}} 
     ins(%padded, %collapsed : tensor<{batch_size}x{out_channels}x{(height//stride) + 2*padding}x{(width//stride) + 2*padding}xf32>, tensor<{out_channels}x{kernel_size}x{kernel_size}xf32>) 
     outs(%broadcasted_190 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

// ===== LAYER NORM 2 =====
%21 = linalg.generic {{indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
     ins(%20 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) 
     outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%34 = linalg.generic {{indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%21, %cst_15 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.addf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// ===== POINTWISE MLP (INVERTED BOTTLENECK) =====
%35 = tensor.empty() : tensor<{out_channels}x{expanded_channels}xf32>
%transposed = linalg.transpose ins(%cst_16 : tensor<{expanded_channels}x{out_channels}xf32>) outs(%35 : tensor<{out_channels}x{expanded_channels}xf32>) permutation = [1, 0]

%36 = linalg.generic {{indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%34 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%37 = tensor.empty() : tensor<{batch_size}x{(height//stride)}x{out_channels}x{expanded_channels}xf32>
%38 = linalg.generic {{indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%transposed : tensor<{out_channels}x{expanded_channels}xf32>) 
      outs(%37 : tensor<{batch_size}x{(height//stride)}x{out_channels}x{expanded_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{out_channels}x{expanded_channels}xf32>

%collapsed_191 = tensor.collapse_shape %36 [[0, 1], [2], [3]] : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32> into tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>
%collapsed_192 = tensor.collapse_shape %38 [[0, 1], [2], [3]] : tensor<{batch_size}x{(height//stride)}x{out_channels}x{expanded_channels}xf32> into tensor<{batch_size*(height//stride)}x{out_channels}x{expanded_channels}xf32>

%39 = tensor.empty() : tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>
%40 = linalg.fill ins(%cst_0 : f32) outs(%39 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) -> tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

%41 = linalg.batch_matmul ins(%collapsed_191, %collapsed_192 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{batch_size*(height//stride)}x{out_channels}x{expanded_channels}xf32>) 
     outs(%40 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) -> tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

%expanded = tensor.expand_shape %41 [[0, 1], [2], [3]] output_shape [{batch_size}, {(height//stride)}, {(width//stride)}, {expanded_channels}] : tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32> into tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

%42 = tensor.empty() : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>
%43 = linalg.generic {{indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%expanded, %cst_17 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>, tensor<{expanded_channels}xf32>) 
      outs(%42 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.addf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

// GELU activation
%44 = linalg.generic {{indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%43 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) 
      outs(%42 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  %630 = arith.divf %in, %cst_5 : f32
  %631 = math.erf %630 : f32
  %632 = arith.addf %631, %cst_1 : f32
  %633 = arith.mulf %632, %cst_2 : f32
  %634 = arith.mulf %in, %633 : f32
  linalg.yield %634 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

// Projection back to original channels
%45 = tensor.empty() : tensor<{expanded_channels}x{out_channels}xf32>
%transposed_193 = linalg.transpose ins(%cst_18 : tensor<{out_channels}x{expanded_channels}xf32>) outs(%45 : tensor<{expanded_channels}x{out_channels}xf32>) permutation = [1, 0]

%46 = linalg.generic {{indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%44 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) 
      outs(%42 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32>

%47 = tensor.empty() : tensor<{batch_size}x{(height//stride)}x{expanded_channels}x{out_channels}xf32>
%48 = linalg.generic {{indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%transposed_193 : tensor<{expanded_channels}x{out_channels}xf32>) 
      outs(%47 : tensor<{batch_size}x{(height//stride)}x{expanded_channels}x{out_channels}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{(height//stride)}x{expanded_channels}x{out_channels}xf32>

%collapsed_194 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{expanded_channels}xf32> into tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>
%collapsed_195 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<{batch_size}x{(height//stride)}x{expanded_channels}x{out_channels}xf32> into tensor<{batch_size*((height//stride))}x{expanded_channels}x{out_channels}xf32>

%49 = tensor.empty() : tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>
%50 = linalg.fill ins(%cst_0 : f32) outs(%49 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>) -> tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>

%51 = linalg.batch_matmul ins(%collapsed_194, %collapsed_195 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{expanded_channels}xf32>, tensor<{batch_size*(height//stride)}x{expanded_channels}x{out_channels}xf32>) 
     outs(%50 : tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>) -> tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32>

%expanded_196 = tensor.expand_shape %51 [[0, 1], [2], [3]] output_shape [{batch_size}, {(height//stride)}, {(width//stride)}, {out_channels}] : tensor<{batch_size*(height//stride)}x{(width//stride)}x{out_channels}xf32> into tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

%52 = linalg.generic {{indexing_maps = [#map4, #map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%expanded_196, %cst_19 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>, tensor<{out_channels}xf32>) 
      outs(%2 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.addf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>

// Back to channel-first
%53 = linalg.generic {{indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%52 : tensor<{batch_size}x{(height//stride)}x{(width//stride)}x{out_channels}xf32>) 
      outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) {{
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}} -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

// Layer scaling
%54 = linalg.generic {{indexing_maps = [#map8, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%cst_20, %53 : tensor<{out_channels}x1x1xf32>, tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) 
      outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.mulf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

// ===== RESIDUAL CONNECTION =====
%55 = linalg.generic {{indexing_maps = [#map4, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} 
      ins(%54, %19 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>, tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) 
      outs(%0 : tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>) {{
^bb0(%in: f32, %in_389: f32, %out: f32):
  %630 = arith.addf %in, %in_389 : f32
  linalg.yield %630 : f32
}} -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>

return %55: tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>
    }}
"""

    return f"func.call @convnext(%arg0) : (tensor<{batch_size}x{in_channels}x{height}x{width}xf32>) -> tensor<{batch_size}x{out_channels}x{(height//stride)}x{(width//stride)}xf32>",("""#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map9 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map11 = affine_map<(d0, d1) -> (0, d1)>
#map12 = affine_map<(d0, d1) -> (d1)>
#map13 = affine_map<(d0, d1) -> (d0, d1)>""",mlir_code)

LINALG_OPERATION_GENERATORS = {
    "add": add,
    "matmul": matmul,
    "conv_2d_nchw_fchw": conv_2d_nchw_fchw,
    "conv_2d_nhwc_hwcf": conv_2d_nhwc_hwcf,
    "pooling_nchw_max": pooling_nchw_max,
    "pooling_nchw_sum": pooling_nchw_sum,
    "pooling_ncw_max": pooling_ncw_max,
    "pooling_ncw_sum": pooling_ncw_sum,
    "pooling_nhwc_max": pooling_nhwc_max,
    "pooling_nhwc_min": pooling_nhwc_min,
    "pooling_nhwc_sum": pooling_nhwc_sum,
    "pooling_nwc_max": pooling_nwc_max,
    "pooling_nwc_sum": pooling_nwc_sum,
    "relu": relu,
    "softmax_2d": lambda *args: softmax(*args, dim=2),
    "sigmoid": sigmoid,
}
