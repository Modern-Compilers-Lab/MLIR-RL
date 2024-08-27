import tensorflow as tf
import numpy as np
import time
# @tf.function(jit_compile=True)
def conv(input_tensor,filter_tensor,strides,dilations):
    return tf.nn.conv2d(input_tensor, filter_tensor, strides=strides, padding='VALID', dilations=dilations)

# input_tensor =  tf.constant(np.random.rand(32, 230, 230, 3), dtype=tf.float32)
# filter_tensor = tf.constant(np.random.rand(7, 7, 3, 64), dtype=tf.float32)

# # Define strides and dilations
# strides = [1, 2, 2, 1]
# dilations = [1, 1, 1, 1]




# @tf.function(jit_compile=True)
def matmul(a, b):
    return tf.matmul(a, b)

@tf.function(jit_compile=True)
def add(a, b):
    return tf.add(a, b)

@tf.function(jit_compile=True)
def maxpool(input_tensor, ksize, strides, padding='VALID'):
    return tf.nn.max_pool2d(input_tensor, ksize=ksize, strides=strides, padding=padding)

@tf.function(jit_compile=True)
def relu(a):
    return tf.nn.relu(a)



matmul_shape = [
    
    [(256, 2048), (2048, 1000)],
    [(256, 1280), (1280, 1000)],
    [(256, 1536), (1536, 1000)],
    [(256, 1408), (1408, 1000)],
    [(256, 768), (768, 768)],
    [(256, 1024), (1024, 1024)],
    [(256, 768), (768, 3072)],
    [(256, 256), (256, 128)],
    [(256, 4096), (4096, 1024)],
    [(256, 1536), (1536, 4096)],
    [(256, 768), (768, 2)],
    [(256, 2048), (2048, 2048)],
    [(256, 128), (128, 256)],
    [(256, 256), (256, 512)],
    [(256, 512), (512, 1024)],

]


conv_2d_shapes = [
    
    [(256, 14, 14, 256), (3, 3, 256, 256), 1],
    [(256, 14, 14, 256), (1, 1, 256, 1024), 1],
    [(256, 28, 28, 128), (3, 3, 128, 128), 1],
    [(256, 28, 28, 128), (1, 1, 128, 512), 1],
    [(256, 28, 28, 512), (1, 1, 512, 128), 1],
    [(256, 14, 14, 128), (3, 3, 128, 32), 1],
    [(256, 7, 7, 128), (3, 3, 128, 32), 1],
    [(256, 16, 16, 256), (3, 3, 256, 256), 1],
    [(256, 14, 14, 576), (1, 1, 576, 576), 1],
    [(256, 28, 28, 128), (3, 3, 128, 32), 1],
    [(256, 14, 14, 336), (1, 1, 336, 336), 1],
    [(256, 56, 56, 64), (3, 3, 64, 64), 1],
    [(256, 28, 28, 448), (1, 1, 448, 448), 1],
    [(256, 56, 56, 64), (1, 1, 64, 256), 1],
    [(256, 128, 128, 16), (7, 7, 16, 8), 2],
    [(256, 64, 64, 64), (3, 3, 64, 16), 1],
    [(256, 32, 32, 32), (7, 7, 32, 256), 2],
    [(256, 230, 230, 3), (7, 7, 3, 64), 2],
    
]


maxpool_shapes = [
    [(256, 114, 114, 64), (3, 3), 2],
    [(256, 147, 147, 64), (3, 3), 2],
    [(256, 71, 71, 192), (3, 3), 2],
    [(256, 167, 167, 42), (3, 3), 2],
    [(256, 85, 85, 84), (3, 3), 2],
    [(256, 43, 43, 336), (3, 3), 2],
    [(256, 23, 23, 672), (3, 3), 2],
    [(256, 113, 113, 11), (3, 3), 2],
    [(256, 57, 57, 22), (3, 3), 2],
    [(256, 29, 29, 88), (3, 3), 2],    
]


add_shapes = [
    (256, 14, 14, 1024),
    (256, 28, 28, 512),
    (256, 7, 7, 2048),
    (256, 56, 56, 256),
    (256, 21, 21, 336),
    (256, 11, 11, 672),
    (256, 42, 42, 168),
    (256, 15, 15, 304),
    (256, 14, 14, 88),
    (256, 7, 7, 176),
]

relu_shapes = [
    (256, 2048),
    (256, 512),
    (256, 1000),
    (256, 100),
    (256, 10),
    (256, 57, 57, 64),
    (256, 74, 74, 64),
    (256, 36, 36, 192),
    (256, 85, 85, 42),
    (256, 43, 43, 84),
    (256, 23, 23, 336),
    (256, 14, 14, 672),
    (256, 29, 29, 22),
    (256, 14, 14, 88),
]

for shape1 in relu_shapes:
    
    # print(f'({shape1[0]},{shape1[1]})x({shape2[0]},{shape2[1]})')
    # print(f'({shape1})+({shape1})')
    
    a = tf.constant(np.random.rand(*shape1), dtype=tf.float32)
    # b = tf.constant(np.random.rand(*shape2), dtype=tf.float32)
    # b = tf.constant(np.random.rand(*shape1), dtype=tf.float32)

    
    res = []
    for i in range(30):
        
        start_time = tf.timestamp()
        # output_tensor = conv(input_tensor,filter_tensor,strides,dilations)
        # matmul(a, b)
        # add(a, b)
        relu(a)
        end_time = tf.timestamp()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        res.append(elapsed_time)
        # print("Time elapsed for matmul operation: {:.9f} seconds".format(elapsed_time.numpy()))
        
    # print(f"Avg time: {np.mean(res):.9f} seconds")
    # print(f"Med time: {np.median(res):.9f} seconds")
    # print('\n\n')
    print(f"{np.mean(res):.9f}")
    
    
    
# for input_shape, kernel, strides in maxpool_shapes:

#     # print(input_shape, kernel, strides)
    
#     input_tensor =  tf.constant(np.random.rand(*input_shape), dtype=tf.float32)
#     filter_tensor = tf.constant(np.random.rand(*kernel), dtype=tf.float32)

#     # # Define strides and dilations
#     # strides = [1, strides, strides, 1]
#     # dilations = [1, 1, 1, 1]
    

    
#     res = []
#     for i in range(30):
        
#         start_time = tf.timestamp()
#         # output_tensor = conv(input_tensor,filter_tensor,strides,dilations)
#         output_tensor = maxpool(input_tensor,kernel[0],strides)
#         end_time = tf.timestamp()

#         # Calculate the elapsed time
#         elapsed_time = end_time - start_time
#         res.append(elapsed_time)
#         # print("Time elapsed for conv2d operation: {:.9f} seconds".format(elapsed_time.numpy()))
        
#     # print(f"Avg time: {np.mean(res):.9f} seconds")
#     # print(f"Med time: {np.median(res):.9f} seconds")
#     print(f"{np.mean(res):.9f}")
#     # print('\n\n')