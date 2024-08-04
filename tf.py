import tensorflow as tf
import numpy as np
import time
# @tf.function(jit_compile=True)
# def conv(input_tensor,filter_tensor,strides,dilations):
#     return tf.nn.conv2d(input_tensor, filter_tensor, strides=strides, padding='VALID', dilations=dilations)

# input_tensor =  tf.constant(np.random.rand(32, 230, 230, 3), dtype=tf.float32)
# filter_tensor = tf.constant(np.random.rand(7, 7, 3, 64), dtype=tf.float32)

# # Define strides and dilations
# strides = [1, 2, 2, 1]
# dilations = [1, 1, 1, 1]




# @tf.function(jit_compile=True)
def matmul(a, b):
    return tf.matmul(a, b)


matmul_shape = [
    [(32,128), (128,128)],
    [(32,512), (512,64)],
    [(32,1024), (1024,1024)],
    [(64,128), (128,32)],
    [(64,1024), (1024,64)],
    [(128,512), (512,256)],
    [(256,512), (512,32)],
    [(512,32), (32,512)],
    [(512,1024), (1024,64)],
    [(1024,64), (64,256)],

]

for shape1, shape2 in matmul_shape:

    print(f'({shape1[0]},{shape1[1]})x({shape2[0]},{shape2[1]})')
    
    a = tf.constant(np.random.rand(*shape1), dtype=tf.float32)
    b = tf.constant(np.random.rand(*shape2), dtype=tf.float32)

    for i in range(30):
        
        start_time = tf.timestamp()
        # output_tensor = conv(input_tensor,filter_tensor,strides,dilations)
        matmul(a, b)
        end_time = tf.timestamp()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print("Time elapsed for matmul operation: {:.9f} seconds".format(elapsed_time.numpy()))
        
    
    print('\n\n\n\n')