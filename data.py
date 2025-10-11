import os

# full list from D:\data_matmul (given by user)
all_files = [
"matmul_1024_1024_128.mlir","matmul_1024_1024_256.mlir","matmul_1024_128_1024.mlir","matmul_1024_128_128.mlir",
"matmul_1024_128_2048.mlir","matmul_1024_128_256.mlir","matmul_1024_128_512.mlir","matmul_1024_128_768.mlir",
"matmul_1024_1536_128.mlir","matmul_1024_2048_128.mlir","matmul_1024_256_1024.mlir","matmul_1024_256_1536.mlir",
"matmul_1024_256_256.mlir","matmul_1024_256_512.mlir","matmul_1024_256_768.mlir","matmul_1024_3072_128.mlir",
"matmul_1024_512_128.mlir","matmul_1024_512_256.mlir","matmul_1024_512_512.mlir","matmul_1024_768_256.mlir",
"matmul_128_1024_1024.mlir","matmul_128_1024_128.mlir","matmul_128_1024_1536.mlir","matmul_128_1024_256.mlir",
"matmul_128_1024_512.mlir","matmul_128_1024_768.mlir","matmul_128_128_1024.mlir","matmul_128_128_128.mlir",
"matmul_128_128_1536.mlir","matmul_128_128_2048.mlir","matmul_128_128_3072.mlir","matmul_128_128_512.mlir",
"matmul_128_128_768.mlir","matmul_128_1536_1024.mlir","matmul_128_1536_128.mlir","matmul_128_1536_256.mlir",
"matmul_128_1536_512.mlir","matmul_128_1536_768.mlir","matmul_128_2048_1024.mlir","matmul_128_2048_128.mlir",
"matmul_128_2048_1536.mlir","matmul_128_2048_256.mlir","matmul_128_2048_512.mlir","matmul_128_2048_768.mlir",
"matmul_128_256_1024.mlir","matmul_128_256_128.mlir","matmul_128_256_2048.mlir","matmul_128_256_3072.mlir",
"matmul_128_256_768.mlir","matmul_128_3072_128.mlir","matmul_128_3072_256.mlir","matmul_128_3072_512.mlir",
"matmul_128_3072_768.mlir","matmul_128_512_1024.mlir","matmul_128_512_128.mlir","matmul_128_512_1536.mlir",
"matmul_128_512_2048.mlir","matmul_128_512_256.mlir","matmul_128_512_3072.mlir","matmul_128_512_512.mlir",
"matmul_128_768_1024.mlir","matmul_128_768_128.mlir","matmul_128_768_1536.mlir","matmul_128_768_256.mlir",
"matmul_128_768_3072.mlir","matmul_128_768_512.mlir","matmul_128_768_768.mlir","matmul_1536_1024_128.mlir",
"matmul_1536_128_128.mlir","matmul_1536_128_1536.mlir","matmul_1536_128_512.mlir","matmul_1536_128_768.mlir",
"matmul_1536_1536_128.mlir","matmul_1536_256_1024.mlir","matmul_1536_256_128.mlir","matmul_1536_256_256.mlir",
"matmul_1536_256_512.mlir","matmul_1536_256_768.mlir","matmul_1536_512_128.mlir","matmul_1536_512_256.mlir",
"matmul_1536_768_256.mlir","matmul_2048_128_1024.mlir","matmul_2048_128_128.mlir","matmul_2048_128_256.mlir",
"matmul_2048_128_512.mlir","matmul_2048_128_768.mlir","matmul_2048_256_128.mlir","matmul_2048_256_512.mlir",
"matmul_2048_256_768.mlir","matmul_2048_512_128.mlir","matmul_2048_512_256.mlir","matmul_2048_768_128.mlir",
"matmul_256_1024_1024.mlir","matmul_256_1024_128.mlir","matmul_256_1024_1536.mlir","matmul_256_1024_256.mlir",
"matmul_256_1024_512.mlir","matmul_256_1024_768.mlir","matmul_256_1280_1000.mlir","matmul_256_128_128.mlir",
"matmul_256_128_1536.mlir","matmul_256_128_2048.mlir","matmul_256_128_256.mlir","matmul_256_128_3072.mlir",
"matmul_256_128_512.mlir","matmul_256_128_768.mlir","matmul_256_1408_1000.mlir","matmul_256_1536_1000.mlir",
"matmul_256_1536_128.mlir","matmul_256_1536_256.mlir","matmul_256_1536_4096.mlir","matmul_256_1536_512.mlir",
"matmul_256_1536_768.mlir","matmul_256_2048_1000.mlir","matmul_256_2048_128.mlir","matmul_256_2048_2048.mlir",
"matmul_256_2048_256.mlir","matmul_256_2048_512.mlir","matmul_256_256_1024.mlir","matmul_256_256_128.mlir",
"matmul_256_256_1536.mlir","matmul_256_256_256.mlir","matmul_256_256_512.mlir","matmul_256_256_768.mlir",
"matmul_256_3072_128.mlir","matmul_256_4096_1024.mlir","matmul_256_512_1024.mlir","matmul_256_512_128.mlir",
"matmul_256_512_1536.mlir","matmul_256_512_2048.mlir","matmul_256_512_256.mlir","matmul_256_512_3072.mlir",
"matmul_256_512_512.mlir","matmul_256_512_768.mlir","matmul_256_768_1024.mlir","matmul_256_768_128.mlir",
"matmul_256_768_1536.mlir","matmul_256_768_2.mlir","matmul_256_768_256.mlir","matmul_256_768_3072.mlir",
"matmul_256_768_512.mlir","matmul_256_768_768.mlir","matmul_3072_128_128.mlir","matmul_3072_128_256.mlir",
"matmul_3072_128_512.mlir","matmul_3072_256_128.mlir","matmul_3072_256_256.mlir","matmul_3072_512_128.mlir",
"matmul_3072_512_256.mlir","matmul_3072_768_128.mlir","matmul_512_1024_128.mlir","matmul_512_1024_256.mlir",
"matmul_512_1024_512.mlir","matmul_512_128_1024.mlir","matmul_512_128_128.mlir","matmul_512_128_1536.mlir",
"matmul_512_128_2048.mlir","matmul_512_128_256.mlir","matmul_512_128_3072.mlir","matmul_512_128_512.mlir",
"matmul_512_128_768.mlir","matmul_512_1536_128.mlir","matmul_512_1536_256.mlir","matmul_512_2048_128.mlir",
"matmul_512_256_1024.mlir","matmul_512_256_128.mlir","matmul_512_256_1536.mlir","matmul_512_256_2048.mlir",
"matmul_512_256_256.mlir","matmul_512_256_512.mlir","matmul_512_256_768.mlir","matmul_512_512_1024.mlir",
"matmul_512_512_128.mlir","matmul_512_512_256.mlir","matmul_512_512_512.mlir","matmul_512_512_768.mlir",
"matmul_512_768_128.mlir","matmul_512_768_256.mlir","matmul_512_768_512.mlir","matmul_512_768_768.mlir",
"matmul_768_1024_128.mlir","matmul_768_128_1536.mlir","matmul_768_128_256.mlir","matmul_768_128_3072.mlir",
"matmul_768_128_512.mlir","matmul_768_128_768.mlir","matmul_768_1536_128.mlir","matmul_768_2048_128.mlir",
"matmul_768_2048_256.mlir","matmul_768_256_1024.mlir","matmul_768_256_128.mlir","matmul_768_256_1536.mlir",
"matmul_768_256_2048.mlir","matmul_768_256_256.mlir","matmul_768_256_768.mlir","matmul_768_3072_128.mlir",
"matmul_768_512_128.mlir","matmul_768_512_256.mlir","matmul_768_512_768.mlir","matmul_768_768_128.mlir",
"matmul_768_768_256.mlir","matmul_768_768_512.mlir"
]

# used files subset
used_files = [
"matmul_256_1024_1024.mlir","matmul_256_1280_1000.mlir","matmul_256_1408_1000.mlir","matmul_256_1536_1000.mlir",
"matmul_256_1536_4096.mlir","matmul_256_2048_1000.mlir","matmul_256_2048_2048.mlir","matmul_256_256_128.mlir",
"matmul_256_256_512.mlir","matmul_256_4096_1024.mlir","matmul_256_512_1024.mlir","matmul_256_768_2.mlir",
"matmul_256_768_3072.mlir","matmul_256_768_768.mlir"
]

unused_files = (set(all_files) - set(used_files))
unused_files = [f for f in unused_files if f.endswith('.mlir') and f.startswith('matmul_')]

unused_files = sorted(unused_files)

print(unused_files[:10])