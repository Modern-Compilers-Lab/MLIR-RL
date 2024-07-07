# pylint: skip-file

import tensorflow as tf
import os

def print_info(message):
    print(f"\033[94m[INFO]\t {message}\033[0m")

def print_success(message):
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")

def print_alert(message):
    print(f"\033[93m[ALERT]\t {message}\033[0m")

def print_error(message):
    print(f"\033[91m[ERROR]\t {message}\033[0m")


def save_model(model, input_shape, save_path):
    
    class TfModule(tf.Module):

        def __init__(self):
            super(TfModule, self).__init__()
            self.model = model()

        @tf.function(input_signature=[
            tf.TensorSpec(input_shape, tf.float32),
        ])
        def my_predict(self, x):
            return self.model(x)

    tf.saved_model.save(TfModule(), save_path)


def tensorflow_to_tosa(model: callable, input_shape: list, output_path: str):
    
    print_info("Model saving ...")
    save_model(model, input_shape, f"{output_path}/saved_model")

    saved_model_path = f"{output_path}/saved_model"
    tf_mlir_output_path = f"{output_path}/mlir_tf.mlir"
    tosa_mlir_output_path = f"{output_path}/mlir_tosa.mlir"
    linalg_mlir_output_path = f"{output_path}/mlir_linalg.mlir"
    loops_mlir_output_path = f"{output_path}/mlir_scf_loops.mlir"


    print_info("[Saved model -> Tf] converting ...")
    os.system(f"tf-mlir-translate --savedmodel-objectgraph-to-mlir  --tf-savedmodel-exported-names=my_predict {saved_model_path} -o {output_path}/mlir_tf_tmp.mlir")
    os.system(f"tf-opt --tf-executor-island-coarsening --canonicalize {output_path}/mlir_tf_tmp.mlir -o {tf_mlir_output_path}")

    print_info("[Tf -> Tosa] converting ...")
    os.system(f"tf-opt --tf-einsum --tf-to-tosa-pipeline {tf_mlir_output_path} -o {tosa_mlir_output_path} ")

    print_info("[Tosa -> Linalg] converting ...")
    os.system(f"tf-opt --pass-pipeline='builtin.module(func.func(tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith))'  {tosa_mlir_output_path} -o {linalg_mlir_output_path}")

    print_info("[Linalg -> Loops] converting ...")
    os.system(f"tf-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --linalg-bufferize --convert-linalg-to-affine-loops {linalg_mlir_output_path} -o {loops_mlir_output_path}")
