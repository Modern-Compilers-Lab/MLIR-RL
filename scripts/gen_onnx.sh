#!/bin/bash
# Generate intermediate ONNX files for all models in data/nn/raw_bench/
# Skips models that already have .onnx files (gat, t5, resnext50)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

PYTHON=/home/tb3654/.conda/envs/mlir/bin/python3
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH=/home/tb3654/.conda/envs/mlir/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

OUTPUT_DIR="$PROJECT_ROOT/data/nn/raw_bench"

echo "=== Vision models ==="
for model in convnext_tiny efficientnet_b0 mobilenet_v3_small resnet50 vgg16 vit_b_16 yolov8m; do
    echo "--- $model ---"
    $PYTHON -m data_utils.convert.vision2mlir --model "$model" --output-dir "$OUTPUT_DIR" --keep-onnx 2>&1 || echo "FAILED: $model"
done

echo "=== Transformer models ==="
for model in albert bart bert distilbert gpt2 llama3_2_1b whisper_base; do
    echo "--- $model ---"
    $PYTHON -m data_utils.convert.transformers2mlir --model "$model" --output-dir "$OUTPUT_DIR" --keep-onnx 2>&1 || echo "FAILED: $model"
done

echo "=== GNN models ==="
echo "--- gin ---"
$PYTHON -m data_utils.convert.gnn2mlir --model gin --output-dir "$OUTPUT_DIR" --keep-onnx 2>&1 || echo "FAILED: gin"

echo "=== Done ==="
ls "$OUTPUT_DIR"/*.onnx 2>/dev/null | wc -l
echo "ONNX files generated."
