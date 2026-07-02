# data_utils

Data pipeline for MLIR-RL: generate benchmarks, convert models, extract operations.

## Directory layout

```
data_utils/
├── model_catalog.py              # Central model registry
├── orchestrate.py                # Unified CLI dispatcher
│
├── convert/                      # Real model → linalg MLIR
│   ├── vision2mlir.py            #   torchvision / ultralytics models
│   ├── transformers2mlir.py      #   HuggingFace transformer models
│   └── gnn2mlir.py               #   Graph Neural Networks (synthetic inputs)
│
├── extract/                      # Full-model MLIR → benchmark fragments
│   ├── extract_ops.py            #   Single linalg op extraction
│   ├── extract_blocks.py         #   Multi-op block extraction (via AST dumper)
│   └── batch_policy.py           #   Batch-size selection for blocks
│
├── generate/                     # Synthetic benchmark generation
│   ├── mlir_generators.py        #   Random linalg op generator library
│   ├── generate_synthetic.py     #   Generate synthetic .mlir files
│   └── id_allocator.py           #   Sequential ID allocation across runs
│
└── postprocess/                  # MLIR file post-processing
    ├── wrap_mlir.py              #   Add timed @main entry point
    └── strip_mlir.py             #   Remove embedded weight constants
```

---

## Pipelines

### 1. Convert a real neural network to linalg MLIR

Each converter takes a pretrained model, exports it through the ONNX route
(`PyTorch → .onnx → shape inference → torch-mlir → linalg MLIR`), and strips
the resulting file. Use `--keep-onnx` to preserve intermediate files.

**Vision models** (`convert/vision2mlir.py`):

```bash
python -m data_utils.convert.vision2mlir --model resnet18
python -m data_utils.convert.vision2mlir --model vit_b_16 --output-dir data/nn/raw_bench
python -m data_utils.convert.vision2mlir --model yolov8m --keep-onnx

# convnext and resnet50 need the direct torch_mlir route
python -m data_utils.convert.vision2mlir --model convnext_tiny --backend direct
```

Supported: `resnet18`, `resnet50`, `resnext50`, `efficientnet_b0`, `mobilenet_v2`,
`mobilenet_v3_small`, `densenet121`, `vit_b_16`, `convnext_tiny/small/base/large`,
`vgg11`, `vgg16`, `yolov8m`.

**Transformer models** (`convert/transformers2mlir.py`):

```bash
python -m data_utils.convert.transformers2mlir --model bert
python -m data_utils.convert.transformers2mlir --model gpt2 --keep-onnx
python -m data_utils.convert.transformers2mlir --model llama3_2_1b
```

Supported: `bert`, `distilbert`, `roberta`, `albert`, `gpt2`, `t5`, `bart`,
`llama3_2_1b`, `whisper_base`, `lstm`.

**GNN models** (`convert/gnn2mlir.py`):

GNNs use synthetic inputs (128 nodes, 64 features) — no pretrained weights.

```bash
python -m data_utils.convert.gnn2mlir --model gcn
python -m data_utils.convert.gnn2mlir --model all
```

Supported: `gcn`, `graphsage`, `gat`, `gin`.

---

### 2. Extract benchmark fragments from full-model MLIR

**Single ops** (`extract/extract_ops.py`):

Extracts individual linalg operations from a full-model MLIR file. Each op is
wrapped in its own `@main` with concrete batch sizes and function arguments.

```bash
python -m data_utils.extract.extract_ops \
    --input data/nn/raw_bench/resnet50_linalg.mlir \
    --output-dir data/nn/code_files/resnet50/ \
    --batch-size 1 --model-name resnet50
```

**Multi-op blocks** (`extract/extract_blocks.py`):

Extracts contiguous multi-op sequences using consumer→producer graph paths
from an external AST dumper. Requires `AST_DUMPER_BIN_PATH` in your env.

```bash
python -m data_utils.extract.extract_blocks \
    --input data/nn/raw_bench/resnet50_linalg.mlir \
    --output-dir data/nn/code_files/bench_train/ \
    --window 5 --stride 3
```

---

### 3. Generate synthetic benchmarks

`generate/generate_synthetic.py` creates random linalg operations as `.mlir`
files without requiring any MLIR runtime.

```bash
python -m data_utils.generate.generate_synthetic \
    --output-dir data/nn/synthetic/ \
    --num-singles 200 --num-benchs 50
```

The generator library (`generate/mlir_generators.py`) provides individual op
generators (matmul, conv2d, relu, softmax, etc.) and composite sub-graph
generators (resnet blocks, vgg blocks, bert blocks).

---

### 4. Post-processing

**Wrap with timed `@main`** (`postprocess/wrap_mlir.py`):

```bash
python -m data_utils.postprocess.wrap_mlir \
    --input data/nn/raw_bench/resnet50_linalg.mlir \
    --model-name resnet50 \
    --output data/nn/code_files/model_resnet50.mlir
```

**Strip weight constants** (`postprocess/strip_mlir.py`):

```bash
python -m data_utils.postprocess.strip_mlir model.mlir --replace
python -m data_utils.postprocess.strip_mlir model.mlir --output stripped.mlir
```

---

## Unified CLI

`orchestrate.py` dispatches all subcommands:

```bash
python -m data_utils.orchestrate vision       --model resnet18
python -m data_utils.orchestrate transformer  --model bert
python -m data_utils.orchestrate gnn          --model gcn
python -m data_utils.orchestrate wrap         --input model.mlir --model-name foo --output wrapped.mlir
python -m data_utils.orchestrate strip        model.mlir --replace
```

---

## model_catalog.py

Central registry of all supported models. Each entry specifies a `category`
(`vision`, `transformer`, `gnn`), a `framework`, and framework-specific keys.

```python
from data_utils.model_catalog import VISION_MODELS, TRANSFORMER_MODELS, GNN_MODELS
```
