# MLIR-RL Dataset: Model Selection & Structure

## Removed Models

| Model | Category | Reason for Removal |
|---|---|---|
| ResNet-18 | CNN | Redundant with ResNet-50. Same architectural pattern (residual conv stacks), just shallower. |
| DeBERTa | Encoder Transformer | Encoder-only family over-represented. Near-identical MLIR compute graphs to BERT. |
| RoBERTa | Encoder Transformer | Same rationale as DeBERTa. Different pretraining, not a different compute graph. |
| ELECTRA | Encoder Transformer | Discriminator/generator design is training-time only; inference IR equals BERT. |
| GRU | RNN | Sequential recurrent ops incompatible with parallelizable workloads. |
| BiLSTM | RNN | Same rationale as GRU. Gate-heavy sequential compute, not representative of production. |
| GCN | Graph Neural Network | Functionally subsumed by GIN. Simplified special case of GIN's aggregation scheme. |
| GraphSAGE | Graph Neural Network | Sampling-based aggregation collapses to patterns already covered by GIN. |
| DenseNet-121 | CNN | Dense connectivity pattern produces op types already well-covered by ResNet/EfficientNet. |
| VGG-11 | CNN | Replaced by VGG-16 for deeper conv stacks with more scheduling surface. |
| LSTM / LSTM-Seq2Seq | RNN | Recurrent patterns incompatible with modern parallelizable benchmarks. |

---

## Kept Models (18)

### Encoder Transformers

| Model | Reason for Keeping |
|---|---|
| BERT | Single representative of encoder-only family. Standard self-attention + FFN (matmul-heavy). |
| ALBERT | Parameter-sharing design; distinct IR footprint despite similar architecture. |
| DistilBERT | Distilled architecture with fewer layers; tests agent generalization across depth. |

### Decoder / Seq2Seq

| Model | Reason for Keeping |
|---|---|
| GPT-2 (medium) | Autoregressive decoder with causal attention mask. 345M params, 24 layers. |
| BART | Encoder-decoder architecture. Covers seq2seq compute patterns. |
| T5 | Encoder-decoder with distinct tokenization/attention. Widely deployed in production. |

### Vision

| Model | Reason for Keeping |
|---|---|
| ViT-B/16 | Canonical vision Transformer. Attention-heavy, matmul-dominant. |
| ResNet-50 | Canonical CNN backbone. Conv2D-dominant with residual connections. |
| ResNeXt-50 | Grouped convolutions; stress-tests group conv scheduling. |
| MobileNetV3-Small | Depthwise separable convolutions + SE blocks; edge-efficiency scenarios. |
| EfficientNet-B0 | Compound scaling across width, depth, resolution; diverse op shapes. |
| ConvNeXt-Tiny | Modernized CNN with 7×7 kernels, LayerNorm, GELU. Bridges CNN→Transformer gap. |
| VGG-16 | Deep uniform conv stacks (no residuals); sanity baseline for pure conv scheduling. |

### Graph Neural Networks

| Model | Reason for Keeping |
|---|---|
| GAT | Attention-weighted neighbor aggregation; closest to modern attention mechanisms. |
| GIN | Weisfeiler-Lehman equivalent; sum aggregation + MLP layers on irregular data. |

### Audio Transformers

| Model | Reason for Keeping |
|---|---|
| Whisper-base (encoder) | Conv1d stem + self-attention layers. Distinct input shape (80 mel bins × time) and GELU. Cross-modality diversity. |

### Object Detection

| Model | Reason for Adding |
|---|---|
| YOLOv8m (backbone) | Conv2D-dominant (1×1 and 3×3 kernels), SPPF pooling block. High real-world deployment relevance. |

### Modern LLMs

| Model | Reason for Adding |
|---|---|
| Llama 3.2 1B | GQA, RoPE positional encoding, RMSNorm. Distinct batch_matmul shapes; compact enough to benchmark. |

---

## Future Works

| Model | Category | Rationale for Deferral |
|---|---|---|
| Qwen 0.8B | LLM | Architecturally very close to Llama (GQA, RMSNorm, rotary embeddings). |
| DeepSeek 1.5B (dense) | LLM | Near-identical op structure to Llama. MoE variant (V2/V3) deferred with Gemma MoE. |
| Gemma MoE 2B | Mixture-of-Experts | Top-k routing introduces data-dependent scheduling problems. Requires dedicated MoE support. |

---

## Data Extraction Pipeline

### Raw Models → Linalg MLIR

18 PyTorch/HuggingFace models are converted to `linalg-on-tensors` MLIR via:

```
PyTorch model → ONNX (opset 18, torch.onnx.export)
              → torch-mlir-import-onnx
              → torch-mlir-opt (lower to linalg)
```

GPT-2 and Llama 3.2 use a wrapper that passes `cache_position` explicitly to bypass `DynamicCache` tracing issues in transformers ≥4.57. Llama 3.2 is exported in bfloat16 to fit within Slurm memory limits.

### Single-Op Extraction (`extract_ops.py`)

Isolated linalg operations are extracted from raw `_linalg.mlir` files. Each output file wraps a single op in a timed `@main` function with `@nanoTime()` calls.

| Variant | Flag | Purpose | Result |
|---|---|---|---|
| **Full Fidelity** | `--no-require-reduction` | Proof of model op coverage (all generics kept) | 952 files → `single_bench_full/` |
| **Train-Efficient** | `--generic-ratio 0.25` | Training + ablation + performance reporting | 405 files → `single_bench/` |

The `--generic-ratio` flag caps `linalg.generic` ops at a fraction of total generics per model. Reduction-containing generics are prioritised; remaining slots filled by random sampling of element-wise generics. Default ratio 0.25 keeps the generic proportion balanced (~25% of generics per model) regardless of model depth.

### Block Extraction (`extract_blocks.py`)

5-op sliding windows (stride 3) are extracted along consumer→producer dataflow paths via the AST dumper. Each block is a self-contained timed benchmark.

| Variant | Flag | Purpose | Result |
|---|---|---|---|
| **Full** | (default) | Real subgraph coverage | 10,198 blocks → `bench/` (backed up) |
| **Train-Efficient** | `--skip-pure-elementwise` | Training-focused; skip blocks with zero heavy ops | 7,393 blocks → `bench_train/` |

A block is skipped if every op in it is an element-wise `linalg.generic` (no matmul, conv2d, pooling, reduce, or reduction-containing generic). This ensures every block has at least one meaningful scheduling target.

---

## Directory Structure

```
data/
├── nn/                           # Neural network model benchmarks
│   ├── raw_bench/                # 18 *_linalg.mlir files
│   ├── single_bench/             # 405 isolated ops (train-efficient)
│   ├── single_bench_full/        # 952 isolated ops (full fidelity)
│   └── bench_train/              # 7,393 blocks (train-efficient)
│
├── legacy/                       # Synthetic legacy benchmarks
│   ├── bench/                    # 3,738 synthetic single-op benchmarks (filtered)
│   ├── single_bench/             # 234 synthetic single-op benchmarks (filtered)
│   ├── bench_removed/            # 3,663 pure element-wise (archived)
│   └── single_bench_removed/     # 64 excess generics (archived)
│
├── all/                          # Final merged dataset (flat per directory)
│   ├── train/                    # 9,407 files (80% stratified from all sources)
│   ├── eval/                     # 2,363 files (20% stratified from all sources)
│   └── eval_full/                # 952 files (100% of nn/single_bench_full)
│
└── backup/                       # Archived data
    ├── nn_bench_unfiltered/       # 10,198 original unfiltered blocks
    ├── all_code_files_old/        # Stale old all/code_files
    └── stale_dirs/               # generated/, removed/, old files
```

### all/ Stratification

`all/train/` and `all/eval/` are populated by an 80/20 stratified split with seed 42. Sources:

- `nn/single_bench/` (405) + `nn/bench_train/` (7,393) — stratified by model name
- `legacy/single_bench/` (234) + `legacy/bench/` (3,738) — stratified by op family

All 18 models + both legacy pools are proportionally represented in both train and eval. Merged flat: `{model}_{op}_{N}.mlir`, `{model}_block_{N}.mlir`, `bench_{N}.mlir`, `single_bench_{N}.mlir` coexist in a single directory.

---

## Final Dataset Summary

| Category | Models | Count |
|---|---|---|
| Encoder Transformers | BERT, ALBERT, DistilBERT | 3 |
| Decoder / Seq2Seq | GPT-2 (medium), BART, T5 | 3 |
| Vision Transformers | ViT-B/16 | 1 |
| CNN Backbones | ResNet-50, ResNeXt-50, MobileNetV3-Small, EfficientNet-B0, ConvNeXt-Tiny, VGG-16 | 6 |
| Object Detection | YOLOv8m (backbone) | 1 |
| Modern LLMs | Llama 3.2 1B | 1 |
| Audio Transformers | Whisper-base (encoder) | 1 |
| Graph Neural Networks | GAT, GIN | 2 |
| **Total models** | | **18** |

| Dataset Split | Files | Composition |
|---|---|---|
| `all/train/` | 9,407 | 80% of nn single_bench + nn bench_train + legacy single_bench + legacy bench |
| `all/eval/` | 2,363 | 20% of same |
| `all/eval_full/` | 952 | 100% of nn/single_bench_full (all model ops, unfiltered) |
| **Total** | **12,722** | |
