"""
model_catalog.py
----------------
Central model catalog for data_utils.

Single source of truth for all supported models.  Each script imports from
here instead of maintaining its own SUPPORTED_MODELS list.

To add a new model:
  1. Add an entry to ``MODEL_REGISTRY`` below.
  2. If it is a transformer, also add its input spec to ``TRANSFORMER_INPUT_SPECS``.
  3. The conversion scripts will pick it up automatically.
"""

# ── Unified model registry ────────────────────────────────────────────────
#
# Every model lives here.  Scripts filter by ``category`` to get their list.
#
# Required keys:
#   category  – "vision" | "transformer" | "gnn"
#   framework – "torchvision" | "huggingface" | "ultralytics" | "custom"
#
# Framework-specific keys (depending on ``framework``):
#   torchvision_name  – name in torchvision.models  (torchvision)
#   hf_repo           – HuggingFace repo ID          (huggingface)
#   input_shape       – tuple for dummy input         (vision only)
#   dummy_input_size  – int, spatial dim for 4-D tensors (vision, default 224)

MODEL_REGISTRY = {
    # ── Vision (torchvision) ──────────────────────────────────────────────
    "resnet18":          {"category": "vision", "framework": "torchvision", "torchvision_name": "resnet18"},
    "resnet50":          {"category": "vision", "framework": "torchvision", "torchvision_name": "resnet50"},
    "resnext50":         {"category": "vision", "framework": "torchvision", "torchvision_name": "resnext50_32x4d"},
    "efficientnet_b0":   {"category": "vision", "framework": "torchvision", "torchvision_name": "efficientnet_b0"},
    "mobilenet_v2":      {"category": "vision", "framework": "torchvision", "torchvision_name": "mobilenet_v2"},
    "mobilenet_v3_small": {"category": "vision", "framework": "torchvision", "torchvision_name": "mobilenet_v3_small"},
    "densenet121":       {"category": "vision", "framework": "torchvision", "torchvision_name": "densenet121"},
    "vit_b_16":          {"category": "vision", "framework": "torchvision", "torchvision_name": "vit_b_16"},
    "convnext_tiny":     {"category": "vision", "framework": "torchvision", "torchvision_name": "convnext_tiny"},
    "convnext_small":    {"category": "vision", "framework": "torchvision", "torchvision_name": "convnext_small"},
    "convnext_base":     {"category": "vision", "framework": "torchvision", "torchvision_name": "convnext_base"},
    "convnext_large":    {"category": "vision", "framework": "torchvision", "torchvision_name": "convnext_large"},
    "vgg11":             {"category": "vision", "framework": "torchvision", "torchvision_name": "vgg11"},
    "vgg16":             {"category": "vision", "framework": "torchvision", "torchvision_name": "vgg16"},

    # ── Vision (ultralytics) ──────────────────────────────────────────────
    "yolov8m":           {"category": "vision", "framework": "ultralytics", "input_shape": (640, 640)},

    # ── Transformers (HuggingFace) ────────────────────────────────────────
    "bert":              {"category": "transformer", "framework": "huggingface", "hf_repo": "bert-base-uncased"},
    "distilbert":        {"category": "transformer", "framework": "huggingface", "hf_repo": "distilbert-base-uncased"},
    "roberta":           {"category": "transformer", "framework": "huggingface", "hf_repo": "roberta-base"},
    "albert":            {"category": "transformer", "framework": "huggingface", "hf_repo": "albert-base-v2"},
    "deberta":           {"category": "transformer", "framework": "huggingface", "hf_repo": "microsoft/deberta-v3-base"},
    "electra":           {"category": "transformer", "framework": "huggingface", "hf_repo": "google/electra-small-discriminator"},
    "gpt2":              {"category": "transformer", "framework": "huggingface", "hf_repo": "gpt2-medium"},
    "t5":                {"category": "transformer", "framework": "huggingface", "hf_repo": "t5-small"},
    "bart":              {"category": "transformer", "framework": "huggingface", "hf_repo": "facebook/bart-base"},
    "llama3_2_1b":       {"category": "transformer", "framework": "huggingface", "hf_repo": "unsloth/Llama-3.2-1B"},
    "whisper_base":      {"category": "transformer", "framework": "huggingface", "hf_repo": "openai/whisper-base"},

    # ── Transformers (synthetic / custom) ─────────────────────────────────
    "lstm":              {"category": "transformer", "framework": "custom"},
    "lstm_seq2seq":      {"category": "transformer", "framework": "custom"},
    "gru":               {"category": "transformer", "framework": "custom"},
    "bilstm":            {"category": "transformer", "framework": "custom"},

    # ── GNN ───────────────────────────────────────────────────────────────
    "gcn":               {"category": "gnn", "framework": "custom"},
    "graphsage":         {"category": "gnn", "framework": "custom"},
    "gat":               {"category": "gnn", "framework": "custom"},
    "gin":               {"category": "gnn", "framework": "custom"},
}


# ── Convenience lists (derived from registry) ─────────────────────────────

VISION_MODELS = [name for name, m in MODEL_REGISTRY.items() if m["category"] == "vision"]
TRANSFORMER_MODELS = [name for name, m in MODEL_REGISTRY.items() if m["category"] == "transformer"]
GNN_MODELS = [name for name, m in MODEL_REGISTRY.items() if m["category"] == "gnn"]


# ── HuggingFace repo mapping (derived from registry) ──────────────────────

TRANSFORMER_HF_REPOS = {
    name: m["hf_repo"]
    for name, m in MODEL_REGISTRY.items()
    if m["category"] == "transformer" and m.get("hf_repo")
}


# ── Synthetic input specs for transformers ────────────────────────────────
#
# Used by transformers2mlir.py to build example inputs when a tokenizer
# is not available or when synthetic mode is preferred.
#
# Schema per entry:
#   default_mode     – "synthetic" or "tokenizer"
#   defaults         – scalar dims resolved at build time
#   synthetic_tensors – ordered list of tensor specs:
#       name, dtype, shape, init, randint_low/int_high (optional)
#   tokenizer_text   – fallback text for tokenizer mode
#   output_names     – names passed to ONNX export

TRANSFORMER_INPUT_SPECS = {
    "bert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "distilbert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "roberta": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50265},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "albert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30000},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "deberta": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 128000},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "electra": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "gpt2": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50257, "use_cache": False},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "t5": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 32128},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
        ],
        "tokenizer_text": "Studies show that owning a dog is good for you",
        "output_names": ["last_hidden_state"],
    },
    "bart": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50265, "use_cache": False},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "The cat sat on the mat.",
        "output_names": ["last_hidden_state"],
    },
    "llama3_2_1b": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 128256},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["logits"],
    },
    "whisper_base": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "n_mels": 80, "seq_len": 3000},
        "synthetic_tensors": [
            {"name": "input_features", "dtype": "float32", "shape": ["batch_size", "n_mels", "seq_len"], "init": "randn"},
        ],
        "output_names": ["last_hidden_state"],
    },
    "lstm": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "h0", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
            {"name": "h1", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "lstm_seq2seq": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "src", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "tgt", "dtype": "float32", "shape": ["batch_size", "seq_len", "hidden_size"], "init": "randn"},
            {"name": "h", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "gru": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "h0", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "bilstm": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 384},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "hf", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
            {"name": "hb", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
}
