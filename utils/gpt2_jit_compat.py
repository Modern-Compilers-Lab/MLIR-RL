"""
GPT2 JIT-compatible wrapper for torch.jit.trace() benchmarking.

The HuggingFace GPT2Model.forward() calls create_causal_mask() which mixes
config access, version checks, and SDPA internals that torch.jit.trace()
cannot capture.  This module bypasses the entire GPT2Model.forward() by:

1. Extracting weight-carrying submodules (wte, wpe, h, ln_f) from a
   pretrained GPT2Model.
2. Implementing a pure-tensor forward() with no config access.
3. Building the 4D causal attention mask manually.

Usage:
    from utils.gpt2_jit_compat import make_gpt2_jit_compat
    model, inputs = make_gpt2_jit_compat("gpt2")
    traced = torch.jit.trace(model, inputs)
"""

import torch
import torch.nn as nn


class GPT2JITCompat(nn.Module):
    """Minimal GPT2 wrapper for JIT tracing — no config objects, pure tensors."""

    def __init__(self, wte, wpe, h, ln_f, dropout_p=0.1):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.h = nn.ModuleList(h)
        self.ln_f = ln_f
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        hidden_states = self.wte(input_ids)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.wpe(position_ids)

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, device=input_ids.device)
        ).view(1, 1, seq_length, seq_length)
        causal_mask = (1.0 - causal_mask) * -10000.0

        padding_mask = (
            (attention_mask[:, None, None, :] == 0).float() * -10000.0
        )
        attention_mask_4d = causal_mask + padding_mask

        for layer in self.h:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask_4d)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


def make_gpt2_jit_compat(model_name="gpt2"):
    """Load a pretrained GPT2 model, wrap it for JIT tracing.

    Returns (model, (input_ids, attention_mask)) ready for _time_jit().
    """
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    pretrained = GPT2Model.from_pretrained(model_name).eval()
    pretrained.config.use_cache = False

    enc = tokenizer(
        "Hello from MLIR-RL!",
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    dropout_p = getattr(pretrained.config, "embd_pdrop", 0.1)

    wrapper = GPT2JITCompat(
        wte=pretrained.wte,
        wpe=pretrained.wpe,
        h=pretrained.h,
        ln_f=pretrained.ln_f,
        dropout_p=dropout_p,
    ).eval()

    return wrapper, (input_ids, attention_mask)


# Pre-built loaders for each variant
def _make_loader(variant):
    def loader():
        return make_gpt2_jit_compat(variant)
    return loader


GPT2_LOADERS = {
    "gpt2":        _make_loader("gpt2"),
    "gpt2-medium": _make_loader("gpt2-medium"),
    "gpt2-large":  _make_loader("gpt2-large"),
}
