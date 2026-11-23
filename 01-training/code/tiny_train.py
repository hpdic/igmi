"""
Chapter 1 Companion: Minimal Modern LLM Training Loop
-----------------------------------------------------
Architecture matches modern decoder-only LLMs (Qwen / DeepSeek building blocks):
  - Pre-norm RMSNorm
  - Rotary Positional Embeddings (RoPE)
  - Causal Grouped-Query Attention (GQA) via PyTorch SDPA (FlashAttention on H100)
  - SwiGLU Feed-Forward Network
  - AdamW + mixed precision (BF16 autocast on CUDA, FP32 on CPU)

Zero external dependencies beyond PyTorch. Runs on NVIDIA H100 (default) or CPU.

Usage:
  python 01-training/code/tiny_train.py                # auto: CUDA (H100) + bf16
  python 01-training/code/tiny_train.py --device cpu   # CPU + fp32
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = (
    "Training an LLM runs all sequence positions in parallel through big matrix multiplies. "
    "Because every token is known upfront, training is compute-bound on Tensor Cores. "
    "Static memory is dominated by AdamW optimizer states, not the BF16 weights. "
    "Qwen uses dense SwiGLU layers with Grouped-Query Attention to shrink the KV cache. "
    "DeepSeek uses Multi-head Latent Attention, fine-grained MoE experts, FP8 GEMMs, and DualPipe. "
    "Autoregressive inference splits into compute-bound prefill and memory-bandwidth-bound decode. "
)


@dataclass
class Config:
    vocab_size: int = 256  # raw UTF-8 bytes (no tokenizer file needed)
    seq_len: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_q_heads: int = 4
    n_kv_heads: int = 2  # GQA: 2 Q heads share 1 KV head (same idea as Qwen)
    d_ff: int = 384  # SwiGLU hidden width


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> rescales each token vector, shape unchanged
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * inv_rms).type_as(x) * self.weight


def precompute_rope(seq_len: int, head_dim: int, device: torch.device):
    """Precompute cos/sin tables of shape [1, 1, T, head_dim // 2] for RoPE."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(pos, freqs)  # [T, head_dim // 2]
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs inside Q or K: x has shape [B, H, T, head_dim]."""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    t = x.size(2)
    c, s = cos[:, :, :t, :].type_as(x), sin[:, :, :t, :].type_as(x)
    out = torch.stack([x_even * c - x_odd * s, x_even * s + x_odd * c], dim=-1)
    return out.flatten(-2)


class CausalGQA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_q_heads == 0
        assert cfg.n_q_heads % cfg.n_kv_heads == 0
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_q_heads
        self.group_size = cfg.n_q_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        trace: list[str] | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)     # [B, H_q, T, d_h]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, T, d_h]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, T, d_h]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if trace is not None:
            trace.append(f"  Q {tuple(q.shape)}, K/V {tuple(k.shape)} (GQA {self.n_q_heads}:{self.n_kv_heads})")

        # Broadcast KV heads to match Q heads (zero-copy view via expand; FlashAttention handles GQA natively)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        # PyTorch SDPA dispatches FlashAttention on H100 (avoids materializing [B, H, T, T] in HBM)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)                  # [B, H_q, T, d_h]
        y = y.transpose(1, 2).contiguous().view(B, T, D)                             # [B, T, D]
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, T, D_ff] -> [B, T, D]
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalGQA(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        trace: list[str] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, trace=trace)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, trace: list[str] | None = None) -> torch.Tensor:
        B, T = idx.shape
        x = self.embed(idx)  # [B, T, D]
        if trace is not None:
            trace.append(f"  tokens {tuple(idx.shape)} -> embed {tuple(x.shape)}")
        cos, sin = precompute_rope(T, self.cfg.d_model // self.cfg.n_q_heads, idx.device)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, trace=trace if i == 0 else None)
        logits = self.lm_head(self.norm(x))  # [B, T, V]
        if trace is not None:
            trace.append(f"  block_out {tuple(x.shape)} -> logits {tuple(logits.shape)}")
        return logits


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    """Sample random slices; targets y are inputs x shifted left by 1 token."""
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def greedy_generate(model: TinyLM, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        window = ids[:, -model.cfg.seq_len :]
        next_id = model(window)[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 1: Minimal LLM training loop")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    device = torch.device("cuda" if use_cuda else "cpu")

    use_bf16 = (args.dtype == "bf16") or (args.dtype == "auto" and use_cuda and torch.cuda.is_bf16_supported())
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    cfg = Config()
    model = TinyLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Device: {gpu_name} | Compute dtype: {autocast_dtype} | Params: {n_params:,}")

    # Static memory bill (standard mixed-precision 16-byte rule vs FP32 master weights in this script)
    w_bf16 = n_params * 2
    g_bf16 = n_params * 2
    adam_fp32 = n_params * 12  # master weights (4) + m (4) + v (4)
    print(
        f"Static 16B/param bill: weights(BF16)={w_bf16/1e6:.2f} MB | "
        f"grads(BF16)={g_bf16/1e6:.2f} MB | AdamW(FP32)={adam_fp32/1e6:.2f} MB | "
        f"total={(w_bf16+g_bf16+adam_fp32)/1e6:.2f} MB"
    )

    data = torch.tensor(list((CORPUS * 4).encode("utf-8")), dtype=torch.long)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(args.steps + 1):
        x, y = make_batch(data, args.batch_size, cfg.seq_len, device)
        trace = [] if step == 0 else None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_bf16):
            logits = model(x, trace=trace)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        if step == 0:
            print("\nStep 0 tensor shape trace:")
            for line in trace:
                print(line)
            print(f"  initial loss={loss.item():.3f} (random guess over 256 bytes = ln(256) = {math.log(256):.3f})\n")

        if step % 50 == 0:
            print(f"step {step:3d}/{args.steps} | loss = {loss.item():.4f}")

    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    toks_per_sec = ((args.steps + 1) * args.batch_size * cfg.seq_len) / elapsed

    print(f"\nDone in {elapsed:.2f}s ({toks_per_sec:,.0f} tokens/s)")
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Peak GPU HBM allocated (weights + grads + AdamW + activations): {peak_mb:.2f} MB")

    prompt = "DeepSeek uses "
    sample = greedy_generate(model, prompt, max_new_tokens=79, device=device)
    print(f"\nSanity-check continuation:\n  prompt : {prompt!r}\n  output : {sample!r}")


if __name__ == "__main__":
    main()
