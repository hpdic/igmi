# Chapter 1: LLM Training Systems

- **Goal**: understand the exact execution loop, memory bill, precision tricks, and multi-GPU partitioning behind training a modern LLM.
- **Running examples**:
  - **Qwen2.5-72B** (clean dense + GQA baseline).
  - **DeepSeek-V3** (671B total / 37B active MoE + MLA + FP8 + DualPipe).
- **Runnable companion**: [code/tiny_train.py](code/tiny_train.py) — standalone ~280-line PyTorch trainer using the exact same block structure, runnable on H100 or CPU.

---

## 1. The Core Loop: One Objective, Four Phases

- **Input**: text converted by a tokenizer into integer IDs of shape `[B, T]`.
  - `B` = batch size (number of sequences processed together).
  - `T` = sequence length (tokens per sequence, e.g., `4096`–`131072`).
  - Vocabulary size `V`: `152,064` (Qwen2.5), `129,280` (DeepSeek-V3), `256` (our byte-level toy model).
- **Self-supervised task (next-token prediction)**:
  - Model input `x = tokens[:, :-1]` (positions `0 .. T-2`).
  - Ground-truth target `y = tokens[:, 1:]` (positions `1 .. T-1`).
  - Every sequence of length `T` yields `T - 1` supervised predictions simultaneously.
- **One training step**:
  1. **Forward**: map `[B, T]` token IDs → `[B, T, V]` unnormalized scores (logits). Save intermediate activations needed for backprop.
  2. **Loss**: compare predicted distribution at every `(b, t)` against true next token `y[b, t]` (cross-entropy → 1 scalar).
  3. **Backward**: walk the graph in reverse to compute one gradient tensor per weight tensor.
  4. **Optimizer step (AdamW)**: update every weight using its gradient and running statistics (`m`, `v`), then zero gradients.
- **The single most important systems property of training**:
  - All `T` input tokens are known upfront.
  - Every linear projection multiplies a `[B * T, D_in]` matrix by a `[D_in, D_out]` weight matrix → huge dense GEMMs.
  - Result: training is overwhelmingly **compute-bound** (Tensor Cores stay busy if communication and memory don't stall them).
  - *Contrast with Chapter 2*: autoregressive decoding generates 1 token at a time (`T = 1`) → tiny GEMVs → **memory-bandwidth-bound**.

---

## 2. Inside One Transformer Block (Tensor Shapes Only)

Modern LLMs (Qwen, DeepSeek, Llama) share the same pre-norm decoder-only skeleton:

```text
tokens [B, T]
  └─> Token Embedding ─────────────────────────────> x [B, T, D]
        └─> Repeat L layers:
              ├─> RMSNorm ─> Attention (token mixing) ──> + residual
              └─> RMSNorm ─> FFN / MoE (channel mixing) ─> + residual
        └─> Final RMSNorm ─> LM Head ──────────────> logits [B, T, V]
```

- **RMSNorm**: rescales each token's `D`-element vector to unit root-mean-square; shape stays `[B, T, D]`. Cheaper than LayerNorm (no mean subtraction, fewer memory reads).
- **Attention (how tokens talk to earlier tokens)**:
  - Linear projections create `Q` (`[B, H_q, T, d_h]`), `K` (`[B, H_kv, T, d_h]`), `V` (`[B, H_kv, T, d_h]`).
  - **RoPE** injects position by rotating pairs of numbers inside `Q` and `K` according to token index `t`; no extra embedding parameters.
  - **Causal mask**: token `t` may only read `K`/`V` from positions `0..t` (blocks peeking at the answer `t+1`).
  - Output projection merges all heads back to `[B, T, D]`.
  - **Three attention flavors (directly determines Chapter 2's KV cache size)**:
    - **MHA**: `H_kv = H_q` (largest KV state).
    - **GQA (Qwen2.5)**: `H_kv < H_q` (several Q heads share one KV head).
    - **MLA (DeepSeek-V3)**: compress K and V jointly into a tiny latent vector (`d_c = 512`) + decoupled RoPE key (`d_r = 64`), decompress on the fly during attention.
- **FFN / MoE (where most parameters live; tokens processed independently)**:
  - **Dense SwiGLU (Qwen2.5)**:
    - Two parallel up-projections (`gate` and `up`): `[B, T, D]` → `[B, T, D_ff]`.
    - Elementwise `silu(gate) * up`, then down-projection `[B, T, D_ff]` → `[B, T, D]`.
    - Every token touches 100% of FFN weights.
  - **Sparse DeepSeekMoE (DeepSeek-V3, layers 4–61)**:
    - Lightweight router projects each token `[D]` → scores over `256` routed experts, picks top `8`.
    - Token also always passes through `1` shared expert (captures common knowledge).
    - Each expert is a small SwiGLU (`D_ff = 2048`) → only `37B / 671B` (**5.5%**) of parameters execute per token.

### Concrete Dimension Comparison

| Spec | `tiny_train.py` | Qwen2.5-72B (Dense) | DeepSeek-V3 (MoE) |
| :--- | :--- | :--- | :--- |
| Total / Active params | 459K / 459K | 72.7B / 72.7B | 671B / 37B |
| Layers (`L`) | 2 | 80 | 61 (3 dense + 58 MoE) |
| Hidden dim (`D`) | 128 | 8,192 | 7,168 |
| Attention type | GQA (4 Q / 2 KV) | GQA (64 Q / 8 KV) | MLA (128 heads, latent 512+64) |
| FFN / Experts | SwiGLU (`D_ff=384`) | SwiGLU (`D_ff=29,568`) | 1 shared + 256 routed (top-8, `D_ff=2,048`) |
| Vocab (`V`) | 256 (UTF-8 bytes) | 152,064 | 129,280 |

---

## 3. Toy Example Walkthrough (`code/tiny_train.py`)

- **What it runs**:
  - Byte-level tokenizer (`text.encode("utf-8")` → integers `0..255`), zero external tokenizer files.
  - 2-layer pre-norm Transformer with `RMSNorm`, `RoPE`, `GQA` causal attention (`F.scaled_dot_product_attention`), and `SwiGLU`.
  - Runs 250 AdamW steps on an embedded systems text corpus (~3s on H100, ~16s on CPU).
- **What to watch in the stdout**:
  1. **Tensor shape trace** on step 0: verify `[B, T]` → `[B, T, D]` → `[B, H, T, d_h]` → `[B, T, V]` matches Section 2.
  2. **Static memory breakdown**: notice AdamW states alone are **4× larger** than BF16 weights.
  3. **Loss drop**: starts at `~5.55` (uniform random guess over 256 bytes) and falls below `0.2`, then prints a short greedy continuation to confirm the loop actually learned.
- **Run commands**:
  - H100 GPU (default, BF16 autocast): `python 01-training/code/tiny_train.py`
  - CPU (FP32): `python 01-training/code/tiny_train.py --device cpu`

---

## 4. The Memory Bill: Why a Single H100 Runs Out of HBM Fast

GPU HBM during training is split into **static state** (fixed by parameter count `P`) and **dynamic activations** (scales with batch & sequence length).

### 4.1 Static State: The 16-Byte Rule (Standard Mixed Precision + AdamW)

For a model with `P` parameters (BF16 compute, FP32 master weights + AdamW):

| Component | Precision | Bytes per Param | Why it exists |
| :--- | :--- | :--- | :--- |
| Model weights | BF16 | `2` | Used in forward/backward GEMMs |
| Gradients | BF16 | `2` | Output of backward pass |
| Master weights | FP32 | `4` | Small weight updates underflow in BF16 |
| AdamW 1st moment (`m`) | FP32 | `4` | Running average of gradients |
| AdamW 2nd moment (`v`) | FP32 | `4` | Running average of squared gradients |
| **Total static floor** | — | **`16 bytes × P`** | **Before batch size = 1** |

- **Reality check against 1× NVIDIA H100 (94 GB HBM)**:
  - **7B dense model**: `16 × 7B = 112 GB` → already **OOMs a single 94 GB H100** with zero activations.
  - **Qwen2.5-72B**: `16 × 72.7B ≈ 1.16 TB` → needs at least ~13–16 H100s just to hold static state.
  - **DeepSeek-V3 (671B)**: `16 × 671B ≈ 10.7 TB` → needs >120 H100s just to hold static state.
  - *Takeaway*: **optimizer states (12 bytes/param), not weights (2 bytes/param), dominate training memory.**

### 4.2 Dynamic Activations (The Hidden Memory Spike)

- Backward pass needs intermediate tensors from forward (inputs to every linear layer, norm, and activation).
- Memory scales roughly with `B × T × L × D` — explodes rapidly at long context (`T = 32K–128K`).
- **Two mandatory fixes in every real LLM trainer**:
  1. **FlashAttention**:
     - Naive attention writes the `[B, H, T, T]` score matrix to HBM (`O(T^2)` memory + slow HBM round-trips).
     - FlashAttention tiles `Q, K, V` into on-chip SRAM (~20 MB on H100), computes exact attention in one fused kernel, and never materializes `[T, T]` in HBM → `O(T)` memory and 2–4× faster.
  2. **Activation Checkpointing (Recomputation)**:
     - Drop intermediate activations inside each layer during forward; keep only layer boundaries.
     - Re-run forward for one layer at a time right before its backward pass.
     - Trade-off: **~33% extra compute** (one extra forward pass) for **5–10× less activation memory**.

---

## 5. Precision Engineering: BF16 vs DeepSeek-V3's FP8

- **Why lower precision matters on H100** (dense Tensor Core peak, no sparsity):
  - `FP32`: ~67 TFLOPS
  - `BF16`: ~989 TFLOPS (**~15× faster**, half the bytes)
  - `FP8`: ~1,979 TFLOPS (**~30× faster**, quarter of the bytes)
- **Standard recipe (Qwen2.5)**:
  - Forward & backward GEMMs in **BF16** (same exponent range as FP32 → rarely overflows, no loss scaling needed unlike FP16).
  - Master weights, optimizer states, and gradient accumulation in **FP32**.
- **DeepSeek-V3 FP8 mixed-precision recipe**:
  - Runs almost all GEMMs in **FP8 (`E4M3`)**, accumulating matrix products in FP32 inside Tensor Cores.
  - **Why naive FP8 breaks**: a few large outlier values in an activation tensor force a huge scale factor, crushing normal values to zero.
  - **DeepSeek's fix — fine-grained block scaling**:
    - Activations: separate scale factor per `1 × 128` tile (1 token × 128 channels).
    - Weights: separate scale factor per `128 × 128` block.
    - Sensitive ops kept in BF16/FP32: embeddings, LM head, MoE gating router, RMSNorm, attention operators.
    - Optimizer states (`m`, `v`) stored in **BF16** instead of FP32 (cuts static state memory further), while master weights stay FP32.

---

## 6. Scaling Beyond One GPU: The 5 Parallelism Axes

Every distributed training system combines up to 5 partitioning dimensions. Think in terms of **what is split**, **what is sent**, and **which wire it needs**.

| Axis | What is split | Collective / Wire | Bottleneck & Typical Scope |
| :--- | :--- | :--- | :--- |
| **DP** (Data) | Batch `B` | `AllReduce` grads (IB / Ethernet) | Replicates full `16P` state on every GPU; doesn't solve OOM alone |
| **ZeRO / FSDP** | Batch `B` + sharded optimizer (Z1), grads (Z2), weights (Z3) | `AllGather` weights + `ReduceScatter` grads (IB) | Cuts static memory by `N_dp`; Z1 is cheap, Z3 adds weight traffic |
| **TP** (Tensor) | Columns/rows of each weight matrix (`D`, `D_ff`, heads) | 2× `AllReduce` **per layer** (NVLink) | Strictly **intra-node** (8 GPUs); blocks compute if interconnect is slow |
| **PP** (Pipeline) | Layers `L` assigned to stages | Point-to-point `Send/Recv` activations (IB) | Tiny traffic, cross-node friendly; suffers from **pipeline bubbles** |
| **EP** (Expert, MoE) | Different experts on different GPUs | 2× `All-to-All` per MoE layer (dispatch + combine) | Cross-node bandwidth + **straggler GPUs** if routing is unbalanced |

- **Rule of thumb for hardware hierarchy**:
  - **Fastest wire (NVLink, 900 GB/s on H100 SXM)**: put **TP** (dense) or **local EP** here.
  - **Slower wire (InfiniBand, ~50 GB/s per GPU)**: put **PP**, **DP / ZeRO-1**, and overlapped **EP** here.
- **Context Parallelism (CP)** (for ultra-long `T = 64K–128K` training):
  - Splits sequence length `T` across GPUs; passes `K`/`V` blocks around a ring (Ring Attention) so no single GPU holds the full sequence activations.

---

## 7. Real Systems Side-by-Side: Qwen2.5-72B vs DeepSeek-V3

### Qwen2.5-72B (Classic Dense 3D Parallelism)
- **Layout**: `TP (intra-node) + PP (inter-node) + ZeRO-1 DP`.
- **Execution profile**: every token executes the exact same dense GEMMs → easy load balance, high Tensor Core MFU, predictable communication.
- **Main cost**: total FLOPs per token scales linearly with total parameters (`~6 × P` FLOPs per training token → `~436 GFLOPs/token`).

### DeepSeek-V3 (671B MoE on 2,048 H800 GPUs — 14.8T tokens in ~2.79M GPU-hours)
- **Layout**: **16-way PP + 64-way EP (spanning 8 nodes) + ZeRO-1 DP**, and **`TP = 1` (zero Tensor Parallelism)**.
- **Why drop TP completely?**
  - EP already saturates cross-node communication with `All-to-All`.
  - Removing TP eliminates blocking per-layer `AllReduce` overhead and keeps each GPU's local GEMMs simpler.
- **DualPipe (hiding communication behind compute)**:
  - Feeds micro-batches from **both ends** of the pipeline simultaneously.
  - Within every pipeline step, overlaps one micro-batch's **computation** (Attention + MoE GEMMs) with another micro-batch's **communication** (`All-to-All` dispatch/combine + PP `Send/Recv`).
  - Even on bandwidth-capped H800s (~50 GB/s IB), communication overhead drops near zero.
- **Auxiliary-loss-free MoE load balancing**:
  - If too many tokens pick the same expert, that GPU becomes a straggler and stalls the entire cluster.
  - Classic fix: add a load-balancing loss term (hurts language modeling quality).
  - DeepSeek's fix: add a tiny dynamic **bias** to each expert's routing score (updated after each step based on overload/underload) — balances GPU load without distorting gradients.
- **Multi-Token Prediction (MTP)**:
  - Adds 1 extra lightweight prediction module during training so each position predicts both `t+1` and `t+2`.
  - Costs minimal extra training compute, improves representation quality, and doubles as a **free draft model for speculative decoding** at inference time.

---

## 8. How Training Decisions Lock In Inference Systems (Bridge to Chapter 2)

Training ends with a static checkpoint (just the `2P` or `1P` weights — gradients and AdamW states are thrown away). Every architectural choice made above permanently dictates how fast and cheap inference can be:

- **Attention design → KV cache capacity wall**:
  - Qwen2.5-72B (GQA, 8 KV heads, 80 layers): `2 × 8 × 128 × 80 × 2 bytes` = **320 KB / token** (BF16).
  - DeepSeek-V3 (MLA, latent `512 + 64`, 61 layers): `576 × 61 × 2 bytes` = **~70 KB / token** (BF16).
  - Despite having **9.2× more total parameters**, DeepSeek-V3 uses **4.5× less KV cache per token** than Qwen2.5-72B.
- **Dense vs MoE → compute vs memory footprint**:
  - Dense 72B: ~145 GB weights (BF16) → fits on 2× H100, burns compute proportional to 72B per token.
  - MoE 671B: ~671 GB weights (FP8) → requires an entire 8× H100 node just to hold weights in HBM, even though only 37B active params compute per token.
- **Post-Training (SFT & RL / GRPO) — where training turns into inference**:
  - **SFT**: same training loop as Section 1, on curated instruction/reasoning data; loss masked out on prompt tokens.
  - **RL (DeepSeek-R1's GRPO)**:
    - For each prompt, generate `G` full responses (e.g., `G = 16`, thousands of reasoning tokens each).
    - Score responses with deterministic verifiers (math answers, unit tests); normalize rewards within the group → **no critic/value model needed** (saves ~50% memory vs PPO).
    - Update policy weights with a standard forward/backward step.
  - **Systems punchline**: in RL post-training, **~70–85% of wall-clock GPU time is spent generating rollouts** — i.e., running an autoregressive LLM inference engine at massive scale.

→ Next: **Chapter 2: Autoregressive LLM Inference (Prefill & Decode)**.
