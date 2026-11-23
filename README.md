# Introduction to Generative Machine Intelligence (IGMI)

Systems-first lecture notes on how modern generative AI models are trained and served.

## Philosophy

- **Real systems first, no intro chapter**: start directly with how an LLM trains and executes on hardware.
- **Lecture-note density**: bullet points only — structured like technical slides, zero textbook fluff.
- **Hardware accounting over math formulas**: explain mechanisms via tensor shapes (`[B, T, D]`), memory bytes, arithmetic intensity, and collective communication rather than derivations.
- **Minimal runnable examples**: every chapter pairs notes with standalone PyTorch scripts that run out-of-the-box on both GPU (default: NVIDIA H100) and CPU.
- **Running examples**:
  - **DeepSeek-V3 / R1** (primary): MLA, fine-grained MoE, FP8 training, DualPipe, MTP, GRPO.
  - **Qwen2.5 / Qwen3** (dense/GQA baseline): clean reference architecture for contrast.

## Contents

| Chapter | Topic | Notes | Code | Status |
| :--- | :--- | :--- | :--- | :--- |
| **01** | LLM Training Systems | [01-training/notes.md](01-training/notes.md) | [tiny_train.py](01-training/code/tiny_train.py) | Draft |
| **02** | Autoregressive LLM Inference (Prefill & Decode) | `02-autoregressive-inference/notes.md` | — | Planned |
| **03** | Diffusion-Based LLMs | `03-diffusion-llm/notes.md` | — | Planned |

*Chapters are modular by design and will later split into dedicated sections (e.g., Encoder, Prefill, Decode, Multimodal).*

## Quick Start

Requires Python 3.10+ and PyTorch (`torch`).

```bash
# Default: auto-selects CUDA GPU (tested on NVIDIA H100) with BF16 autocast
python 01-training/code/tiny_train.py

# Force CPU execution (FP32)
python 01-training/code/tiny_train.py --device cpu
```
