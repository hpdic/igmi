# Introduction to Generative Machine Intelligence (IGMI)

Systems-first lecture notes (in LaTeX) on how modern generative AI models are trained and served.

> **Note**: This repository contains the author's personal lecture notes, shared publicly as-is. You are welcome to read and fork it, but the author is under no obligation to respond to issues, review pull requests, or provide support.

## Philosophy

- **Real systems first, no intro chapter**: start directly with how an LLM trains and executes on hardware.
- **Lecture-note density**: bullet points only — structured like technical slides, zero textbook fluff.
- **Hardware accounting over math formulas**: explain mechanisms via tensor shapes (`[B, T, D]`), memory bytes, arithmetic intensity, and collective communication rather than derivations.
- **Minimal runnable examples**: every chapter pairs notes with standalone PyTorch scripts that run out-of-the-box on both GPU (default: NVIDIA H100) and CPU.
- **Running examples**:
  - **DeepSeek-V3 / R1** (primary): MLA, fine-grained MoE, FP8 training, DualPipe, MTP, GRPO.
  - **Qwen2.5 / Qwen3** (dense/GQA baseline): clean reference architecture for contrast.

## Contents

| Chapter | Topic | LaTeX Notes | Code | Status |
| :--- | :--- | :--- | :--- | :--- |
| **01** | LLM Training Systems | [01-training/notes.tex](01-training/notes.tex) | [tiny_train.py](01-training/code/tiny_train.py) | Draft |
| **02** | Autoregressive LLM Inference (Prefill & Decode) | `02-autoregressive-inference/notes.tex` | — | Planned |
| **03** | Diffusion-Based LLMs | `03-diffusion-llm/notes.tex` | — | Planned |

*Chapters are modular (`\input{...}` in [main.tex](main.tex)) and will later split into dedicated sections (e.g., Encoder, Prefill, Decode, Multimodal).*

## Build Lecture Notes (PDF)

Requires TeX Live (`pdflatex` + `latexmk`).

```bash
make          # builds igmi.pdf
make clean    # removes auxiliary LaTeX files
```

## Run Toy Examples

Requires Python 3.10+ and PyTorch (`torch`).

```bash
# Default: auto-selects CUDA GPU (tested on NVIDIA H100) with BF16 autocast
python 01-training/code/tiny_train.py

# Force CPU execution (FP32)
python 01-training/code/tiny_train.py --device cpu
```
