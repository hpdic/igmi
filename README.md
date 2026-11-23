# Introduction to Generative Machine Intelligence (IGMI)

Systems-first lecture notes (in LaTeX) on how modern generative AI models are trained and served.

> **Note**: This repository contains the author's personal lecture notes, shared publicly as-is. You are welcome to read and fork the repository, but the author has no obligation to respond to issues, review pull requests, or provide support.

## Philosophy

- **Real systems first, no intro chapter**: start directly with how a Large Language Model (LLM) trains and executes on hardware.
- **Lecture-note density**: bullet points structured like technical slides, without textbook filler.
- **Hardware accounting over math formulas**: explain mechanisms via tensor shapes (`[B, T, D]`), memory bytes, arithmetic intensity, and collective communication rather than derivations.
- **Minimal runnable examples**: every chapter pairs notes with standalone PyTorch scripts that run out-of-the-box on both GPU (default: NVIDIA H100) and CPU.
- **Running examples**:
  - **DeepSeek-V3 and DeepSeek-R1** (primary): Multi-head Latent Attention (MLA), Mixture-of-Experts (MoE), FP8 training, DualPipe, Multi-Token Prediction (MTP), and Group Relative Policy Optimization (GRPO).
  - **Qwen2.5 and Qwen3** (dense baseline): Grouped-Query Attention (GQA) reference architecture for contrast.

## Contents

| Chapter | Topic | Lecture Notes | Lab Guide | Code | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | LLM Training Systems | [01-training/chap01.tex](01-training/chap01.tex) | [01-training/lab01.tex](01-training/lab01.tex) | [tiny_train.py](01-training/code/tiny_train.py) | Draft |
| **02** | Autoregressive Inference: Prefill and Decode | `02-autoregressive-inference/chap02.tex` | `02-autoregressive-inference/lab02.tex` | Planned | Planned |
| **03** | Diffusion-Based LLMs | `03-diffusion-llm/chap03.tex` | `03-diffusion-llm/lab03.tex` | Planned | Planned |

*Lecture notes compile via [igmi.tex](igmi.tex) into [igmi.pdf](igmi.pdf); hands-on lab guides and H100/CPU benchmarks compile via [igmi-labs.tex](igmi-labs.tex) into [igmi-labs.pdf](igmi-labs.pdf).*

## Build PDFs

Requires TeX Live (`pdflatex` and `latexmk`).

```bash
make          # builds both igmi.pdf (lecture notes) and igmi-labs.pdf (lab manual)
make notes    # builds igmi.pdf only
make labs     # builds igmi-labs.pdf only
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
