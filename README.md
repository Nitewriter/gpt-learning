# GPT Learning

This repository is my personal attempt at learning how to build a GPT-like model using PyTorch. The goal is to understand the architecture and the training process of a generative model.

## References

- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Create a Large Language Model from Scratch with Python](https://www.youtube.com/watch?v=UU1WVnMk4E8&list=PLdR6MEBPfLDl-PMbBXkS_EPzBbqyhG4Zt)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## Requirements

- Python 3.10+
- Accelerator (CUDA, MPS, etc.)

## Installation (Linux/MacOS)

A makefile is include in the project that will create the virtual environment and install the dependencies.

```bash
make
```

## Installation (Windows)

A setup.ps1 script is included in the project that will create the virtual environment and install the dependencies.

```powershell
.\setup.ps1
```

> Note: Reload VSCode using the command palette `Developer: Reload Window` to allow the virtual environment to be auto-selected.