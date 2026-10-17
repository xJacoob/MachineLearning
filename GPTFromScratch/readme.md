# Mini GPT from Scratch

A character-level, decoder-only Transformer (GPT) trained on the **Tiny Shakespeare** dataset, written in PyTorch while following Andrej Karpathy's lecture *"Let's build GPT: from scratch, in code, spelled out"*.
The goal of this project is to understand every building block of a modern language model by implementing it myself, rather than to reach state-of-the-art quality.

## What's inside

| File | Description |
|---|---|
| `gpt.py` | Full model, training loop and text generation |
| `attention_head.py` | A single self-attention head re-implemented from scratch (without looking at the lecture code) as a self-check |
| `input.txt` | Tiny Shakespeare dataset (~1 MB of text) |

> Rename the files in the table if yours are named differently.

## Model architecture

```
input characters (B, T)
  → token embedding + position embedding   (B, T, C)
  → N × Transformer Block                   (B, T, C)
       ├─ x = x + MultiHeadAttention(LayerNorm(x))
       └─ x = x + FeedForward(LayerNorm(x))
  → final LayerNorm
  → linear layer to vocabulary              (B, T, vocab_size)
```

- **Tokenization:** character level, vocabulary of 65 characters.
- **Self-attention head:** query/key/value projections, scaled dot-product attention, a causal (lower-triangular) mask so a token cannot attend to future tokens, and a weighted sum of values.
- **Multi-head attention:** several heads in parallel, concatenated and passed through an output projection.
- **Feed-forward:** `Linear(C, 4C) → ReLU → Linear(4C, C)`, applied to every position independently.
- **Residual connections + pre-norm LayerNorm** in every block, dropout for regularization.
- **Generation:** autoregressive sampling — the model predicts a distribution over the next character, samples from it, appends it, and repeats.

## Hyperparameters

| Parameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` (context length) | 256 |
| `n_embd` | 384 |
| `n_head` | 6 |
| `n_layer` | 6 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 (Adam) |
| `max_iter` | 5000 |

This gives roughly 10M parameters.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). On Windows, PyTorch from PyPI is CPU-only, so `pyproject.toml` points `torch` to the CUDA index:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Then:

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run gpt.py
```

To download the dataset:

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Differences from the lecture code

While reviewing my implementation I fixed or changed a few details:

- attention scores are scaled by `head_size ** -0.5` (the dimension of the query/key vectors, as in *Attention Is All You Need*) instead of `n_embd ** -0.5`,
- the final LayerNorm (`ln_f`) is applied before the language-model head,
- dropout is applied after the output projection in multi-head attention,
- evaluation runs every `eval_interval` steps.

## What I learned

- How self-attention works at the level of matrices: I first computed it by hand on paper, then in NumPy, and finally as a PyTorch module.
- Why queries and keys are separate projections (attention weights become asymmetric, so relations can be directional).
- Why scores are scaled before the softmax (otherwise the softmax becomes too peaked and gradients vanish).
- Why a causal mask is needed for a language model.
- How residual connections and LayerNorm make deep stacks of blocks trainable.
- How a language model generates text one token at a time.

## Credits

- Andrej Karpathy — [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) repository.
- Tiny Shakespeare dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017).