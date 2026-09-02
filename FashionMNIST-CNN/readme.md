# FashionMNIST CNN — Convolutional Architecture vs. MLP Baseline

A convolutional neural network trained on FashionMNIST, built to directly compare against 
the [FashionMNIST-MLP](../FashionMNIST-MLP) baseline in the same W&B project — same 
dataset, same evaluation protocol, different architecture.

## What's implemented

- A CNN (`nn.Module`) with two convolutional blocks (`Conv2d → ReLU → MaxPool2d`) followed 
  by a small MLP head (`Flatten → Linear → ReLU → Linear`)
- Spatial dimensions and parameter count worked out by hand before writing any code 
  (see "Dimension math" below)
- Learning rate comparison across three values, logged to the **same W&B project** as the 
  MLP baseline (`fashion_mnist`) to enable direct side-by-side comparison

## Architecture
 - Input: 1×28×28
 - Conv2d(1→32, k=3, s=1, p=1) + ReLU → 32×28×28 (same padding, spatial size preserved)
 - MaxPool2d(k=2, s=2) → 32×14×14
 - Conv2d(32→64, k=3, s=1, p=1) + ReLU → 64×14×14
 - MaxPool2d(k=2, s=2) → 64×7×7
 - Flatten → 3136
 - Linear(3136→128) + ReLU
 - Linear(128→10) → raw logits (10 classes)


## Dimension math

Worked out manually using `output_size = (input − kernel + 2×padding) / stride + 1` 
before implementation, to avoid the classic "wrong `input_dim`" bug:

- Conv layers use `padding=1` with a 3×3 kernel specifically to preserve spatial size 
  (`P = (F-1)/2` for "same" convolution) — all size reduction is deliberately delegated 
  to the pooling layers, not the conv layers themselves
- Each `MaxPool2d(kernel_size=2, stride=2)` halves both spatial dimensions: 28→14→7
- Final flattened size: `64 channels × 7 × 7 = 3136`, matching `Linear(in_features=64*7*7, ...)`

## Results

Three learning rates compared (Adam, 20 epochs, batch_size=64), logged alongside the 
MLP baseline in the same W&B project:

| Learning rate | Test accuracy | Behavior |
|---|---|---|
| 0.001 | **~91.5%** | Smooth convergence, best result |
| 0.01 | ~88–89% | Noisier, still learning, less stable |
| 0.1 | **10.0%** | Network died after the first step |

**CNN (91.5%) outperforms the MLP baseline (83.25%) by ~8 percentage points** on the 
same dataset — consistent with the theoretical expectation that convolution preserves 
spatial structure that flattening destroys.

### Why lr=0.1 collapsed to exactly 10%

Since FashionMNIST has 10 balanced classes (1,000 test samples each), 
0.1 accuracy with zero variance is the signature of a model that always predicts a 
single fixed class. Most likely cause: the first update step was large enough to push 
every ReLU unit into permanent zero-gradient territory ("dying ReLU" at the scale of 
the whole network) — after that, no gradient flows and the weights never update again.

## Bugs found along the way

- **Wrong W&B project name**: initially logged to a separate project 
  (`FashionMNIST-CNN`) instead of the shared `fashion_mnist` project used by the MLP 
  run, which would have made direct comparison impossible. Fixed by matching the 
  project name.
- **Hardcoded run name in a loop over 3 learning rates** — same class of mistake made 
  (and fixed) twice before in earlier projects. Fixed with `name=f'cnn-lr{lr}'`.
- **Curly braces around values in the `config` dict** (`{'lr': {lr}, ...}`) — this is 
  Python *set* syntax, not variable interpolation, so W&B was about to log single-element 
  sets instead of plain numbers. Fixed by removing the braces.

## Tech stack

- PyTorch, torchvision
- Weights & Biases (experiment tracking, shared project with FashionMNIST-MLP)

## Status

✅ Done — CNN beats MLP baseline (91.5% vs. 83.25%), with a documented explanation for 
why the highest tested learning rate failed outright.