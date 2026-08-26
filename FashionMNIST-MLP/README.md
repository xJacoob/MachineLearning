# FashionMNIST MLP — Custom Training Pipeline

A simple multilayer perceptron (MLP) trained from scratch on FashionMNIST using PyTorch, 
built to practice the full deep learning training pipeline without relying on high-level 
frameworks like PyTorch Lightning.

## What's implemented

- Data loading via `torchvision.datasets.FashionMNIST`, wrapped in a `DataLoader` for batching and shuffling
- A custom MLP defined as an `nn.Module` (784 → 128 → 128 → 128 → 10, ReLU activations, dropout, raw logits output)
- A full training loop written from scratch: `forward → loss → backward → step → zero_grad`
- Model evaluation on the test set (accuracy computed under `torch.no_grad()`)
- Experiment tracking with [Weights & Biases](https://wandb.ai) — logged loss and accuracy per epoch
- A standalone, reproducible training script (`train.py`) with CLI arguments

## Project structure

- `fashion_mnist.ipynb` — exploration and experiments (learning rate comparison, overfitting/regularization study)
- `train.py` — clean, reproducible training script for reruns

## How to run

```bash
python train.py --lr 0.1 --epochs 10 --batch_size 64
```

All arguments are optional and default to the values above if omitted.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lr` | float | 0.1 | Learning rate for SGD |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size for both train and test DataLoaders |

Each run is automatically logged to Weights & Biases under the `fashion_mnist` project.

## Results

Final test accuracy: **83.25%** (SGD, lr=0.1, 10 epochs)

### Learning rate comparison

Three learning rates were compared using W&B (`lr=0.1`, `lr=0.01`, `lr=0.001`):

![Learning rate comparison](./lr-comparasion.png)

`lr=0.1` converged fastest and reached the highest accuracy without oscillating. 
`lr=0.001` was too slow to converge within 10 epochs — its loss curve was still 
decreasing when training stopped.

### Overfitting & regularization experiment

To observe overfitting directly, the model was trained on a small subset of 1,000 
training samples for 50 epochs (SGD, lr=0.1), logging both train and test metrics 
per epoch. Two runs were compared:

- **overfit-baseline** — no regularization
- **regularized** — dropout (p=0.3) after each hidden layer, weight decay (1e-4) on the optimizer

![Overfitting and regularization comparison](./overfitting.png)

Both runs show a persistent train/test accuracy gap of roughly 15 points, so 
regularization did not meaningfully close the generalization gap in this run. 
What it did change was training stability: the baseline run showed a sharp spike 
in test loss and a matching accuracy drop in the final epochs, while the 
regularized run stayed stable through the same region.

**Takeaway:** dropout and weight decay here acted more as a stabilizer against 
training instability than as a fix for overfitting itself — a useful distinction, 
since the two effects are often assumed to be the same thing.

## Tech stack

- PyTorch, torchvision
- Weights & Biases (experiment tracking)
- argparse (CLI configuration)