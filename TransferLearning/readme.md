# Transfer Learning vs. Training from Scratch (ResNet18)

Fine-tuning a pretrained ResNet18 on a small dataset (ants vs. bees) by freezing 
the backbone and retraining only the classification head — compared directly 
against the same architecture trained from scratch on the same data.

## Dataset

[`hymenoptera_data`](https://download.pytorch.org/tutorial/hymenoptera_data.zip) 
(official PyTorch transfer learning tutorial dataset) — 124 training / 75 
validation images of ants, 121 training / 75 validation images of bees. Deliberately 
tiny, to make the case for transfer learning obvious.

- **Train transforms**: `RandomResizedCrop(224)`, `RandomHorizontalFlip()` — data 
  augmentation to squeeze more variety out of a small dataset
- **Val transforms**: `Resize(256)` + `CenterCrop(224)` — deterministic, no augmentation
- Both normalized with ImageNet's mean/std (`[0.485, 0.456, 0.406]` / 
  `[0.229, 0.224, 0.225]`), kept identical across both experiments to isolate 
  pretrained-vs-scratch as the only variable

## Approach 1: Frozen backbone + fine-tuned head

```python
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 2)  # new head, trainable by default
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

Setting `requires_grad = False` on every parameter stops gradients from being 
computed for the entire backbone — those weights stay exactly as trained on 
ImageNet. Replacing `model.fc` swaps in a fresh `Linear` layer (trainable by 
default, since it never went through the freezing loop), so only ~1,026 parameters 
(512×2 weights + 2 biases) actually get updated during training.

## Approach 2: Same architecture, trained from scratch

Identical architecture, identical data and transforms, identical training loop — 
the only differences: `weights=None` (random initialization instead of pretrained) 
and no freezing, so all ~11M parameters train from the first epoch.

## Results

| Approach | Final accuracy | Behavior |
|---|---|---|
| Frozen backbone + fine-tuned head | **~94–95%** | High accuracy from epoch 1, stable in the 88–94% range |
| Trained from scratch | **~67%** | Slow, noisy convergence (55–75% range), loss plateaus around 0.6 |

**~27–28 percentage point gap**, entirely attributable to starting from pretrained 
weights vs. random initialization on the same ~245-image dataset.

This matches the CS231n notes directly: *"very few people train an entire 
Convolutional Network from scratch... because it is relatively rare to have a 
dataset of sufficient size."* The frozen-backbone model only had to learn how to 
combine already-useful, general-purpose features (edges, textures — the same kind 
seen in [ResNet-Inference](../ResNet-Inference)); the from-scratch model had to 
discover those features itself from 245 images, which isn't enough data to do so reliably.

## Bug found along the way

Early in writing the from-scratch training function, the forward pass and 
`model.train()`/`model.eval()` calls still referenced the old pretrained `model` 
variable instead of the newly created `model_scratch` — meaning the optimizer was 
correctly attached to `model_scratch`'s parameters, but training and evaluation 
were silently running on a completely different model. Caught by checking every 
reference to `model` inside the function, not just the ones near the top.

## Tech stack

- PyTorch, torchvision (`resnet18`, `ImageFolder`)
- Weights & Biases (experiment tracking)