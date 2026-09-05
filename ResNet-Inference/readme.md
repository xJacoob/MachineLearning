# ResNet18 Inference & Feature Map Visualization

Using a pretrained ResNet18 (torchvision) to run inference on custom photos and 
visualize what the first convolutional layer actually detects, using PyTorch 
forward hooks.

## What's implemented

- Loading ResNet18 with ImageNet-pretrained weights (`ResNet18_Weights.DEFAULT`)
- Correct preprocessing via `weights.transforms()` — resize, center crop, and 
  ImageNet-specific normalization, matching exactly what the model was trained on
- Inference on a personal photo, with the predicted class index mapped to a 
  human-readable label via `weights.meta["categories"]`
- A forward hook registered on `model.conv1` to capture intermediate activations — 
  the only way to inspect a layer's output, since a normal forward pass only 
  returns the final classification result
- Visualization of the first 16 (of 64) feature maps as a grid, using matplotlib

## Results

A photo of a dachshund was classified as **"Doberman, Doberman pinscher"** (class 
236) — a confident but visually implausible prediction, given how different the two 
breeds look. This is a useful reminder that a correct-looking pipeline can still 
produce wrong classifications, and that checking predictions against real photos 
matters more than trusting output shapes alone.

### conv1 feature maps

`model.conv1` outputs a tensor of shape `[1, 64, 112, 112]` — 64 channels because 
ResNet18's first layer has 64 filters; 112×112 because `conv1` uses `stride=2`, 
halving the 224×224 input immediately (unlike the custom CNN in `FashionMNIST-CNN`, 
where downsampling was handled separately by `MaxPool2d`).

Visually inspecting the grid shows three distinct filter behaviors:
- **Edge detectors** — several maps clearly highlight the dog's outline against the background
- **Near-identity maps** — a few closely resemble the grayscale original, picking up general brightness/contrast rather than a specific pattern
- **Near-blank maps** — some filters simply didn't find whatever pattern they're tuned to detect in this particular photo

No single filter "recognizes a dog" — each one responds to a simple, local pattern. 
Classification only emerges after many more layers combine these signals.

## Tech stack

- PyTorch, torchvision (`resnet18`, forward hooks)
- PIL (image loading)
- matplotlib (feature map visualization)