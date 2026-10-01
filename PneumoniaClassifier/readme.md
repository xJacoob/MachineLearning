# Chest X-Ray Pneumonia Classifier

A transfer-learning image classifier detecting pneumonia from chest X-rays, built on 
a frozen, ImageNet-pretrained ResNet18. Includes a full experiment log (augmentation, 
learning rate scheduling, decision threshold), quantitative error analysis, and three ways to 
run inference on new images: a standalone CLI tool, a REST API (FastAPI), 
and a web demo (Gradio).

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
(Paul Mooney, Kaggle) — 5,863 X-ray images, 2 classes (`NORMAL`, `PNEUMONIA`).

| Split | NORMAL | PNEUMONIA |
|---|---|---|
| Train | 1,341 | 3,875 |
| Test | 234 | 390 |
| Val (original) | 8 | 8 |

The dataset's own `val` split (16 images total) was too small to give a stable signal 
between epochs — a single misclassification shifted accuracy by ~6 points. It was 
merged into `train`, and a new, stratified 80/20 train/val split was carved out instead 
(`train_test_split(..., stratify=targets, random_state=42)`), preserving the original 
class imbalance (~1:2.9, NORMAL:PNEUMONIA) in both halves. `test` was left untouched, 
reserved for final evaluation only.

**Note:** despite being grayscale radiographs, the source JPEGs are stored as 3-channel 
RGB (R=G=B per pixel) — a common artifact of exporting single-channel DICOM images to 
a standard image format. This is convenient here, since it matches the 3-channel input 
ResNet18 expects.

## Preprocessing

Two separate transform pipelines, applied via two `ImageFolder` instances pointing at 
the same directory (so training and validation subsets can share the same stratified 
indices while using different transforms):

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Deliberately **excludes `RandomHorizontalFlip`** — chest anatomy is not left-right 
symmetric (heart position), so mirroring would produce anatomically implausible images. 
`RandomResizedCrop`'s `scale` was narrowed from the default `(0.08, 1.0)` to `(0.8, 1.0)` 
to reduce the risk of cropping out a diagnostically relevant region of the lung field.

## Model & training

ResNet18 (ImageNet-pretrained), backbone fully frozen (`requires_grad=False` on all 
parameters), classification head replaced and trained from scratch:

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 2)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

Only the new head (~1,026 parameters) is trained — the backbone's ImageNet features 
(edges, textures) are reused as-is.

## Experiments

All runs logged to Weights & Biases (`Pneumonia-Classifier` project), Adam, lr=0.001, 
batch_size=64, decision threshold 0.3 unless noted.

| Variant | Epochs | Recall | Precision | F1 | Accuracy | Loss |
|---|---|---|---|---|---|---|
| Augmentation ON | 20 | 0.990 | 0.919 | 0.954 | 0.928 | 0.147 |
| **Augmentation OFF** | 20 | 0.990 | **0.969** | **0.979** | **0.968** | **0.092** |
| Aug OFF + CosineAnnealingLR | 10 | 0.991 | 0.955 | 0.973 | 0.959 | 0.122 |

**Augmentation OFF is the best-performing and final model.** This is a counterintuitive 
result — augmentation is usually expected to help. Likely explanation: with a frozen 
backbone and a ~1,026-parameter head, overfitting risk is inherently low, so augmentation 
has little to protect against, while the transforms (rotation, crop, brightness/contrast 
jitter) may distort subtle diagnostic texture that matters in radiographs more than in 
natural images.

The scheduler run is not directly comparable to the other two — it also used half the 
epochs (10 vs. 20), so the effect of the scheduler itself is confounded with reduced 
training time. Treated as inconclusive, not as evidence against LR scheduling.

## Decision threshold

Evaluated on the best model (aug OFF, lr=0.001, 20 epochs), comparing the default 
sigmoid threshold (0.5) against a lowered one (0.3), chosen to favor recall — in a 
screening context, a missed case of pneumonia (false negative) is a costlier error 
than an unnecessary follow-up (false positive).

| Threshold | Recall | Precision | FN | FP |
|---|---|---|---|---|
| 0.5 | 0.983 | 0.979 | 13 | 16 |
| **0.3** | **0.990** | 0.967 | **7** | 24 |

Lowering the threshold cut missed pneumonia cases from 13 to 7, at the cost of 8 
additional false alarms — a trade generally worth making in a screening tool.

A histogram of predicted probabilities showed a strongly bimodal distribution (most 
predictions clustered near 0.0 or 1.0, with the 0.3–0.8 range nearly empty), which 
explains why the threshold change had a limited effect on the overall metrics: the 
model is rarely "on the fence."

## Error analysis

Manual inspection of misclassified validation images (threshold 0.3):

**False Negatives (7 cases, actual PNEUMONIA → predicted NORMAL):**
3 of 7 show visible medical equipment (EKG electrodes, a catheter) unrelated to the 
lung pathology itself. This suggests possible shortcut learning — the model may have 
partly learned to associate the *absence* of visible equipment with "healthy-looking" 
patients, rather than relying solely on radiological features of pneumonia. The 
remaining 4 cases show no obvious visual abnormality, consistent with genuinely 
borderline, difficult cases.

**False Positives (24 cases, actual NORMAL → predicted PNEUMONIA):**
No consistent visual pattern was identified across these images. Unlike the FN group, 
this looks more like general model uncertainty on ambiguous images than an 
identifiable shortcut.

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib wandb pillow
```

Download the dataset from Kaggle and place it as `chest_xray/` next to the notebook 
(`train/`, `test/`, each with `NORMAL/` and `PNEUMONIA/` subfolders). Training and 
evaluation live in `pneumonia_classifier.ipynb`; it requires a Weights & Biases account 
(`wandb login`) for experiment logging and saves the trained weights to 
`model_weights.pth`. The inference scripts below expect that file in the same folder.

## Inference

```bash
python predict.py path/to/xray.jpg
```

Loads the saved model weights, applies the same evaluation preprocessing used during 
training, and prints the predicted class along with the model's confidence (predicted 
probability of `PNEUMONIA`) at the same 0.3 decision threshold used throughout this 
project.

## Demo / API

The trained model can also be used through a REST API and a small web demo. Both reuse 
the model-loading and prediction code from `app.py`, so there is a single source of 
truth for preprocessing and the 0.3 decision threshold. Extra dependencies:

```bash
pip install fastapi uvicorn python-multipart gradio
```

### API (FastAPI)

Start the server from this directory:

```bash
uvicorn app:app --reload
```

Interactive documentation (Swagger UI) is then available at 
`http://127.0.0.1:8000/docs`, where an image can be uploaded straight from the browser.

**Endpoint:** `POST /predict` — multipart form with the X-ray image in the `file` field.

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -F 'file=@path/to/xray.jpeg;type=image/jpeg'
```

Example response:

```json
{"label": "NORMAL", "confidence": 0.9974}
```

**About `confidence`:** in the API (and in the Gradio demo) it is the model's probability 
for the **returned label**. Because the decision threshold for `PNEUMONIA` is 0.3, that 
label can be returned with a confidence as low as 0.3. This differs from `predict.py`, 
which always prints the probability of `PNEUMONIA`.

### Demo (Gradio)

```bash
python gradio_app.py
```

Open `http://127.0.0.1:7860`, drop in a chest X-ray and the interface shows the predicted 
class and the confidence in two separate fields.

![Gradio demo](images/screen.png)

## Tech stack

- PyTorch, torchvision (`resnet18`, `ImageFolder`)
- scikit-learn (`train_test_split`, classification metrics, confusion matrix)
- Weights & Biases (experiment tracking)
- matplotlib (data/error visualization)
- FastAPI, Uvicorn (REST API)
- Gradio (web demo)

## Disclaimer

This is an educational project and not a medical device. It must not be used for 
clinical decisions.