from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model_weights.pth", map_location=device))

model.to(device)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class_names = ['NORMAL', "PNEUMONIA"]

def predict_image(image: Image.Image):
    transformed_image = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        probs = torch.softmax(output, dim=1)
        pneumonia_probs = probs[0, 1].item()
        result = pneumonia_probs >= 0.3

    predicted_class = class_names[result]
    confidence = pneumonia_probs if result else probs[0, 0].item()

    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    label, confidence = predict_image(image)
    return {"label": label, "confidence": confidence}
