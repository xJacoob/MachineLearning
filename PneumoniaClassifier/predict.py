from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import argparse

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

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transformed_image = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        probs = torch.softmax(output, dim=1)
        result = probs[0, 1].item() >= 0.3
        confidence = probs[0, 1].item()

    return class_names[result], confidence

parser = argparse.ArgumentParser(description='Image Classification with PyTorch')
parser.add_argument('image_path', type=str, help='path to image')
args = parser.parse_args()

class_name, confidence_of_pneumonia = predict(args.image_path)
print(f"Prediction: {class_name}, Confidence of PNEUMONIA: {confidence_of_pneumonia}")
