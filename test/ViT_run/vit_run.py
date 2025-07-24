# vit_run.py
import torch
import timm
from PIL import Image
from torchvision import transforms
import requests

# === Load image ===
img = Image.open("example.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to tensor and scales to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply transforms and ensure it's a tensor
x = transform(img)
if not isinstance(x, torch.Tensor):
    raise ValueError("Transform output is not a tensor!")
    
x = x.unsqueeze(0)  # Add batch dimension: shape becomes [1, 3, 224, 224]

# === Load model ===
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

with torch.no_grad():
    out = model(x)  # shape: [1, 1000]
    pred = out.argmax(dim=1).item()

# === Get class label ===
labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.strip().split("\n")

print(f"Predicted class: {labels[pred]}")
