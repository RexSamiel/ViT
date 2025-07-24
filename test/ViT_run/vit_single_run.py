import torch
import timm
from PIL import Image
from torchvision import transforms
import requests
from pathlib import Path

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
model.eval()

# Labels
labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.strip().split("\n")

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Folder path
image_folder = Path("~/Documents/ViT/test/pytorch-image-models/images").expanduser()
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

if not image_folder.exists():
    print(f"Folder not found: {image_folder}")
    exit(1)

# Process images
for img_path in image_folder.iterdir():
    if img_path.suffix.lower() not in image_extensions:
        continue

    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Transform failed on {img_path.name}")

        batch_tensor = tensor[None, ...].to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(batch_tensor)
            pred_index = output.argmax(dim=1).item()

        print(f"Image: {img_path.name} | Predicted: {labels[pred_index]}")
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")


