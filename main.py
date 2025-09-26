import os, json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

from fault_injection import inject_fault   # import your fault injector

# Config
root_dir = "/gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
model_name = "vit_base_patch16_224"
batch_size = 128
num_workers = min(4, os.cpu_count() or 2)
use_amp = True
max_batches = None  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.samples, self.targets = [], []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "r") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "r") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            syn_id = self.val_to_syn[entry]
            target = self.syn_to_class[syn_id]
            self.samples.append(os.path.join(samples_dir, entry))
            self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create model
model = timm.create_model(model_name, pretrained=True).to(device)
model.eval()
print(f"Using device: {device}")
print(f"Model loaded: {model_name}")

# fault injection
num_faults = 100

for _ in tqdm(range(num_faults), desc="Injecting faults"):
    inject_fault(model, component_type="attention")

# loader
val_dataset = ImageNetValDataset(root_dir, "val", transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

# accuracy
criterion = nn.CrossEntropyLoss()
total_loss, total_samples, top1_correct, top5_correct = 0.0, 0, 0, 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating", total=len(val_loader))):
        if max_batches and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        _, top5_pred = torch.topk(outputs, 5, dim=1)

        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        top1_correct += (predicted == labels).sum().item()
        top5_correct += (labels.unsqueeze(1) == top5_pred).any(dim=1).sum().item()
        
        print(f"Pred={predicted[0].item()}, Label={labels[0].item()}")

top1_acc = (top1_correct / total_samples) * 100
top5_acc = (top5_correct / total_samples) * 100
avg_loss = total_loss / total_samples

print("\n=== FINAL RESULTS ===")
print(f"Samples: {total_samples}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print(f"Average Loss: {avg_loss:.4f}")

with open("results.txt", "w") as f:
    f.write(f"Evaluated {total_samples} samples\n")
    f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
    f.write(f"Average Loss: {avg_loss:.4f}\n\n")

print(f"\nResults saved to results.txt")
