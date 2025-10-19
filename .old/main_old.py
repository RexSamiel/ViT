import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# Config
root_dir = "/gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
val_dir = os.path.join(root_dir, "Documents/vit/data/imagenet/ILSVRC/Data/CLS-LOC/val")
eval_file = os.path.join(root_dir, "val.txt")
model_name = "vit_base_patch16_224"
batch_size = 128
num_workers = min(4, os.cpu_count() or 2)
use_amp = True
max_batches = None  # Set to None for full evaluation, or number for quick testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset loader
class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
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
            if split == "val":
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

#Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Create model
model = timm.create_model(model_name, pretrained=True)
model.to(device)
print(f"Using device: {device}")
print(f"Model loaded: {model_name}")
model.eval()

#Loader functions
val_dataset = ImageNetValDataset(root_dir, "val", transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

print(f"Total validation samples: {len(val_dataset)}")
print(f"Batch size: {batch_size}")
if max_batches:
    print(f"Testing on {max_batches} batches ({max_batches * batch_size} samples max)")
else:
    print("Running full evaluation")
print("-" * 50)

#Accuracy tester
criterion = nn.CrossEntropyLoss()
total_loss = 0.0
total_samples = 0
top1_correct = 0
top5_correct = 0

sample_results = []
max_sample_results = 25

with torch.no_grad():  
    for batch_idx, (images, labels) in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        _, top5_pred = torch.topk(outputs, 5, dim=1)
        
        batch_size_actual = labels.size(0)
        total_samples += batch_size_actual
        total_loss += loss.item() * batch_size_actual  # Scale by batch size for proper averaging
        
        top1_correct += (predicted == labels).sum().item()
        
        top5_correct += (labels.unsqueeze(1) == top5_pred).any(dim=1).sum().item()
        
        if batch_idx == 0 and len(sample_results) < max_sample_results:
            for i in range(min(batch_size_actual, max_sample_results)):
                sample_results.append(
                    f"Sample {len(sample_results) + 1}: Label={labels[i].item()}, "
                    f"Pred={predicted[i].item()}, "
                    f"Correct={'Yes' if predicted[i] == labels[i] else 'No'}"
                )
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            current_top1 = (top1_correct / total_samples) * 100
            current_top5 = (top5_correct / total_samples) * 100
            print(f"Batch {batch_idx + 1}: "
                  f"Samples {total_samples}, "
                  f"Top-1: {current_top1:.2f}%, "
                  f"Top-5: {current_top5:.2f}%")

top1_acc = (top1_correct / total_samples) * 100
top5_acc = (top5_correct / total_samples) * 100
avg_loss = total_loss / total_samples

print("-" * 50)
print(f"FINAL RESULTS:")
print(f"Evaluated {total_samples} samples")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print(f"Average Loss: {avg_loss:.4f}")

# Save results to file
with open("results.txt", "w") as f:
    f.write(f"Evaluated {total_samples} samples\n")
    f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
    f.write(f"Average Loss: {avg_loss:.4f}\n\n")
    f.write(f"Sample predictions from first batch:\n")
    f.write("\n".join(sample_results))

print(f"\nResults saved to results.txt")
