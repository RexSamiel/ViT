import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset_loader import ImageNetValDataset
from fault_injection import inject_fault

# Config
root_dir = "/gpfs/mariana/home/svloor/Documents/vit/data/imagenet"
model_name = "vit_base_patch16_224"
batch_size = 128
num_workers = min(4, os.cpu_count() or 2)
use_amp = True
max_batches = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Create model
model = timm.create_model(model_name, pretrained=True).to(device)
model.eval()
print(f"Using device: {device}")
print(f"Model loaded: {model_name}")

# fault injection
inject_fault(model, component_type="attention")

# loader
val_dataset = ImageNetValDataset(root_dir, "val", transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


def run_model():
    logits_list = []
    labels_list = []
    losses_list = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Store everything on CPU
            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())
            losses_list.append(loss.item())

    top_metrics(logits_list, labels_list)


def top_metrics(logits_list, labels_list):
    """
    Compute Top-1 and Top-5 accuracy metrics.
    """
    top1_list = []
    top5_list = []

    for outputs, labels in zip(logits_list, labels_list):
        _, predicted = torch.max(outputs, 1)
        _, top5_pred = torch.topk(outputs, 5, dim=1)

        top1_list.append((predicted == labels).numpy())
        top5_list.append(
            ((top5_pred == labels.unsqueeze(1)).any(dim=1)).numpy()
        )

    top1_all = np.concatenate(top1_list)
    top5_all = np.concatenate(top5_list)

    metrics = {
        "top1_avg": float(top1_all.mean()),
        "top1_best": float(top1_all.max()),
        "top1_worst": float(top1_all.min()),
        "top5_avg": float(top5_all.mean()),
        "top5_best": float(top5_all.max()),
        "top5_worst": float(top5_all.min()),
    }

    print("\n=== TOP-K METRICS ===")
    print(
        f"Top-1 Accuracy: Avg={metrics['top1_avg']*100:.2f}%, "
        f"Best={metrics['top1_best']*100:.2f}%, "
        f"Worst={metrics['top1_worst']*100:.2f}%"
    )
    print(
        f"Top-5 Accuracy: Avg={metrics['top5_avg']*100:.2f}%, "
        f"Best={metrics['top5_best']*100:.2f}%, "
        f"Worst={metrics['top5_worst']*100:.2f}%\n"
    )

    return metrics


if __name__ == "__main__":
    run_model()
