"""Data loading utilities."""

import json
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """ImageNet validation dataset."""

    def __init__(self, root: str, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform

        # Load class mappings
        with open(os.path.join(root, "imagenet_class_index.json")) as f:
            class_idx = json.load(f)
            syn_to_class = {v[0]: int(k) for k, v in class_idx.items()}

        with open(os.path.join(root, "ILSVRC2012_val_labels.json")) as f:
            val_to_syn = json.load(f)

        # Collect samples
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/val")
        for entry in os.listdir(samples_dir):
            syn_id = val_to_syn[entry]
            self.samples.append(os.path.join(samples_dir, entry))
            self.targets.append(syn_to_class[syn_id])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
