"""Data loading utilities."""

import json
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """ImageNet dataset (validation or training)."""

    def __init__(self, root: str, transform=None, split: str = "val"):
        """
        Args:
            root: Path to ImageNet root directory
            transform: Image transforms to apply
            split: "val" for validation, "train" for training
        """
        self.samples = []
        self.targets = []
        self.transform = transform
        self.split = split

        with open(os.path.join(root, "imagenet_class_index.json")) as f:
            class_idx = json.load(f)
            self.syn_to_class = {v[0]: int(k) for k, v in class_idx.items()}

        if split == "val":
            self._load_validation(root)
        else:
            self._load_training(root)

    def _load_validation(self, root: str):
        """Load validation set."""
        with open(os.path.join(root, "ILSVRC2012_val_labels.json")) as f:
            val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/val")
        for entry in os.listdir(samples_dir):
            syn_id = val_to_syn[entry]
            self.samples.append(os.path.join(samples_dir, entry))
            self.targets.append(self.syn_to_class[syn_id])

    def _load_training(self, root: str):
        """Load training set."""
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/train")

        if not os.path.exists(samples_dir):
            raise FileNotFoundError(
                f"Training data not found at {samples_dir}. "
                "Training data is organized by synset folders (n01440764, n01443537, etc.)"
            )

        for syn_id in os.listdir(samples_dir):
            if syn_id not in self.syn_to_class:
                continue

            class_dir = os.path.join(samples_dir, syn_id)
            if not os.path.isdir(class_dir):
                continue

            class_label = self.syn_to_class[syn_id]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.targets.append(class_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
