import json
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageNetValDataset(Dataset):
    def __init__(self, root, split="val", transform=None):
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
