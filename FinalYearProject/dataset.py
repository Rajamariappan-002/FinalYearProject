import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random

class ManufacturingDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.good_images = [os.path.join(root_dir, "good", img) for img in os.listdir(os.path.join(root_dir, "good"))]
        self.bad_images = [os.path.join(root_dir, "bad", img) for img in os.listdir(os.path.join(root_dir, "bad"))]
        self.data_pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        for img1 in self.good_images:
            img2 = random.choice(self.good_images)
            pairs.append((img1, img2, 1))  # Same class (Good-Good)

        for img1 in self.good_images:
            img2 = random.choice(self.bad_images)
            pairs.append((img1, img2, 0))  # Different class (Good-Bad)

        for img1 in self.bad_images:
            img2 = random.choice(self.bad_images)
            pairs.append((img1, img2, 1))  # Same class (Bad-Bad)

        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data_pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Define transformations (Augmentation & Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Updated for 3-channel images
])

# Load dataset
def get_dataloader(root_dir, batch_size=16, shuffle=True):
    dataset = ManufacturingDefectDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
