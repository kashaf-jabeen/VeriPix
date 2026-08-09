import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split

# 1. Data transformation pipeline setup
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. ImageFolder datasets setup
train_dir = 'dataset/train'
test_dir = 'dataset/test'

full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# 3. Train/Validation Split (90% Train, 10% Validation)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Set generator seed for reproducible split
train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

if __name__ == "__main__":
    print("Train/Validation split completed successfully!")
    print(f"Classes mapping: {full_train_dataset.class_to_idx}")
    print(f"Total original train samples: {len(full_train_dataset)}")
    print(f"Actual Training samples (90%): {len(train_dataset)}")
    print(f"Validation samples (10%): {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")