import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader

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

train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 4. DataLoaders Creation (batch_size = 32)
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if __name__ == "__main__":
    print("DataLoaders created successfully!")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    print(f"Test Batches: {len(test_loader)}")
    
    # Verify one batch structure
    images, labels = next(iter(train_loader))
    print(f"Sample Batch Image Shape: {images.shape}")
    print(f"Sample Batch Label Shape: {labels.shape}")