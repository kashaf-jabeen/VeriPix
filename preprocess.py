import os
import torchvision.transforms as transforms
from torchvision import datasets

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

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

if __name__ == "__main__":
    print("Dataset successfully loaded using ImageFolder!")
    print(f"Classes mapping: {train_dataset.class_to_idx}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total testing samples: {len(test_dataset)}")