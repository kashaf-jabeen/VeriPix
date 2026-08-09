import torchvision.transforms as transforms

# Data transformation pipeline setup for training and testing
# EfficientNet/ResNet models require 224x224 input with ImageNet normalization

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

if __name__ == "__main__":
    print("Transforms Pipeline defined successfully!")
    print(f"Train Transforms: {train_transforms}")
    print(f"Test Transforms: {test_transforms}")