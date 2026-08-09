from preprocess import train_loader

if __name__ == '__main__':
    # Fetch a single batch from DataLoader safely
    images, labels = next(iter(train_loader))

    print("=== Batch Verification ===")
    print(f"Images tensor shape: {images.shape}")
    print(f"Labels tensor shape: {labels.shape}")
    print(f"First 5 labels in batch: {labels[:5]}")