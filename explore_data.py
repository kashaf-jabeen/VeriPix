import os

# Paths setup
train_real = 'dataset/train/REAL'
train_fake = 'dataset/train/FAKE'
test_real = 'dataset/test/REAL'
test_fake = 'dataset/test/FAKE'

# Count images using os.listdir
num_train_real = len(os.listdir(train_real)) if os.path.exists(train_real) else 0
num_train_fake = len(os.listdir(train_fake)) if os.path.exists(train_fake) else 0
num_test_real = len(os.listdir(test_real)) if os.path.exists(test_real) else 0
num_test_fake = len(os.listdir(test_fake)) if os.path.exists(test_fake) else 0

print("--- DATASET COUNT REPORT ---")
print(f"Train - REAL Images: {num_train_real}")
print(f"Train - FAKE Images: {num_train_fake}")
print(f"Test  - REAL Images: {num_test_real}")
print(f"Test  - FAKE Images: {num_test_fake}")
print(f"Total Dataset Images: {num_train_real + num_train_fake + num_test_real + num_test_fake}")