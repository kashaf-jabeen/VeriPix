import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Path setup
real_dir = 'dataset/train/REAL'
fake_dir = 'dataset/train/FAKE'

# REAL aur FAKE folders se 4-4 random images uthayein
real_images = random.sample(os.listdir(real_dir), 4)
fake_images = random.sample(os.listdir(fake_dir), 4)

# Plot setup (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('CIFAKE Dataset: REAL vs FAKE Images', fontsize=16)

# Display REAL images (Row 1)
for i, img_name in enumerate(real_images):
    img_path = os.path.join(real_dir, img_name)
    img = Image.open(img_path)
    axes[0, i].imshow(img)
    axes[0, i].set_title("REAL")
    axes[0, i].axis('off')

# Display FAKE images (Row 2)
for i, img_name in enumerate(fake_images):
    img_path = os.path.join(fake_dir, img_name)
    img = Image.open(img_path)
    axes[1, i].imshow(img)
    axes[1, i].set_title("FAKE")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()