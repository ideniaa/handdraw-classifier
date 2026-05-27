import os
import urllib.request
import numpy as np
import cv2

# QuickDraw dataset URL
categories = ["apple", "banana", "cat", "dog", "tree", "car", "fish"]  # Add more if needed

save_dir = "quickdraw_data"

# Make sure the base directory exists
os.makedirs(save_dir, exist_ok=True)

def download_quickdraw(category):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
    file_path = os.path.join(save_dir, f"{category}.npy")
    if not os.path.exists(file_path):
        print(f"Downloading {category} dataset...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"{category} dataset already exists.")

# Download each category
for category in categories:
    download_quickdraw(category)

# Convert .npy to images and organize them
for category in categories:
    data = np.load(os.path.join(save_dir, f"{category}.npy"))
    category_dir = os.path.join(save_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # FIX: Use min(1000, len(data)) to avoid IndexError if a category
    # has fewer than 1000 samples in the dataset
    num_images = min(1000, len(data))
    for i in range(num_images):
        img_path = os.path.join(category_dir, f"{i}.png")
        img = data[i].reshape(28, 28)  # Reshape to 28x28 pixels
        cv2.imwrite(img_path, img)

    print(f"Extracted {num_images} {category} images.")

    # Delete the .npy file after saving images
    npy_file_path = os.path.join(save_dir, f"{category}.npy")
    os.remove(npy_file_path)
    print(f"Deleted {category}.npy file.")

print("QuickDraw dataset ready!")
