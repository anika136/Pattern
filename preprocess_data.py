import os
import pickle
import cv2
import numpy as np

# ✅ Dataset Path (Update if needed)
DATA_DIR = "./asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 64  # Resize images to 64x64 pixels

data = []
labels = []

# ✅ Check if dataset exists
if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Dataset folder not found at {DATA_DIR}")
    exit()

# ✅ Loop through dataset folders (A-Z)
valid_labels = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("🔄 Processing images...")
for dir_ in sorted(os.listdir(DATA_DIR)):
    if dir_ not in valid_labels:
        print(f"⚠️ Skipping folder: {dir_} (not A-Z)")
        continue

    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"📂 Processing folder: {dir_}")

    for img_path in sorted(os.listdir(dir_path)):
        img_file = os.path.join(dir_path, img_path)
        img = cv2.imread(img_file)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_file}")
            continue  # Skip unreadable images

        # ✅ Resize Image and Normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # Normalize between 0-1
        data.append(img)
        labels.append(dir_)

print(f"✅ Total Processed Samples: {len(data)}")

# ✅ Convert Lists to NumPy Arrays
if len(data) == 0:
    print("❌ Error: No images processed! Exiting.")
    exit()

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# ✅ Save Processed Data
try:
    with open('data_cnn.pickle', 'wb') as f:  # Save in main project directory
        pickle.dump({'data': data, 'labels': labels}, f)
    print("🎉 File saved successfully: data_cnn.pickle")
except Exception as e:
    print(f"❌ Error saving data_cnn.pickle: {e}")
    exit()

print(f"🎉 Data preprocessing complete! {len(data)} samples saved in 'data_cnn.pickle'.")
