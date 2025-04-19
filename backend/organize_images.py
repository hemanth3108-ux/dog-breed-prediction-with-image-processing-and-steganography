import os
import shutil
import pandas as pd

# Paths
base_dir = 'C:/Users/lenovo/OneDrive/Desktop/dog/dog-breed-identification'
train_dir = os.path.join(base_dir, 'train')
labels_path = os.path.join(base_dir, 'labels.csv')

# Read the labels
df = pd.read_csv(labels_path)

# Loop through the CSV and move images to breed-specific folders
for _, row in df.iterrows():
    breed = row['breed']
    img_name = row['id'] + '.jpg'
    
    src = os.path.join(train_dir, img_name)
    breed_dir = os.path.join(train_dir, breed)
    dst = os.path.join(breed_dir, img_name)
    
    os.makedirs(breed_dir, exist_ok=True)
    
    if os.path.exists(src):  # Only move if the image exists
        shutil.move(src, dst)

print("Images reorganized by breed successfully!")
