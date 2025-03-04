import os
import random

# Path to your dataset's image files
dataset_path = "C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb"

# Output directory for split files
splits_path = "C:/Users/ndixo/Desktop/ML_Final/part_2/splits"
os.makedirs(splits_path, exist_ok=True)

# List all image files in the dataset directory
all_files = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# Randomly select 100 files
random.seed(42)  # Set seed for reproducibility
selected_files = random.sample(all_files, 100)

# Split into 80% train and 20% test
split_index = int(len(selected_files) * 0.8)
train_files = selected_files[:split_index]
test_files = selected_files[split_index:]

# Save splits to files
with open(os.path.join(splits_path, 'short_train.txt'), 'w') as f:
    f.write("\n".join(train_files))

with open(os.path.join(splits_path, 'short_test.txt'), 'w') as f:
    f.write("\n".join(test_files))

print(f"Train/Test splits created with 80 train and 20 test images and saved to {splits_path}")
