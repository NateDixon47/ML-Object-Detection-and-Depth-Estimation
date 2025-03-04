import os
import random

# Path to your dataset's JSON files
dataset_path = "C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb"

# Output directory for split files
splits_path = "C:/Users/ndixo/Desktop/ML_Final/part_1/splits"
os.makedirs(splits_path, exist_ok=True)

# List all image files in the dataset directory
all_files = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.jpg')]



# Split into 80% train, 20% test
train_files = []
test_files = []
all = []


        
# Save splits to files
with open(os.path.join(splits_path, 'all.txt'), 'w') as f:
    f.write("\n".join(all_files))

# with open(os.path.join(splits_path, 'test.txt'), 'w') as f:
#     f.write("\n".join(test_files))

print(f"Train/Test splits created and saved to {splits_path}")
