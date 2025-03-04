import os
import shutil

# Paths to your resources
images_folder = "C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb"  # Path where all images are stored
labels_folder = "C:/Users/ndixo/Desktop/ML_Final/part_1/yolo_annotations"  # Path where all label `.txt` files are stored
train_split_file = "C:/Users/ndixo/Desktop/ML_Final/part_1/splits/train.txt"  # Train split file with image filenames (without extension)
val_split_file = "C:/Users/ndixo/Desktop/ML_Final/part_1/splits/test.txt"      # Validation split file with image filenames (without extension)
output_folder = "C:/Users/ndixo/Desktop/ML_Final/part_1"  # Output folder for YOLO dataset structure

# Subfolder names
train_images_folder = os.path.join(output_folder, "train/images")
train_labels_folder = os.path.join(output_folder, "train/labels")
val_images_folder = os.path.join(output_folder, "val/images")
val_labels_folder = os.path.join(output_folder, "val/labels")

# Create necessary folders
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

def copy_files(split_file, images_dest, labels_dest):
    """
    Copies image and label files into the appropriate directories based on a split file.
    """
    with open(split_file, "r") as f:
        file_names = [line.strip() for line in f.readlines()]

    for file_name in file_names:
        # Paths for images and labels
        image_file = os.path.join(images_folder, f"{file_name}.jpg")
        label_file = os.path.join(labels_folder, f"{file_name}.txt")

        # Destination paths
        dest_image_file = os.path.join(images_dest, f"{file_name}.jpg")
        dest_label_file = os.path.join(labels_dest, f"{file_name}.txt")

        # Copy image and label files if they exist
        if os.path.exists(image_file):
            shutil.copy(image_file, dest_image_file)
        else:
            print(f"Warning: Image file not found: {image_file}")
        
        if os.path.exists(label_file):
            shutil.copy(label_file, dest_label_file)
        else:
            print(f"Warning: Label file not found: {label_file}")

# Copy files for train and val splits
print("Copying training files...")
copy_files(train_split_file, train_images_folder, train_labels_folder)

print("Copying validation files...")
copy_files(val_split_file, val_images_folder, val_labels_folder)

print(f"Dataset organized successfully in {output_folder}")
