import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import DPT.util.io as dpt_io
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from DPT.dpt.models import DPTDepthModel


# Dataset class for Syndrone images and depth maps
class SyndroneData(Dataset):
    def __init__(self, image_folder, depth_folder, image_transforms=None, split="train"):
        """
        Initializes the SyndroneDataset with image and depth directories.

        Args:
            image_folder: folder containing RGB images.
            depth_folder: folder containing depth maps.
            image_transforms: Transformations for RGB images.
            split : split to use ('train' / 'test').
        """
        self.image_transforms = image_transforms or transforms.ToTensor()
        self.image_folder = image_folder
        self.depth_folder = depth_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.depth_files = sorted(os.listdir(depth_folder))

        # Load split file for valid indices
        split_file_path = f"splits/{split}.txt"
        with open(split_file_path) as file:
            files = set(int(line.strip()) for line in file.readlines())

        # Filter files based on valid indices
        self.image_files = [file for idx, file in enumerate(self.image_files) if idx in files]
        self.depth_files = [file for idx, file in enumerate(self.depth_files) if idx in files]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding depth map.

        Args:
            index: Index of the sample.

        Returns:
            tuple: Transformed image tensor and depth map.
        """
        # File paths for image and depth map
        image_path = os.path.join(self.image_folder, self.image_files[index])
        depth_path = os.path.join(self.depth_folder, self.depth_files[index])
        
        # Read and transform RGB image
        rgb_image = dpt_io.read_image(image_path)
        transformed_image = self.image_transforms({"image": rgb_image})["image"]
        
        # Load and preprocess depth map
        depth_image = Image.open(depth_path)
        depth_array = np.asarray(depth_image, dtype=np.float32)
        depth_min, depth_max = depth_array.min(), depth_array.max()
        resized_depth = cv2.resize(depth_array, (transformed_image.shape[2], transformed_image.shape[1]), interpolation=cv2.INTER_CUBIC)
        normalized_depth = np.clip(resized_depth, depth_min, depth_max)
        inverse_depth = 1 / normalized_depth

        return transformed_image, inverse_depth


# Function to load the DPT model
def initialize_depth_model(pretrained_weights="dpt_hybrid", backbone_type="vitl16_384", device=None, eval_mode=False):
    """
    Loads and initializes the DPT depth estimation model.

    Args:
        pretrained_weights (str): Pretrained weights to load ('dpt_large' or 'dpt_hybrid').
        backbone_type (str): Backbone configuration for the model.
        device (torch.device): Device for loading the model (CPU/GPU).
        eval_mode (bool): Whether to set the model in evaluation mode.

    Returns:
        torch.nn.Module: Initialized DPT depth model.
    """
    if pretrained_weights == "dpt_large":
        pretrained_weights = "pretrained_models/dpt_large.pt"
        backbone_type = "vitl16_384"
    elif pretrained_weights == "dpt_hybrid":
        pretrained_weights = "pretrained_models/dpt_hybrid.pt"
        backbone_type = "vitb_rn50_384"

    # Initialize DPT model
    depth_model = DPTDepthModel(
        path=pretrained_weights,
        backbone=backbone_type,
        non_negative=True,
        enable_attention_hooks=False,
    )

    # Auto-select CPU or GPU if device is not specified
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model.to(device)

    # Set the model in evaluation mode if specified
    if eval_mode:
        depth_model.eval()

    print(f"Depth model initialized with weights: {pretrained_weights} on device: {device}")
    return depth_model


# DataLoader function for Syndrone dataset
def create_syndrone_dataloader(
    rgb_dir="C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb",
    depth_folder="C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/depth",
    batch_size=1,
    shuffle_data=False,
    split="short_train"
):
    """
    Sets up the Syndrone dataset DataLoader.

    Args:
        rgb_dir (str): Path to RGB images directory.
        depth_folder (str): Path to depth maps directory.
        batch_size (int): Number of samples per batch.
        shuffle_data (bool): Whether to shuffle the dataset.
        split (str): Split to use ('short_train', 'short_test', etc.).

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the dataset.
    """
    # Image resizing dimensions
    image_width = image_height = 384

    # Define RGB transformations
    rgb_image_transforms = Compose([
        Resize(
            image_width,
            image_height,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])

    # Create dataset and dataloader
    syndrone_dataset = SyndroneData(image_folder=rgb_dir, depth_folder=depth_folder, image_transforms=rgb_image_transforms, split=split)
    dataloader = DataLoader(dataset=syndrone_dataset, batch_size=batch_size, shuffle=shuffle_data)

    print(f"Syndrone DataLoader setup complete for '{split}' with {len(syndrone_dataset)} samples, batch size: {batch_size}, shuffle: {shuffle_data}")
    return dataloader
