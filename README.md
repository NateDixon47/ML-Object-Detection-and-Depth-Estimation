# Object Detection and Depth Estimation for Drone Camera Images
## Project Description
This repository contains the final project for RBE 577: Machine Learning for Robotics, completed in March 2025. The project focuses on two core tasks using the SynDrone dataset:

- **Object Detection**: Fine-tuning the YOLO v11 model to detect vehicles (cars and trucks) in RGB drone images, generating accurate bounding boxes for real-time applications.
- **Depth Estimation**: Implementing a Vision Transformer (ViT)-based model to predict depth maps from drone-captured RGB images, supervised by ground-truth depth data.

The goal is to enable drone perception in urban environments, supporting applications like autonomous navigation, traffic monitoring, or search-and-rescue operations. The project includes custom data preprocessing, model training pipelines, and evaluation scripts, all tailored to the synthetic SynDrone Town01 dataset.

# Library Versions
The following libraries were used to develop and run this project. Ensure compatibility by installing these specific versions:

- **Python**: 3.8.20
- **Torch**: 2.4.0
- **TorchVision**: 0.19.0
- **Numpy**: 1.24.3
- **Matplotlib**: 3.7.3
- **OpenCV**: 4.10.0
- **Timm**: 1.0.12
- **Ultralytics**: 8.3.40

## Dataset Preparation
This project uses the SynDrone Town01 dataset, which includes RGB images, depth maps, and bounding box annotations for vehicles in an urban setting. Follow these steps to set up the dataset:

1. Download the Dataset: Obtain the SynDrone Town01 dataset and place it in the data/ directory of this repository.
2. Organize Depth Maps:
Locate the depth folder inside Town01_Opt_120_depth/height20m/.
Move this depth folder into color/height20m/ within the dataset structure.
Final path: data/color/height20m/depth/.
3. Verify Structure: Ensure RGB images, depth maps, and annotations are correctly aligned (e.g., matching filenames). The project assumes this layout for loading and preprocessing.
4. Download the 'yolo11m.pt' model from https://github.com/ultralytics/ultralytics.git and place it in the 'part_1' directory.
5. Download the 'dpt_hybrid-midas-501f0c75.pt' model from https://github.com/isl-org/DPT.git and place it in the 'part_2/pretrained_models' directory.

## Setup
1. Clone the Repository:
- **git clone https://github.com/NateDixon47/ML-Object-Detection-and-Depth-Estimation.git**
- **cd ML-Object-Detection-and-Depth-Estimation**

2. Install Dependencies:
- **pip install torch==2.4.0 torchvision==0.19.0 numpy==1.24.3 matplotlib==3.7.3 opencv-python==4.10.0 timm==1.0.12 ultralytics==8.3.40**
3. Prepare the Dataset: Follow the steps in Dataset Preparation.

## Usage
### Object Detection
- **Script**: Run the 'train.py' script in the 'part_1' directory to fine-tune the YOLO model with the Syndrone dataset.
- **Output**: Trained YOLO v11 model weights and detection results.
- **Results**: Run the 'inference.py' script in the 'part_1' directory to view the models predictions on the given images.

    ![Vehicle Detection Sample](part_1\results\result_image_4.jpg "Sample output of YOLO v11 detecting cars")

### Depth Estimation
- **Script**: Run the 'train.py' script in the 'part_2' directory to train the depth estimation model on the Syndrone dataset.
- **Output**: Fine-tuned depth estimation model weights and predicted depth maps.
- **Results**: Run the 'inference.py' script in the 'part_2' directory to view the models predicted depth maps.

    ![Depth Estimation Sample](presentation_and_results\part_2_results\result_004.png "Sample output of depth estimation model")

## Results
Object Detection: Achieved high accuracy in detecting vehicles, with bounding boxes correctly placed in most test cases.
Depth Estimation: Generated depth maps aligning well with ground truth, though some scenarios showed room for improvement.
