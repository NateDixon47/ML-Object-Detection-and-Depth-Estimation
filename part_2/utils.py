import torch
from matplotlib import pyplot as plt
import numpy as np


def plot_images(input_image, truth_depth_map, predicted_depth_map, save_path):
    """
    Plots the original image, predicted depth map, and ground truth depth map.
    Saves the resulting figure to the specified path.

    """
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the original image
    axs[0].imshow(np.transpose(input_image, (1, 2, 0)))
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot the predicted depth map
    axs[1].imshow(np.array(predicted_depth_map).squeeze(), cmap='magma')
    axs[1].axis('off')
    axs[1].set_title('Predicted Depth Map')

    # Plot the ground truth depth map
    axs[2].imshow(np.array(truth_depth_map).squeeze(), cmap='magma')
    axs[2].axis('off')
    axs[2].set_title('Truth Depth Map')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Free up memory by closing the figure
    
def adjust_prediction(prediction, scale=0.00002517, offset=0.00003332):
    """
    Adjusts the predicted values using a linear transformation.

    Args:
        prediction (torch.Tensor): Predicted values.
        scale (float): Scaling factor.
        offset (float): Offset value.

    Returns:
        torch.Tensor: Adjusted predictions.
    """
    transformed_prediction = scale * prediction + offset
    return transformed_prediction

def compute_eigen_loss(predicted, target, regularization=0.5):
    """
    Computes the Eigen loss between the predicted and ground truth values.

    Args:
        predicted (torch.Tensor): Predicted depth values.
        target (torch.Tensor): Ground truth depth values.
        regularization (float): Regularization parameter for scale-invariant loss.

    Returns:
        torch.Tensor: Computed Eigen loss.
    """
    # Convert to inverse depth
    inv_predicted = 1 / (predicted + 10**-4.5)
    inv_target = 1 / (target + 10**-4.5)

    # Calculate scale-invariant loss components
    log_difference = torch.log(inv_predicted) - torch.log(inv_target)
    num_elements = log_difference.numel()
    mse_term = torch.sum(log_difference**2) / num_elements
    regularization_term = regularization * (torch.sum(log_difference) / num_elements)**2

    # Compute final loss
    scale_invariant_loss = mse_term - regularization_term
    return scale_invariant_loss


