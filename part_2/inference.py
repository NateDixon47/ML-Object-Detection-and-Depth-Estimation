import os
import numpy as np
import torch
from utils import *  
from sydrone_dataset import *  



def make_prediction(model_weights, results_dir="results"):
    """
    Loads a trained model and performs depth prediction on the test dataset.
    Saves the visualized results.

    """
    # Select computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the trained model
    model = initialize_depth_model(pretrained_weights=model_weights, device=device, eval_mode=True)

    # Load the test dataset
    dataloader = create_syndrone_dataloader(batch_size=1, shuffle_data=True, split='test')

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Iterate through the test dataset for inference
    for batch_idx, (input_image, truth_depth_map) in enumerate(dataloader):
        
        # Move input to the selected computation device
        input_image = input_image.to(device)

        # Perform inference to get the predicted depth map
        with torch.no_grad():
            predicted_depth_map = model(input_image)

        # Convert tensors to numpy arrays for visualization
        input_image_np = np.array(input_image.cpu()).squeeze()
        truth_depth_map_np = np.array(truth_depth_map).squeeze()
        predicted_depth_map_np = np.array(predicted_depth_map.cpu()).squeeze()

        # Save the visualized results as an image file
        save_path = os.path.join(results_dir, f"result_{batch_idx:03d}.png")
        plot_images(input_image_np, truth_depth_map_np, predicted_depth_map_np, save_path)

        print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    # Path to the model weights
    model_weights = 'tuned_models/syndrone_weights_final.pt'
    
    # Call the prediction function to process the test set and save results
    make_prediction(model_weights=model_weights)
