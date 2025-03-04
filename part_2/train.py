import os
import torch
import time
from utils import *
from sydrone_dataset import *
from torch.optim.lr_scheduler import ExponentialLR

# Set up runtime information and directories
print("Starting training...")

# Select the computing device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters for the training process
hyper_parameters = {
    'lr': 0.0001,  # Initial learning rate
    'batch_size': 1,         # Batch size for training and testing
    'epochs': 25,        # Total number of training epochs
    'weights': "dpt_hybrid",  # Pretrained model weights
    'lamda': 0.5,            # Lambda parameter for eigen loss
    'gamma': 0.95            # Learning rate decay factor
}

# Initialize the depth model
model = initialize_depth_model(
    pretrained_weights=hyper_parameters['weights'],
    device=device,
    eval_mode=False 
)

# Load the training and testing dataloaders
dataloader_train = create_syndrone_dataloader(batch_size=hyper_parameters['batch_size'], shuffle_data=True, split='train')
dataloader_test = create_syndrone_dataloader(batch_size=hyper_parameters['batch_size'], shuffle_data=False, split='test')
print(f"Number of batches in training set: {len(dataloader_train)}")

# Set up the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameters['lr'])
scheduler = ExponentialLR(optimizer, gamma=hyper_parameters['gamma'])  # Decay learning rate by gamma after each epoch

# Training loop
epochs = hyper_parameters['epochs']
for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"Epoch: {epoch + 1}/{epochs}")

    # Training phase
    model.train()  # Set model to training mode
    train_loss = 0
    for batch_idx, (images, depth_maps) in enumerate(dataloader_train):
        batch_start_time = time.time()

        # Move images and depth_maps to the selected device
        images, depth_maps = images.to(device), depth_maps.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Align predictions and calculate loss
        aligned_predictions = adjust_prediction(outputs.cpu())
        loss = compute_eigen_loss(aligned_predictions.cpu(), depth_maps.cpu(), regularization=hyper_parameters['lamda'])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # Accumulate training loss
        print(f"Batch {batch_idx + 1}/{len(dataloader_train)} | Time: {time.time() - batch_start_time:.2f}s")

    # Average training loss
    avg_train_loss = train_loss / len(dataloader_train)

    # Testing phase
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    with torch.no_grad():
        for images, depth_maps in dataloader_test:
            # Move images and depth_maps to the selected device
            images, depth_maps = images.to(device), depth_maps.to(device)

            # Forward pass
            outputs = model(images)

            # Align predictions and calculate loss
            aligned_predictions = adjust_prediction(outputs.cpu())
            loss = compute_eigen_loss(aligned_predictions.cpu(), depth_maps.cpu(), regularization=hyper_parameters['lamda'])

            test_loss += loss.item()  # Accumulate testing loss

    # Average testing loss
    avg_test_loss = test_loss / len(dataloader_test)

    # Log epoch information
    epoch_duration = time.time() - epoch_start_time
    print(f"\tTrain Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f} | "
          f"Epoch Time: {epoch_duration.seconds // 60}m {epoch_duration.seconds % 60}s")

    # Save model weights for the current epoch
    model_save_path = f"tuned_models/syndrone_weights_test.pt"
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    model.to(device)  # Move model back to the selected device

    # Step the learning rate scheduler
    scheduler.step()

# Save final model weights
final_model_path = f"tuned_models/syndrone_weights_test.pt"
torch.save(model.cpu().state_dict(), final_model_path)
print(f"Training complete. Final model saved to {final_model_path}")
