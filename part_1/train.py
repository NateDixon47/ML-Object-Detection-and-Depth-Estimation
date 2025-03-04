from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pre-trained YOLO model by specifying the model weights file
    model = YOLO("yolo11m.pt")  # Path to the YOLO model weights file 

    # Train the YOLO model
    results = model.train(
        data="dataset.yaml",  # Path to the dataset YAML file specifying training data and class names
        epochs=20,                 # Number of training epochs
        patience=5,               # Early stopping patience 
        batch=8,                   # Batch size for training
        workers=8,                # Number of data loader workers 
        imgsz=640,                 # Input image size 
        device='cpu',              # Device to use for training 
        lr0=0.01,                  # Initial learning rate
        lrf=0.1,                   # Final learning rate as a fraction of the initial learning rate
        warmup_epochs=2,           # Number of warmup epochs 
        warmup_bias_lr=0.05,  # Initial learning rate for biases during warmup
        warmup_momentum=0.95,  # Start with higher momentum during warmup
        weight_decay=0.0005,       # L2 regularization for preventing overfitting
        save_period=2
    )

    # Validate the trained model to evaluate its performance on the validation dataset
    metrics = model.val()  
    test_results = model.val(split='test')  # Evaluate on the test dataset
    # Export the trained YOLO model to the ONNX format for compatibility with other frameworks
    path = model.export(format="onnx")  
