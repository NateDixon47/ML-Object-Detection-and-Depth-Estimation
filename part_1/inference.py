from ultralytics import YOLO
import os
import cv2
import numpy as np

# Load the YOLO model with trained weights
model = YOLO("best.pt")

# Paths to input images and output results
image_location = "C:/Users/ndixo/Desktop/ML_Final/part_1/short_train/images"
save_folder = "C:/Users/ndixo/Desktop/ML_Final/part_1/results"
os.makedirs(save_folder, exist_ok=True)

# Specify class IDs to detect 
vehicle_class_ids = [1, 2]  

# Path to the specific image
images = ['00022', '00684', '01516', '01247', '02104', '02254', '02623', '02946', '01649', '00048']

for idx, img in enumerate(images):
    image_path = f"{image_location}/{img}.jpg"

    # Perform inference, filtering only for vehicle classes
    result_image = model(image_path, classes=vehicle_class_ids)

    # Read the original image
    original_image = cv2.imread(image_path)

    # Create the image with bounding boxes
    annotated_image = result_image[0].plot()

    # Combine the original image and the annotated image side-by-side
    combined_image = np.hstack((original_image, annotated_image))

    # Save the combined image
    output_path = os.path.join(save_folder, f"result_image_{idx}.jpg")
    cv2.imwrite(output_path, combined_image)
    print(f'Image saved to {output_path}')

# Display the combined image
# cv2.imshow("Original and Result", combined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

