import cv2
import os

def draw_yolo_boxes(image_path, annotation_path, image_width, image_height):

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image: {image_path}")
        return

    # Read the YOLO annotation file
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found: {annotation_path}")
        return

    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    # Loop through each bounding box in the annotation
    for annotation in annotations:
        parts = annotation.strip().split()
        if len(parts) != 5:
            print(f"Skipping invalid annotation line: {annotation}")
            continue

        # Parse YOLO format
        class_id, x_center, y_center, width, height = map(float, parts)
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # Convert YOLO format to (x_min, y_min, x_max, y_max)
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw bounding box and label on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"Class {int(class_id)}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_width = 1920
image_height = 1080
image_path = "C:/Users/ndixo/Desktop/ML_Final/data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb/02999.jpg"
annotation_path = "C:/Users/ndixo/Desktop/ML_Final/part_1/yolo_annotations/02999.txt"

draw_yolo_boxes(image_path, annotation_path, image_width, image_height)
