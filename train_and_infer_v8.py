import os
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Define paths (update these paths according to your setup)
data_yaml_path = '/home/mdabubakrsiddique/Documents/animal_tag/data.yaml'  
weights_path = '/home/mdabubakrsiddique/Documents/animal_tag/yolov8/yolov8s.pt'  
train_img_folder = '/home/mdabubakrsiddique/Documents/animal_tag/dataset/images/train'  
val_img_folder = '/home/mdabubakrsiddique/Documents/animal_tag/dataset/images/val'  
inference_output_folder = '/home/mdabubakrsiddique/Documents/animal_tag/output'  
yolov8_path = '/home/mdabubakrsiddique/Documents/animal_tag/yolov8'  
run_path = '/home/mdabubakrsiddique/Documents/animal_tag'  

# Ensure the paths are correct
assert os.path.exists(data_yaml_path), f"{data_yaml_path} does not exist"
assert os.path.exists(train_img_folder), f"{train_img_folder} does not exist"
assert os.path.exists(val_img_folder), f"{val_img_folder} does not exist"
assert os.path.exists(yolov8_path), f"{yolov8_path} does not exist"

# Print the paths to verify
print(f"data_yaml_path: {data_yaml_path}")
print(f"train_img_folder: {train_img_folder}")
print(f"val_img_folder: {val_img_folder}")
print(f"yolov8_path: {yolov8_path}")

# Function to get the latest detector folder
def get_latest_detector_folder(base_path):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No such directory: {base_path}")
    folders = [f for f in os.listdir(base_path) if re.match(r'animal_tag_detector\d+', f)]
    if not folders:
        raise FileNotFoundError(f"No detector folders found in {base_path}")
    latest_folder = max(folders, key=lambda x: int(re.search(r'\d+', x).group()))
    return latest_folder

# Training function
def train_model():
    print("Starting training...")
    model = YOLO('yolov8n.yaml')  # Load a new YOLOv8 model from a YAML file
    model.train(data=data_yaml_path, epochs=300, imgsz=640, batch=4, name='animal_tag_detector', patience=10)
    print("Training completed")

# Validation function
def validate_model():
    print("Starting validation...")
    latest_folder = get_latest_detector_folder(f'{run_path}/runs/detect')
    weights_path = f'{run_path}/runs/detect/{latest_folder}/weights/best.pt'
    model = YOLO(weights_path)
    model.val(data=data_yaml_path)
    print("Validation completed")

def run_inference(image_folder, output_folder):
    latest_folder = get_latest_detector_folder(f'{run_path}/runs/detect')
    model_path = f'{run_path}/runs/detect/{latest_folder}/weights/best.pt'
    model = YOLO(model_path)

    # Create a directory to save the output images with detected objects
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)

            # Load the image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(img_rgb)

            # Process each detection result
            for result in results:
                # Convert results to a DataFrame
                results_df = result.pandas().xyxy[0]

                # Draw bounding boxes on the image
                for _, row in results_df.iterrows():
                    x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
                    label = f'{cls} {conf:.2f}'
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the output image with detections
            output_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_img_path, img)

            # Optionally, display the image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


# Main script
if __name__ == '__main__':
    # Train the model
    train_model()

    # Validate the model
    validate_model()

    # Run inference on validation images
    run_inference(val_img_folder, inference_output_folder)
