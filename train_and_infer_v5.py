import os
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# Define paths (update these paths according to your setup)
data_yaml_path = '/home/mdabubakrsiddique/Documents/animal_tag/data.yaml'  
model_cfg_path = '/home/mdabubakrsiddique/Documents/animal_tag/yolov5/models/yolov5s.yaml'  
weights_path = 'yolov5s.pt'  
train_img_folder = '/home/mdabubakrsiddique/Documents/animal_tag/dataset/images/train'  
val_img_folder = '/home/mdabubakrsiddique/Documents/animal_tag/dataset/images/val'  
inference_output_folder = '/home/mdabubakrsiddique/Documents/animal_tag/output'  
yolov5_path = '/home/mdabubakrsiddique/Documents/animal_tag/yolov5' 

# Ensure the paths are correct
assert os.path.exists(data_yaml_path), f"{data_yaml_path} does not exist"
assert os.path.exists(train_img_folder), f"{train_img_folder} does not exist"
assert os.path.exists(val_img_folder), f"{val_img_folder} does not exist"
assert os.path.exists(yolov5_path), f"{yolov5_path} does not exist"

# Print the paths to verify
print(f"data_yaml_path: {data_yaml_path}")
print(f"train_img_folder: {train_img_folder}")
print(f"val_img_folder: {val_img_folder}")
print(f"yolov5_path: {yolov5_path}")

# Function to get the latest detector folder
def get_latest_detector_folder(base_path):
    folders = [f for f in os.listdir(base_path) if re.match(r'animal_tag_detector\d+', f)]
    latest_folder = max(folders, key=lambda x: int(re.search(r'\d+', x).group()))
    return latest_folder

# Training function
def train_model():
    print("Starting training...")
    os.system(f'python {yolov5_path}/train.py --img 640 --batch 4 --epochs 10 --data {data_yaml_path} --cfg {model_cfg_path} --weights {weights_path} --name animal_tag_detector --patience 10')
    print("Training completed")

# Validation function
def validate_model():
    print("Starting validation...")
    latest_folder = get_latest_detector_folder(f'{yolov5_path}/runs/train')
    weights_path = f'{yolov5_path}/runs/train/{latest_folder}/weights/best.pt'
    os.system(f'python {yolov5_path}/val.py --data {data_yaml_path} --weights {weights_path}')
    print("Validation completed")

# Inference function
def run_inference(image_folder, output_folder):
    latest_folder = get_latest_detector_folder(f'{yolov5_path}/runs/train')
    model_path = f'{yolov5_path}/runs/train/{latest_folder}/weights/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

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

            # Convert results to a DataFrame
            results_df = results.pandas().xyxy[0]

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
