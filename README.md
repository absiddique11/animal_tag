# Animal_tag_detection from Drone Images

Image datasets are created from videos captured by drones.

# Download and Install Yolo Models

## yolov8
git clone https://github.com/autogyro/yolo-V8 /home/mdabubakrsiddique/Documents/animal_tag/yolov8

cd /home/mdabubakrsiddique/Documents/animal_tag/yolov8

pip install -r /home/mdabubakrsiddique/Documents/animal_tag/yolov8/requirements.txt

## yolov5
git clone https://github.com/ultralytics/yolov5 /home/mdabubakrsiddique/Documents/animal_tag/yolov5

cd /home/mdabubakrsiddique/Documents/animal_tag/yolov5

pip install -r /home/mdabubakrsiddique/Documents/animal_tag/yolov5/requirements.txt


# Run Models
PYTHONWARNINGS="ignore" python3 train_and_infer_v8.py

PYTHONWARNINGS="ignore" python3 train_and_infer_v5.py



