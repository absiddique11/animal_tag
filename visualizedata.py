import cv2
import matplotlib.pyplot as plt
import os

train_img_folder = '/home/mdabubakrsiddique/Documents/animal_tag/dataset/images/train' 

def visualize_annotations(image_folder, label_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            img = cv2.imread(image_path)
            height, width = img.shape[:2]

            with open(label_path, 'r') as f:
                labels = f.readlines()

            for label in labels:
                cls, x_center, y_center, w, h = map(float, label.split())
                x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
                x1, y1, x2, y2 = int(x_center - w / 2), int(y_center - h / 2), int(x_center + w / 2), int(y_center + h / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

visualize_annotations(train_img_folder, '/home/mdabubakrsiddique/Documents/animal_tag/dataset/labels/train')
