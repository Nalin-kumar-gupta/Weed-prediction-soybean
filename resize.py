import cv2
import os
import numpy as np



target_size = (200, 200)  # Change to the desired dimensions

categories = ['broadleaf', 'grass', 'soil', 'soybean']
images = []
labels = []



source_directory = r'E:\vscode\Weed Detection-Soyabean ML\Project\dataset'

for i, category in enumerate(categories):
    category_dir = os.path.join(source_directory, category)
#     new_dir = os.path.join(destination_directory, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.tif'):
            # Read image file and resize to target size
            image = cv2.imread(os.path.join(category_dir, filename))
            image = cv2.resize(image, target_size)
            image=np.array(image)
            images.append(image)
            ohe=np.zeros(len(categories))
            ohe[i]=1
            labels.append(ohe)
print("resized sussesfully")

