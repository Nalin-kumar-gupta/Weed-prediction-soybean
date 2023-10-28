import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

categories = ['broadleaf', 'grass', 'soil', 'soybean']
images=[]
labels=[]

base_dir = r'E:\vscode\Weed Detection-Soyabean ML\Project\resized_dataset'

for i, category in enumerate(categories):
    category_dir = os.path.join(base_dir, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.tif'):
            # Read image file and resize to target size
            image = cv2.imread(os.path.join(category_dir, filename))
            np_image=np.array(image)
            images.append(np_image)
            labels.append(np.array(i))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=80718)

print(images[0].shape)
print(labels[0])
print(X_train[0])
print(X_train[0].shape)
print(X_test[0])
print(X_test[0].shape)
print(y_train[0])
print(y_train[0].shape)
print(y_test[0])
print(y_test[0].shape)