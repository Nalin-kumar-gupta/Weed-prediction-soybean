import cv2
import os

target_size = (200, 200)  # Change to the desired dimensions

categories = ['broadleaf', 'grass', 'soil', 'soybean']
images = []
labels = []

source_directory = r'E:\vscode\Weed Detection-Soyabean ML\Project\dataset'  # Replace with the path to your source dataset
destination_directory = r'E:\vscode\Weed Detection-Soyabean ML\Project\resized_dataset'  # Replace with the path to your destination dataset


for i, category in enumerate(categories):
    category_dir = os.path.join(source_directory, category)
    new_dir = os.path.join(destination_directory, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.tif'):
            # Read image file and resize to target size
            image = cv2.imread(os.path.join(category_dir, filename))
            image = cv2.resize(image, target_size)
            
            # Append image and label data to lists
            os.makedirs(new_dir, exist_ok=True)
            cv2.imwrite(os.path.join(new_dir, filename), image)
            
print("Dataset resized successfully.")