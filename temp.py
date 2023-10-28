
from PIL import Image
import os
import random
import numpy as np

# Define the path to your source dataset
source_dataset_dir = r'E:\vscode\Weed Detection-Soyabean ML\Project\resized_dataset'  # Replace with the path to your source dataset

# Define the categories (subdirectories) in your dataset
categories = ['broadleaf', 'grass', 'soil', 'soybean']

# Define the ratio for splitting (e.g., 80% train, 20% test)
train_ratio = 0.8
test_ratio = 0.2

# Initialize lists to store file paths and labels for each set
train_files = []
test_files = []

train_labels = []
test_labels = []

# Split the dataset into training and test sets
for category in categories:
    category_dir = os.path.join(source_dataset_dir, category)
    image_files = os.listdir(category_dir)
    random.shuffle(image_files)  # Shuffle to randomize the order

    train_split = int(train_ratio * len(image_files))

    train_files.extend([os.path.join(category_dir, filename) for filename in image_files[:train_split]])
    test_files.extend([os.path.join(category_dir, filename) for filename in image_files[train_split:]])

    train_labels.extend([categories.index(category)] * train_split)
    test_labels.extend([categories.index(category)] * (len(image_files) - train_split))

# Now, convert the lists to NumPy arrays
X_train = np.array([np.array(Image.open(file)) for file in train_files])
X_test = np.array([np.array(Image.open(file)) for file in test_files])
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
