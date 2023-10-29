import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
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



x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=80718)

X_train=np.array(x_train)
X_test=np.array(x_test)
Y_train=np.array(y_train)
Y_test=np.array(y_test)



model = keras.Sequential()
num_channels = 3
image_width = 200
image_height = 200
num_classes = 4

# Add the input layer
model.add(layers.Input(shape=(image_height, image_width, num_channels)))  # Specify your input shape

# Add convolutional layers

# For Conv2D layers:
# 32: This argument specifies the number of filters (also known as kernels) that the layer will use to convolve the input. In this case, there are 32 filters. Each filter will detect different features in the input data.
# (3, 3): This argument specifies the size of the convolutional kernel or filter. It's a 3x3 filter, meaning it slides over the input data in a 3x3 grid to perform convolutions.
# activation='relu': This argument specifies the activation function used in the layer. In this case, Rectified Linear Unit (ReLU) is used as the activation function.



model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# Flatten the output for the fully connected layers
model.add(layers.Flatten())

# Add fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Optional dropout layer for regularization
model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer, num_classes is the number of classes in your problem



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()


# Assuming you've prepared your dataset and one-hot encoded the labels

# Set hyperparameters
batch_size = 32
epochs = 15

# Training the model
history = model.fit(
    X_train,  # Your training data
    Y_train,  # One-hot encoded training labels
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, Y_test)  # Your validation data
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)  # Your test data and one-hot encoded test labels

print(f"Test accuracy: {test_accuracy}")