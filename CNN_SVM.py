import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from skimage.feature import hog
from sklearn.svm import SVC


# Define data paths
train_data_dir = 'archive_2/Training'
test_data_dir = 'archive_2/Testing'

# Define image dimensions
img_width, img_height = 100, 100

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20, # Rotates the image by up to 20 degrees
    width_shift_range=0.1, # Shifts the image horizontally
    height_shift_range=0.1, # Shifts the image vertically
    horizontal_flip=True,
    vertical_flip=True # Adds vertical flipping
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Load testing data using flow_from_directory
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Define input shape
input_tensor = keras.Input(shape=(img_width, img_height, 1))

# Custom CNN architecture
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.GlobalAveragePooling2D()(x)

# Fully connected layers
x = keras.layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
y = keras.layers.Dense(4, activation='softmax', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)

# Define the model
model = keras.Model(input_tensor, y)

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs
)

cnn_predictions = model.predict(test_generator)

def extract_features(generator):
    features = []
    labels = []
    for i in range(len(generator)):
        batch_data, batch_labels = generator[i]
        for j in range(len(batch_data)):
            image = np.squeeze(batch_data[j]) # Remove extra dimensions
            image_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            features.append(image_features)
            labels.append(np.argmax(batch_labels[j]))
    return np.array(features), np.array(labels)


# Extract features and labels using HOG
train_features, train_labels = extract_features(train_generator)
test_features, test_labels = extract_features(test_generator)


# Train the SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(train_features, train_labels)

svm_predictions = svm.predict_proba(test_features)
# svm_predictions = svm.predict(test_features)

from scipy.stats import mode
# Stack predictions
combined_predictions = np.stack([cnn_predictions, svm_predictions])

# Take the mode along the stack to find the ensemble prediction
ensemble_predictions = mode(combined_predictions, axis=0)[0]

from sklearn.metrics import accuracy_score

ensemble_predictions = 0.8*svm_predictions + 0.2*cnn_predictions
ensemble_predictions = np.squeeze(ensemble_predictions)

# from keras.utils import to_categorical
# test_labels = to_categorical(test_labels,num_classes=4)
predicted_class = np.argmax(ensemble_predictions, axis=1)
# Assuming validation_labels contains the actual labels for the validation set
ensemble_accuracy = accuracy_score(test_labels, predicted_class)

print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')