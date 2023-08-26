from sklearn.metrics import classification_report
import seaborn as sns
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# Define data paths
train_data_dir = 'archive_2/Training'
test_data_dir = 'archive_2/Testing'

# Define image dimensions
img_width, img_height = 100, 100

# Hyperparameters
batch_size = 16
epochs = 30
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

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')


# Plot accuracy and loss during training
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions on the test set
y_pred = model.predict(test_generator)
y_pred_labels = np.argmax(y_pred, axis=1)

# Get the true labels for the test set
y_true_labels = test_generator.classes

# Calculate and print classification metrics
print('Classification Report:')
print(classification_report(y_true_labels, y_pred_labels))

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print('Confusion Matrix:')
print(conf_matrix)


# Get the true labels and predicted labels
y_true_labels = []
y_pred_labels = []
for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    y_true_labels.extend(np.argmax(y_batch, axis=1))
    y_pred_batch = model.predict(x_batch)
    y_pred_labels.extend(np.argmax(y_pred_batch, axis=1))

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix as a color plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
print(report)