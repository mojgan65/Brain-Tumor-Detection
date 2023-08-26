from sklearn.metrics import accuracy_score
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from tensorflow import keras
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import hinge_loss


# Define data paths
train_data_dir = 'archive_2/Training'
test_data_dir = 'archive_2/Testing'

# Define image dimensions
img_width, img_height = 100, 100
input_shape_rgb = (img_width, img_height, 3)  # 3 channels for RGB images

# Hyperparameters
batch_size = 16
epochs = 10

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale for testing set
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data and apply RGB conversion using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Load testing data and apply RGB conversion using flow_from_directory
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNetV2 model (with weights trained on ImageNet)
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape_rgb)

# Mapping grayscale to RGB
input_tensor = keras.Input(shape=(img_width, img_height, 1))
x = keras.layers.Conv2D(3, (3, 3), padding='same')(input_tensor)

# Followed by the MobileNetV2 model
x = mobilenet(x)

y = keras.layers.GlobalAveragePooling2D()(x)
y = keras.layers.Dense(256, activation='relu')(y)
y = keras.layers.Dense(4, activation='softmax')(y)  # 4 neurons for 4 tumor classes

m_model = keras.Model(input_tensor, y)

# Compile the model
m_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
m_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs
)
cnn_predictions = m_model.predict(test_generator)

# Evaluate the model on the test set
test_loss, test_acc = m_model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Mobilenet Test loss: {test_loss:.4f}')
print(f'Mobilenet Test accuracy: {test_acc:.4f}')


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

# Make predictions on the test set
test_predictions = svm.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f'SVM Test accuracy: {accuracy:.4f}')


svm_predictions = svm.predict_proba(test_features)

ensemble_predictions = 0.8*svm_predictions + 0.2*cnn_predictions
predicted_class = np.argmax(ensemble_predictions, axis=1)
# Assuming validation_labels contains the actual labels for the validation set
ensemble_accuracy = accuracy_score(test_labels, predicted_class)
print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')


# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_class)
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix as a color plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import classification_report

report = classification_report(test_labels, predicted_class, output_dict=True)
print(report)