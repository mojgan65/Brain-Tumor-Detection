**Brain Tumor Detection from MRI Dataset**

**Overview:**
This repository contains robust implementations for detecting brain tumors using MRI scans. By harnessing the power of deep learning and machine learning, we've demonstrated multiple methodologies to achieve this objective.

**Models Implemented:**
1. **Convolutional Neural Network (CNN):** A deep learning approach using convolutional layers to directly process the MRI images, capturing spatial hierarchies and details essential for accurate classification.
  
2. **CNN + Support Vector Machine (SVM):** A hybrid approach that leverages the feature extraction capabilities of CNNs, followed by SVM classification. This method harnesses the convolutional layers of CNNs to extract high-level features from the MRI scans, which are then fed into an SVM for the final classification task.

3. **MobileNet + SVM:** A lightweight and efficient model, MobileNet, is used to extract features which are then classified using SVM. This approach offers a balance between computational efficiency and accuracy, making it suitable for devices with limited computational resources.

**Dataset:**
The repository utilizes the Brain Tumor MRI Dataset, which contains labeled MRI scans that assist in training and validating the models. More details about the dataset, its origin, and its structure can be found [here]https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

**Contribute:**
Contributions are welcome! Whether it's improving model accuracy, computational efficiency, or adding new methodologies, feel free to raise an issue or submit a pull request.
