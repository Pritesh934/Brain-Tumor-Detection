# Brain Tumor Detection Using CNN

A deep learning project aimed at automating the detection of brain tumors from Magnetic Resonance Imaging (MRI) scans using Convolutional Neural Networks (CNNs). This system is implemented in Python using the PyTorch framework.

# 📝 Project Overview
Medical image analysis plays a crucial role in early disease diagnosis. This project focuses on the binary classification of MRI images to identify the presence or absence of brain tumors. The model is designed to assist medical professionals by providing a rapid and automated diagnostic tool.

**Key Objectives:**

- Detect brain tumors from MRI images.

- Differentiate between "Healthy" (no tumor) and "Tumor" (yes tumor) classes.

- Implement a CNN architecture from scratch using PyTorch.

# 📂 Dataset

The project utilizes a dataset sourced from Kaggle, consisting of MRI scans categorized into two classes:

Yes: Images containing brain tumors (154 images).

No: Images of healthy brains (91 images).

Total Images: 245.

**Preprocessing:**

Images are resized to 128x128 pixels.

Data is split into training and validation sets to ensure robust model performance.

# 🛠 Technologies Used

- Language: Python 3

- Deep Learning Framework: PyTorch 

- Computer Vision: OpenCV (cv2) 

- Data Manipulation: NumPy 

- Visualization: Matplotlib 

- Metrics: Scikit-learn (Confusion Matrix, Accuracy Score)

# 🧠 Methodology

1) Data Loading: Images are loaded from the directory structure using glob and OpenCV.

2) Preprocessing: Images are resized and normalized.

3) Model Architecture: A Convolutional Neural Network (CNN) is designed to extract features from the MRI images and perform binary classification.

4) Training: The model is trained using a defined loss function (likely Binary Cross-Entropy) and optimizer over multiple epochs.

5) Evaluation: The model's performance is evaluated using accuracy metrics and confusion matrices.

# 🚀 Getting Started
**Prerequisites**

Install the required libraries: Bash: _pip install torch torchvision numpy opencv-python matplotlib scikit-learn_

**Usage**

1) Clone the repository.

2) Ensure the dataset is located in ./data/brain_tumor_dataset/ with subfolders yes and no.

3) Run the Jupyter Notebook to train the model and visualize results: Bash: _jupyter notebook Brain-Tumor-Detection.ipynb_

# 📊 Results

The project includes visualizations of the training process and prediction results:

- Sample Visualizations: Displays random batches of "Healthy" vs. "Tumor" MRI scans.

- Performance Metrics: Training/Validation loss graphs and confusion matrix analysis.
