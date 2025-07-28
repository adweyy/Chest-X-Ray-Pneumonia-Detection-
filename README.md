# Chest-X-Ray-Pneumonia-Detection-
Chest X-Ray Pneumonia Detection Using Deep Learning
This project implements a deep learning model to classify chest X-ray images as either NORMAL or PNEUMONIA using a pre-trained ResNet-18 architecture. The model is fine-tuned with Adam and SGD optimizers, achieving up to 87.98% accuracy on the test set. The project leverages the Kaggle Chest X-Ray Images (Pneumonia) dataset and is implemented in PyTorch.
Project Overview
The goal is to develop a convolutional neural network (CNN) to assist in detecting pneumonia from chest X-ray images, potentially aiding radiologists in early diagnosis. The model uses transfer learning with ResNet-18, modified to handle grayscale images converted to RGB format, and is trained with weighted cross-entropy loss to address class imbalance.
Dataset

Source: Chest X-Ray Images (Pneumonia)
Size:
Training: 5,216 images
Validation: 16 images
Test: 624 images


Classes: NORMAL, PNEUMONIA
Preprocessing:
Training: Resize to 224x224, random horizontal flips, random rotations (10°), normalization (mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
Validation/Test: Resize to 224x224, normalization



Model

Architecture: Pre-trained ResNet-18 from torchvision.models
Modifications:
First convolutional layer adjusted for 3-channel input (grayscale images converted to RGB)
Fully connected layer modified for 2-class output (NORMAL, PNEUMONIA)


Optimizers:
Adam: Learning rate 0.001, weight decay 1e-4
SGD: Learning rate 0.01, momentum 0.9, weight decay 1e-4, cosine annealing scheduler


Loss Function: Weighted cross-entropy to handle class imbalance
Device: GPU (if available) or CPU

Results



Optimizer
Accuracy
F1-Score
AUC-ROC
Confusion Matrix



Adam
86.54%
89.80%
93.45%
[[208, 26], [58, 332]]


SGD
87.98%
90.91%
94.78%
[[216, 18], [57, 333]]


SGD with cosine annealing outperforms Adam, showing better generalization due to momentum and learning rate scheduling.
Visuals

Adam Loss Curve:
SGD Loss Curve:

Installation
To run this project locally, follow these steps:
Prerequisites

Python 3.8 or higher
Git
CUDA-enabled GPU (optional, for faster training)

Steps to Run Locally

Clone the Repository:
git clone https://github.com/your-username/Chest-XRay-Pneumonia-Detection.git
cd Chest-XRay-Pneumonia-Detection


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the Dataset:

Download the dataset from Kaggle.
Extract the dataset to a local directory (e.g., ./chest_xray_dataset).
Update the data_dir variable in chestxray.py to point to the extracted dataset path:data_dir = "./chest_xray_dataset"  # Replace with your dataset path




Run the Script:
python chestxray.py


This will train the model with both Adam and SGD optimizers for 10 epochs each, generate loss curves, and evaluate performance on the test set.
Outputs include:
Training and validation loss per epoch
Test set metrics (accuracy, F1-score, AUC-ROC, confusion matrix)
Loss curve plots saved as adam_loss_curves.png and sgd_loss_curves.png




Verify Outputs:

Check the console for training progress and final metrics.
View the generated loss curve images in the project directory.



Notes

Dataset Path: Ensure the dataset directory structure matches /chest_xray/train, /chest_xray/val, and /chest_xray/test.
Large Files: Trained model weights are not included due to size constraints. To use pre-trained weights, contact the repository owner or retrain the model.
Hardware: Training on a CPU is possible but slower. A CUDA-enabled GPU is recommended for faster computation.

License
MIT License
Credits

Dataset: Kermany et al., 2018
Tools: PyTorch, Google Colab
Author: Adwaith Arun
