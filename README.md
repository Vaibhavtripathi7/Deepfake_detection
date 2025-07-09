# Deepfake Video Detection Project
---
<img src="Result Images/deepfake_cover_page.jpg" alt="" > 
---

This project implements a deep learning model to detect deepfake videos. It utilizes a ResNeXt50 convolutional neural network (CNN) for spatial feature extraction from video frames and a Long Short-Term Memory (LSTM) network to capture temporal dependencies across frames.

The model was trained using a dataset of real and fake face videos processed with MediaPipe for face extraction. This project is structured for local development and execution (e.g., on a local machine with PyCharm).

## Table of Contents
1.  [Overview](#overview)
2.  [Model Architecture](#model-architecture)
3.  [Dataset](#dataset)
4.  [Setup and Installation](#setup-and-installation)
5.  [Data Preprocessing](#data-preprocessing)
6.  [Usage](#usage)
    *   [Training](#training)
    *   [Evaluation](#evaluation)
7.  [Results](#results)
    *   [Training Performance](#training-performance)
    *   [Test Set Evaluation](#test-set-evaluation)

## Overview
The goal of this project is to build an effective deepfake detector that can distinguish between authentic videos and videos manipulated using deep learning techniques. The model processes sequences of video frames to make its predictions.

## Model Architecture
The `DeepFakeDetector` model consists of:
1.  **CNN Feature Extractor**: A pre-trained ResNeXt50-32x4d model (weights from ImageNet) is used to extract spatial features from individual video frames. The final classification layer of ResNeXt50 is removed, and features are taken from the penultimate layer.
2.  **Adaptive Average Pooling**: An `AdaptiveAvgPool2d((1, 1))` layer is applied to the CNN feature maps.
3.  **LSTM Layer**: An LSTM network processes the sequence of frame features to model temporal relationships.
    *   Input Size: 2048 (output dimension of ResNeXt50 feature extractor)
    *   Hidden Dimension: 64
    *   Number of Layers: 1
    *   Bidirectional: False
4.  **Dropout**: A dropout layer (rate 0.8) is applied after the LSTM to prevent overfitting.
5.  **Fully Connected Layer**: A final linear layer maps the LSTM output to the two classes (REAL, FAKE).

## Dataset
The model requires a dataset of `.mp4` videos of faces, organized into `train/real_faces/`, `train/fake_faces/`, `test/real_faces/`, and `test/fake_faces/` subdirectories. The preprocessing step (described below) is responsible for generating this dataset from raw video files.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Python Version**: Python 3.11 or similar.
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio
    pip install scikit-learn opencv-python numpy matplotlib tqdm seaborn pandas mediapipe
    ```
    (Ensure you install the correct PyTorch version for your CUDA setup if using a GPU).

## Data Preprocessing
Before training the model, raw video data needs to be processed to extract face crops. This is handled by the `preprocessing.py` script.

**Purpose**:
The script takes raw videos as input, detects faces in each frame using MediaPipe, crops these faces, resizes them, and saves them as new, shorter video clips containing only the processed face frames.

**Key Steps**:
1.  **Frame Extraction**: Extracts individual frames from the input videos.
2.  **Face Detection**: Uses MediaPipe's Face Detection solution to identify faces in each frame.
3.  **Face Cropping & Resizing**: Crops the detected face region and resizes it to a standard size (`OUTPUT_FACE_VIDEO_SIZE`).
4.  **Video Creation**: Writes the processed face frames into a new `.mp4` video file.
5.  **Limited Frames**: Processes a limited number of frames per video (`FRAMES_TO_PROCESS_PER_VIDEO`) to manage processing time and dataset size.

**Configuration**:
The script uses the following important constants defined at the top of `preprocessing.py`:
*   `RAW_DATASET_BASE_PATH`: Path to your raw dataset (e.g., `/home/vaibhav/PROJECTS/deepfake_dataset`). This directory should contain `train/real`, `train/fake`, `test/real`, and `test/fake` subfolders with your original `.mp4` videos.
*   `PROCESSED_DATASET_BASE_PATH`: Path where the processed face videos will be saved (e.g., `/home/vaibhav/PROJECTS/processed_dataset_faces_mediapipe`). The script will create `train/real_faces`, `train/fake_faces`, etc., subfolders here.
*   `OUTPUT_FACE_VIDEO_SIZE`: Target size for the cropped face videos (default: 112x112).
*   `FRAMES_TO_PROCESS_PER_VIDEO`: Maximum number of frames to process from each input video (default: 150).
*   `MIN_DETECTION_CONFIDENCE`: Minimum confidence for MediaPipe face detection (default: 0.5).

**How to Run**:
1.  Ensure your raw videos are organized correctly under `RAW_DATASET_BASE_PATH`.
2.  Modify the `RAW_DATASET_BASE_PATH` and `PROCESSED_DATASET_BASE_PATH` variables in `preprocessing.py` to match your local setup.
3.  Execute the script:
    ```bash
    python preprocessing.py
    ```
The script will then process the videos and save the face-cropped versions to the `PROCESSED_DATASET_BASE_PATH`. This processed dataset is what the training script will use.

## Usage
After preprocessing your data, you can proceed with training and evaluation.

### Training

The training script (`model.py`) in its original form, which you might convert to (`train.py`) performs the following:
*   **Hyperparameters**:
    *   Sequence Length: 20 frames
    *   Image Size: 112x112 pixels (should match `OUTPUT_FACE_VIDEO_SIZE` from preprocessing)
    *   Batch Size: 8
    *   Number of Epochs: 10
    *   Learning Rate: 1e-4
    *   Weight Decay: 1e-4
    *   Validation Split: 20% of training data
*   **Data Augmentation (for training)**: Random crop, random horizontal flip, color jitter.
*   **Optimizer**: Adam
*   **Loss Function**: CrossEntropyLoss
*   The script saves the model with the best validation AUC and the model from the last epoch to the `models/` directory

To train locally (assuming you have a `model.py` and your processed data is at `PROCESSED_DATASET_BASE_PATH`):
```bash
python train.py --data_path /path/to/your/processed_dataset_faces_mediapipe --model_save_path ./models
```

## Results
The model was trained for 7.54 minutes using GPU acceleration.

### Training Performance
*   **Training Samples**: 681
*   **Validation Samples**: 171
*   **Best Validation AUC**: 0.9034
*   The training progress plots (Loss, Accuracy, AUC) are shown below:

  
  <img src="Result Images/Untitled.png" alt="" > 


### Test Set Evaluation
*   **Total Test Samples**: 61
*   **Metrics**:
    *   Accuracy: 72.13%
    *   Precision (for REAL class): 0.6471
    *   Recall (for REAL class): 0.8148
    *   F1-Score (for REAL class): 0.7213
    *   AUC: 0.8028
      

      <img src="Result Images/Screenshot_20250601_024157.png" alt="" > 

    
*   **Confusion Matrix**:

    
     <img src="Result Images/result_72.png" alt="" > 
    
    
    The matrix shows the following distribution of predictions:

    |             | Predicted FAKE (0) | Predicted REAL (1) |
    |-------------|--------------------|--------------------|
    | True FAKE (0) | 22                 | 12                 |
    | True REAL (1) | 5                  | 22                 |
