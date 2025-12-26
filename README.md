# Object Ownership Project System (OOPS)

This project is an **Object Ownership Protection System** built as part of the *Introduction to Machine Learning* final project.  
It combines several **Machine Learning** components, including **Object Detection**, **Face Recognition**, **Pose Detection**, **Prediction**, combined with **FastAPI backend**, and a **Flutter web frontend**.

The system performs object-related ownership logic processing in the main code and seat state prediction using trained machine learning models. The functionality is exposed through an API that is consumed by a Flutter web application.
## Requirements

- Python 3.11+
- Flutter (with Chrome enabled)
- Internet connection (for model/API usage)

## Configuration

Before running the project, the following values must be configure:
- 'Main/api.py': Set the required API keys.
- 'Main/main.py': Configure camera input sources (IP cameras or USB camera indices).

# Project Inferance
```bash
cd Main/
```
## Setup Instructions

1. Create virtual environtment
```bash
python -m venv venv
```
2. Activate virtual environtment
```bash
venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the FastAPI backend
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
2. Run the Flutter web frontend
```bash
flutter run -d chrome
```
3. Run the main script
```bash
python main.py
```

# Left-Behind Model 
```bash
cd decision_model
```

1. Synthetic data generation
```bash
python generate_data.py
```
2. Prediction model training
```bash
python train.py
```

The model will be located in Main/decision_model/ as lost_item_rf_model.pkl

# Model and Training
```bash
cd ../Dataset
```

# Face Recognition
This includes the dataset collection, training, evaluation, and real-time inference using a webcam, for face recognition, before combining it with other models.

---

## Requirements
```bash
pip install face_recognition opencv-python numpy scikit-learn pandas joblib matplotlib
```
---

## Files Overview
```bash
face_dataset.py
```
Used to collect face images using a webcam and save them for training or testing
- Captures faces from webcam
- Saves images automatically

```bash
train.py
```
Used to train an SVM face recognition model.
- Extracts 128-D face embeddings
- Splits data into train/test
- Trains Linear SVM
- Saves trained model and label encoder
- Outputs training metrics

```bash
evaluate_metrics.py
```
Used to evaluate the trained model on a test dataset.
- Computes accuracy, F1-score, and confusion matrix
- Supports unknown face rejection using probability threshold
- Outputs results as JSON

```bash
inference.py
```
Used for real-time face recognition using webcam.
- Loads trained SVM model
- Displays name or "Unknown" on detected faces
- Uses confidence threshold and margin for unknown handling
- Optimized using frame skipping and resizing

```bash
illustration.py
```
Generates a visual comparison between KNN and SVM decision boundaries.
- Saves the figure as knn_vs_svm.png
 
## How to Run 
### 1, Collect Face Dataset
```bash
python face_dataset.py
```
Controls:
- SPACE – start / stop capturing images
- P – pause
- Q – skip current person

### 2. Train the Face Recognition Model
```bash
python train.py --dataset face_dataset
```
Outputs:
- face_recognition_model.pkl – trained SVM model
- label_encoder.pkl – label encoder
  
Note: This two outputs need to be moved inside Main/ directory of the project to be used

### 3. Evaluate the Model
```bash
python evaluate_metrics.py --test-dataset face_dataset_test
```
Output:
- svm_eval_results.json – evaluation metrics and confusion matrix

### 4. Run Inference
```bash
python inference.py
```

### 5. Generate Illustration
```bash
python illustration.py
```
Output:
- knn_vs_svm.png

# Object Recognition
This includes the dataset, training, evaluation, and real-time inference using a webcam, for face recognition, before combining it with other models.

## Files Overview
```bash
dataset/
```
Contains the object detection dataset structured in YOLO format.
- Includes training, validation, and test images
- Corresponding annotation files with bounding boxes and class labels
- Organized for direct use with Ultralytics YOLO

```bash
data_oops.yaml
```
Dataset configuration file for YOLO training.
- Defines dataset paths (train/val/test)
- Lists all object class names
- Used by YOLO during training and evaluation

```bash
training.py
```
Main script used to train the object detection model.
- Loads a pretrained YOLO model
- Trains on the custom dataset
- Configures epochs, image size, and training parameters
- Outputs training metrics and logs

```bash
delete.py
```
Used to clean the dataset by removing empty label files and their corresponding images.
- Detects empty or whitespace-only .txt files
- Deletes invalid label files
- Removes corresponding image files automatically
- Outputs a deletion summary

```bash
check_class_ids.py
```
Scans YOLO label files in train, valid, and test folders
- Reports which class IDs are present
- Checks consistency with expected class mappings
- Helps debug dataset labeling issues

```bash
label.py
```
Used to relabel YOLO annotation files by modifying class IDs in-place.
- Iterates through YOLO .txt label files
- Updates class IDs to ensure consistency
- Useful for fixing incorrect or inconsistent annotations

```bash
best.pt
```
Trained YOLO model weights.
- Best-performing checkpoint saved during training
- Used for inference and evaluation
- Can be loaded for real-time or offline object detection
  
Note: This pretrained output need to be moved inside Main/ directory of the project to be used

## Authors
1. A. M. Wijaya
2. E. L. Suryasatria
3. J. N. Hartono
4. J. Thiadi
5. M. A. Salim
6. S. Indrawan



