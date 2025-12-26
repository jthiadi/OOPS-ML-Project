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
