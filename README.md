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
