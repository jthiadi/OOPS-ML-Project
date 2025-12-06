import os
import numpy as np
import cv2
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (160, 160)
KNOWN_CLASSES = ['Justin', 'Jennifer', 'Mario']

def load_test_data(dataset_path, label_encoder):
    """Load test data for evaluation"""
    images = []
    labels = []
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
            
        if person_name not in KNOWN_CLASSES:
            continue
            
        print(f"Loading {person_name} for testing...")
        img_files = os.listdir(person_path)
        
        # Use last 20% as test set (or adjust as needed)
        test_files = img_files[-int(len(img_files) * 0.2):]
        
        for img_name in test_files:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Detect and crop face
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
            else:
                face = img
            
            face = cv2.resize(face, IMG_SIZE)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32') / 255.0
            
            images.append(face)
            labels.append(person_name)
    
    return np.array(images), np.array(labels)

def calculate_map(y_true, y_pred_proba, label_encoder):
    """Calculate mean Average Precision for each class"""
    n_classes = len(label_encoder.classes_)
    y_true_encoded = label_encoder.transform(y_true)
    
    # Convert to binary format
    y_true_binary = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true_encoded):
        y_true_binary[i, label] = 1
    
    # Calculate AP for each class
    aps = []
    print("\n" + "="*50)
    print("Average Precision (AP) per class:")
    print("="*50)
    
    for i, class_name in enumerate(label_encoder.classes_):
        ap = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])
        aps.append(ap)
        print(f"{class_name:15s}: {ap:.4f}")
    
    map_score = np.mean(aps)
    print("="*50)
    print(f"{'mAP':15s}: {map_score:.4f}")
    print("="*50)
    
    return map_score, aps

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

def plot_precision_per_class(aps, label_encoder):
    """Plot AP per class"""
    plt.figure(figsize=(10, 6))
    classes = label_encoder.classes_
    x_pos = np.arange(len(classes))
    
    plt.bar(x_pos, aps, color='steelblue', alpha=0.8)
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.title('Average Precision per Class')
    plt.xticks(x_pos, classes)
    plt.ylim([0, 1.0])
    
    for i, v in enumerate(aps):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ap_per_class.png', dpi=300, bbox_inches='tight')
    print("AP per class plot saved as 'ap_per_class.png'")

def main():
    print("Loading model and label encoder...")
    model = tf.keras.models.load_model('face_recognition_model.h5')
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("\nLoading test data...")
    X_test, y_test = load_test_data('face_dataset', label_encoder)
    
    if len(X_test) == 0:
        print("No test data found!")
        return
    
    print(f"Loaded {len(X_test)} test images")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Calculate mAP
    map_score, aps = calculate_map(y_test, y_pred_proba, label_encoder)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_precision_per_class(aps, label_encoder)
    
    # Save results to file
    with open('evaluation_results.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("Face Recognition Model Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total test samples: {len(X_test)}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        f.write("\n" + "="*50 + "\n")
        f.write("Average Precision per class:\n")
        f.write("="*50 + "\n")
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_name:15s}: {aps[i]:.4f}\n")
        f.write("="*50 + "\n")
        f.write(f"{'mAP':15s}: {map_score:.4f}\n")
        f.write("="*50 + "\n")
    
    print("\nResults saved to 'evaluation_results.txt'")
    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()