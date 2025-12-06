import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import pickle
import face_recognition

class FaceRecognitionSystem:
    def __init__(self, dataset_path='face_dataset'):
        self.dataset_path = dataset_path
        self.recognizer = None
        self.label_encoder = LabelEncoder()
        self.known_names = ['Justin', 'Jennifer', 'Mario', 'Steven', 'Angel', 'Edbert']
        self.face_encodings = []
        self.face_labels = []
        
    def load_dataset(self):
        """Load images and extract face encodings"""
        X = []
        labels = []
        
        print("Loading dataset and extracting face encodings...")
        for person_name in self.known_names:
            person_path = os.path.join(self.dataset_path, person_name)
            
            if not os.path.exists(person_path):
                print(f"Warning: {person_path} not found!")
                continue
            
            img_files = [f for f in os.listdir(person_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {len(img_files)} images for {person_name}...")
            
            for idx, img_file in enumerate(img_files):
                img_path = os.path.join(person_path, img_file)
                
                # Load image
                img = face_recognition.load_image_file(img_path)
                
                # Get face encodings (128-dimensional face embeddings)
                face_encodings = face_recognition.face_encodings(img)
                
                if len(face_encodings) > 0:
                    # Take the first face found
                    encoding = face_encodings[0]
                    X.append(encoding)
                    labels.append(person_name)
                    
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(img_files)} images...")
        
        print(f"Total faces loaded: {len(X)}")
        return np.array(X), np.array(labels)
    
    def train(self):
        """Train the face recognition model"""
        X, y = self.load_dataset()
        
        if len(X) == 0:
            raise ValueError("No training data found!")
        
        print(f"\nTraining on {len(X)} face samples...")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Store encodings for direct matching
        self.face_encodings = X
        self.face_labels = y
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train SVM classifier
        print("\nTraining SVM classifier...")
        self.recognizer = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        self.recognizer.fit(X_train, y_train)
        
        # Calculate validation accuracy
        val_pred = self.recognizer.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        print(f"Validation accuracy: {val_acc:.2%}")
        
        # Calculate mAP on validation set
        y_proba = self.recognizer.predict_proba(X_val)
        
        # One-vs-rest AP for each class
        aps = []
        print("\nAverage Precision per class:")
        for i, name in enumerate(self.label_encoder.classes_):
            y_true_binary = (y_val == i).astype(int)
            y_score = y_proba[:, i]
            ap = average_precision_score(y_true_binary, y_score)
            aps.append(ap)
            print(f"  {self.label_encoder.inverse_transform([i])[0]}: {ap:.4f}")
        
        mean_ap = np.mean(aps)
        print(f"\nmAP Score: {mean_ap:.4f}")
        
        return mean_ap
    
    def save_model(self, model_path='face_model.pkl'):
        """Save trained model"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'recognizer': self.recognizer,
                'label_encoder': self.label_encoder,
                'face_encodings': self.face_encodings,
                'face_labels': self.face_labels
            }, f)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='face_model.pkl'):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.recognizer = data['recognizer']
            self.label_encoder = data['label_encoder']
            self.face_encodings = data['face_encodings']
            self.face_labels = data['face_labels']
        print(f"Model loaded from {model_path}")
    
    def recognize_face(self, face_encoding, threshold=0.55):
        """Recognize a face from its encoding"""
        if face_encoding is None or len(face_encoding) == 0:
            return "Unknown", 0.0
        
        face_encoding = face_encoding.reshape(1, -1)
        
        # Get prediction probabilities from SVM
        proba = self.recognizer.predict_proba(face_encoding)[0]
        max_proba = np.max(proba)
        pred_label = self.recognizer.predict(face_encoding)[0]
        
        # Additional verification: compare with known face encodings
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(self.face_encodings, face_encoding[0])
        min_distance = np.min(distances)
        
        # Combine SVM probability and face distance
        # Lower distance = more similar (threshold ~0.6)
        if max_proba < threshold or min_distance > 0.6:
            return "Unknown", max_proba
        
        pred_name = self.label_encoder.inverse_transform([pred_label])[0]
        
        return pred_name, max_proba
    
    def real_time_recognition(self):
        """Real-time face recognition from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nStarting real-time face recognition...")
        print("Press 'q' to quit")
        
        # Process every N frames for performance
        process_frame = 0
        face_locations = []
        face_encodings_list = []
        face_names = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Only process every 3rd frame
            if process_frame % 3 == 0:
                # Find faces and encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings_list = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings_list:
                    name, confidence = self.recognize_face(face_encoding)
                    face_names.append((name, confidence))
            
            process_frame += 1
            
            # Draw results on full-size frame
            for (top, right, bottom, left), (name, conf) in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                label = f"{name} ({conf:.2f})"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Initialize system
    fr_system = FaceRecognitionSystem(dataset_path='face_dataset')
    
    # Train the model
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    map_score = fr_system.train()
    
    # Save model
    fr_system.save_model('face_model.pkl')
    
    # Start real-time recognition
    print("\n" + "=" * 60)
    print("INFERENCE PHASE")
    print("=" * 60)
    fr_system.real_time_recognition()


if __name__ == "__main__":
    main()