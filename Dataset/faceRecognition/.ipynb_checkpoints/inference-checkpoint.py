import cv2
import numpy as np
from deepface import DeepFace
import pickle
import time

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_model():
    """Load the trained face recognition model"""
    try:
        with open("face_recognition_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        return model
    except FileNotFoundError:
        print("❌ Model file not found. Please run training first.")
        return None

def recognize_face(frame, model, threshold=0.7):
    """
    Recognize face in the frame
    Returns: (name, similarity_score, bbox)
    """
    try:
        # Detect and extract face embedding
        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        if not result:
            return None, 0, None
        
        # Get embedding and face region
        embedding = result[0]["embedding"]
        face_region = result[0]["facial_area"]
        
        # Compare with all known faces
        best_match = None
        best_similarity = 0
        
        for person, avg_embedding in model.items():
            similarity = cosine_similarity(embedding, avg_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person
        
        # Convert face region to bbox format
        bbox = (face_region['x'], face_region['y'], 
                face_region['w'], face_region['h'])
        
        return best_match, best_similarity, bbox
        
    except Exception as e:
        return None, 0, None

def main():
    """Main function for real-time face recognition"""
    # Load trained model
    model = load_model()
    if model is None:
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎥 Camera started. Press 'q' to quit.")
    print(f"Recognition threshold: 0.7")
    print("System will stop detecting once confidence >= 0.7\n")
    
    # State variables
    recognition_locked = False
    locked_person = None
    locked_time = None
    frame_skip = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for performance (process every 3rd frame)
        frame_skip += 1
        if frame_skip % 3 != 0 and not recognition_locked:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # If recognition is locked, just display the result
        if recognition_locked:
            # Display locked recognition
            text = f"✓ {locked_person} (LOCKED)"
            cv2.rectangle(frame, (10, 10), (630, 60), (0, 255, 0), -1)
            cv2.putText(frame, text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Perform face recognition
        name, similarity, bbox = recognize_face(frame, model, threshold=0.7)
        
        if name and bbox:
            x, y, w, h = bbox
            
            # Check if similarity meets threshold
            if similarity >= 0.7:
                # Lock the recognition
                recognition_locked = True
                locked_person = name
                locked_time = time.time()
                color = (0, 255, 0)  # Green
                label = f"{name}: {similarity:.2f} - CONFIRMED!"
                print(f"\n✓ Recognition locked: {name} (confidence: {similarity:.2f})")
            else:
                color = (0, 165, 255)  # Orange
                label = f"{name}: {similarity:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y-35), (x+label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            # No face detected or recognized
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Camera closed")

if __name__ == "__main__":
    main()