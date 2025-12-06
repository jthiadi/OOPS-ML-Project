import cv2
import numpy as np
from deepface import DeepFace
import os
import pickle

def train_face_recognition():
    """
    Train face recognition model by extracting embeddings from dataset
    """
    dataset_path = "face_dataset"
    people = ["Justin", "Jennifer", "Mario"]
    
    # Dictionary to store embeddings for each person
    embeddings_db = {person: [] for person in people}
    
    print("Starting training process...")
    
    for person in people:
        person_folder = os.path.join(dataset_path, person)
        print(f"\nProcessing {person}...")
        
        # Get all images for this person
        images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(person_folder, img_name)
            
            try:
                # Extract face embedding using DeepFace
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    enforce_detection=False
                )[0]["embedding"]
                
                embeddings_db[person].append(embedding)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(images)} images")
                    
            except Exception as e:
                print(f"  Error processing {img_name}: {str(e)}")
                continue
        
        print(f"  Completed: {len(embeddings_db[person])} embeddings extracted")
    
    # Calculate average embedding for each person
    avg_embeddings = {}
    for person, embeddings in embeddings_db.items():
        if embeddings:
            avg_embeddings[person] = np.mean(embeddings, axis=0)
            print(f"\n{person}: Average embedding calculated from {len(embeddings)} images")
    
    # Save the trained model
    with open("face_recognition_model.pkl", "wb") as f:
        pickle.dump(avg_embeddings, f)
    
    print("\n✓ Training completed! Model saved as 'face_recognition_model.pkl'")
    print(f"Total people trained: {len(avg_embeddings)}")
    
    return avg_embeddings

if __name__ == "__main__":
    train_face_recognition()