import cv2
import os
import time

def collect_face_dataset(mode="train"):
    """
    mode = "train" or "test"
    Train set = normal capturing session
    Test set = different day / lighting / angles for evaluation
    """

    # ================= CONFIG =================
    DATASET_DIR = "face_dataset"
    PEOPLE = ["Justin", "Mario", "Jennifer", "Edbert", "Steven", "Angel", "Unknown"]
    IMAGES_PER_PERSON = 250  # adjust depending on mode (test may need fewer)

    # ==========================================
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not video_capture.isOpened():
        print("❌ Cannot access camera!")
        return

    print("=" * 60)
    print(f"Face Dataset Collector ({mode.upper()} MODE)")
    print("=" * 60)
    print(f"Target: {IMAGES_PER_PERSON} images per person")
    print(f"Saving to folder: {DATASET_DIR}")
    print("=" * 60)

    for person in PEOPLE:
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)

        print(f"\n📸 Ready for: {person} ({mode.upper()})")
        print("   SPACE = Start/Stop | P = Pause | Q = Skip Person")
        print("   TIP: Change lighting / angle / glasses depending on mode")

        count = 0
        capturing = False
        paused = False

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                if capturing and not paused and count < IMAGES_PER_PERSON:
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    face_img = frame[y1:y2, x1:x2]

                    filename = os.path.join(person_dir, f"{person}_{count:04d}.jpg")
                    cv2.imwrite(filename, face_img)
                    count += 1
                    time.sleep(0.1)

            status_text = f"{mode.upper()} | {person}: {count}/{IMAGES_PER_PERSON}"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Dataset Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                capturing = not capturing
                paused = False
            elif key == ord('p'):
                if capturing:
                    paused = not paused
            elif key == ord('q'):
                break

            if count >= IMAGES_PER_PERSON:
                print(f"   ✅ Completed {person}")
                break

        time.sleep(1)

    video_capture.release()
    cv2.destroyAllWindows()
    print("\n✅ Dataset collection complete!")
    print(f"Saved to: {DATASET_DIR}")

if __name__ == "__main__":
    # ------- SELECT MODE HERE -------
    collect_face_dataset(mode="test")  # change to "test" when collecting test data
