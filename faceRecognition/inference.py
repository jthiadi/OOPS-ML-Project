import argparse
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import face_recognition
import joblib
import numpy as np


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face recognition using a trained classifier.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("face_recognition_model.pkl"),
        help="Path to the trained classifier artifact.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=Path("label_encoder.pkl"),
        help="Path to the label encoder artifact.",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model to use with face_recognition.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.75,
        help="Minimum predicted probability required to accept a face as known.",
    )
    parser.add_argument(
        "--probability-margin",
        type=float,
        default=0.2,
        help="Require the top prediction to exceed the second-best by this margin, else mark as unknown.",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default="0",
        help="Index of the webcam (e.g., '0') or path to a video file.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.25,
        help="Scale factor applied to frames before running face detection (smaller is faster).",
    )
    parser.add_argument(
        "--hide-names",
        action="store_true",
        help="Do not overlay predicted names on the video stream.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Only run face detection every N frames to reduce lag (1 means every frame).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI display and print predictions to stdout only.",
    )
    return parser.parse_args(args=args)


def load_artifacts(model_path: Path, encoder_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
    classifier = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return classifier, label_encoder


def prepare_video_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source {source}")
    return capture


def classify_face(
    encoding: np.ndarray,
    classifier,
    label_encoder,
    probability_threshold: float,
    probability_margin: float,
) -> Tuple[str, float]:
    if not hasattr(classifier, "predict_proba"):
        raise AttributeError("Classifier must support predict_proba. Re-train with probability=True.")

    probabilities = classifier.predict_proba([encoding])[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    sorted_probs = np.sort(probabilities)
    second_best = float(sorted_probs[-2]) if sorted_probs.size >= 2 else 0.0
    margin = confidence - second_best

    if confidence < probability_threshold or margin < probability_margin:
        return "Unknown", confidence

    predicted_label = label_encoder.inverse_transform([best_index])[0]
    return str(predicted_label), confidence


def process_frame(
    frame_rgb: np.ndarray,
    resize_scale: float,
    detection_model: str,
    classifier,
    label_encoder,
    probability_threshold: float,
    probability_margin: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[str, float]]]:
    if resize_scale != 1.0:
        small_frame = cv2.resize(
            frame_rgb,
            (0, 0),
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        small_frame = frame_rgb

    face_locations_small = face_recognition.face_locations(
        small_frame,
        model=detection_model,
    )
    face_encodings = face_recognition.face_encodings(
        small_frame,
        face_locations_small,
    )

    predictions: List[Tuple[str, float]] = []
    for encoding in face_encodings:
        predictions.append(
            classify_face(
                encoding,
                classifier,
                label_encoder,
                probability_threshold,
                probability_margin,
            )
        )

    scale = 1.0 / resize_scale if resize_scale not in (0.0, 1.0) else 1.0
    face_locations = [
        (
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
            int(left * scale),
        )
        for top, right, bottom, left in face_locations_small
    ]

    return face_locations, predictions


def annotate_frame(
    frame: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    names_and_scores: Sequence[Tuple[str, float]],
) -> np.ndarray:
    for (top, right, bottom, left), (name, score) in zip(boxes, names_and_scores):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} ({score*100:.1f}%)"
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 4, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)

    try:
        classifier, label_encoder = load_artifacts(args.model_path, args.encoder_path)
    except (FileNotFoundError, AttributeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    capture = prepare_video_capture(args.video_source)
    print("[INFO] Press 'q' to quit.")

    frame_index = 0
    last_face_locations: List[Tuple[int, int, int, int]] = []
    last_predictions: List[Tuple[str, float]] = []
    last_logged_predictions: List[Tuple[str, float]] = []
    executor = ThreadPoolExecutor(max_workers=1)
    processing_future: Future | None = None

    try:
        frame_skip = max(1, args.frame_skip)
        while True:
            ret, frame = capture.read()
            if not ret:
                print("[WARN] Frame grab failed. Exiting.")
                break
            frame_index += 1

            process_this_frame = frame_index % frame_skip == 0

            if processing_future and processing_future.done():
                try:
                    last_face_locations, last_predictions = processing_future.result()
                    if last_predictions and last_predictions != last_logged_predictions:
                        for name, confidence in last_predictions:
                            print(f"[INFO] Detected {name} ({confidence*100:.1f}%)")
                        last_logged_predictions = last_predictions
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"[WARN] Face processing failed: {exc}")
                finally:
                    processing_future = None

            if process_this_frame and processing_future is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processing_future = executor.submit(
                    process_frame,
                    rgb_frame,
                    args.resize_scale,
                    args.detection_model,
                    classifier,
                    label_encoder,
                    args.probability_threshold,
                    args.probability_margin,
                )

            face_locations = last_face_locations
            predictions = last_predictions

            if not args.headless:
                if predictions:
                    if args.hide_names:
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    else:
                        annotate_frame(frame, face_locations, predictions)
                elif face_locations:
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()

