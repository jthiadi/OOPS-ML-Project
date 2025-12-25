import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import face_recognition


def list_images(dataset_dir: Path) -> List[Tuple[Path, str]]:
    """Return (image_path, label) pairs for supported image files."""
    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    items: List[Tuple[Path, str]] = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in supported_suffixes:
                items.append((image_path, label))
    if not items:
        raise ValueError(f"No image files found in dataset directory {dataset_dir}")
    return items


def encode_dataset(
    images_and_labels: Sequence[Tuple[Path, str]],
    detection_model: str = "hog",
) -> Tuple[np.ndarray, List[str]]:
    """Convert image paths to face embeddings."""
    embeddings: List[np.ndarray] = []
    labels: List[str] = []

    for image_path, label in images_and_labels:
        image = face_recognition.load_image_file(str(image_path))
        face_locations = face_recognition.face_locations(image, model=detection_model)

        if not face_locations:
            print(f"[WARN] No face found in {image_path}. Skipping.")
            continue

        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            print(f"[WARN] Could not encode face in {image_path}. Skipping.")
            continue

        embeddings.append(face_encodings[0])
        labels.append(label)

    if not embeddings:
        raise ValueError("Failed to encode any faces. Check dataset quality.")

    return np.vstack(embeddings), labels


def train_classifier(
    embeddings: np.ndarray,
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[SVC, LabelEncoder, float, dict]:
    """Train an SVM classifier and return metrics."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    classifier = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val,
        y_pred,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )

    return classifier, label_encoder, accuracy, report


def save_artifacts(
    classifier: SVC,
    label_encoder: LabelEncoder,
    model_path: Path,
    encoder_path: Path,
    metrics_path: Path,
    accuracy: float,
    report: dict,
) -> None:
    """Persist model, label encoder, and metrics to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, model_path)
    joblib.dump(label_encoder, encoder_path)

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "accuracy": accuracy,
                "classification_report": report,
            },
            fp,
            indent=2,
        )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face recognition classifier.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("face_dataset"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model to use with face_recognition.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples held out for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("face_recognition_model.pkl"),
        help="Where to store the trained classifier.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=Path("label_encoder.pkl"),
        help="Where to store the label encoder.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("training_metrics.json"),
        help="Where to write training metrics.",
    )
    return parser.parse_args(args=args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

    print("[INFO] Loading dataset...")
    images_and_labels = list_images(args.dataset)
    unique_labels = sorted({label for _, label in images_and_labels})
    print(f"[INFO] Found {len(images_and_labels)} images across {len(unique_labels)} classes: {unique_labels}")

    print("[INFO] Encoding faces...")
    embeddings, labels = encode_dataset(images_and_labels, args.detection_model)
    print(f"[INFO] Encoded {len(embeddings)} faces.")

    print("[INFO] Training classifier...")
    classifier, label_encoder, accuracy, report = train_classifier(
        embeddings,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"[INFO] Validation accuracy: {accuracy:.3f}")

    print("[INFO] Saving artifacts...")
    save_artifacts(
        classifier,
        label_encoder,
        args.model_path,
        args.encoder_path,
        args.metrics_path,
        accuracy,
        report,
    )

    print(f"[INFO] Model saved to {args.model_path.resolve()}")
    print(f"[INFO] Label encoder saved to {args.encoder_path.resolve()}")
    print(f"[INFO] Metrics written to {args.metrics_path.resolve()}")


if __name__ == "__main__":
    main()

