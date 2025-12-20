import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import face_recognition


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_dataset(dataset_dir: Path, detection_model: str = "hog") -> Tuple[np.ndarray, List[str]]:
    """Load images from dataset_dir/<person>/* and compute face embeddings."""
    embeddings: List[np.ndarray] = []
    labels: List[str] = []

    class_dirs = [d for d in sorted(dataset_dir.iterdir()) if d.is_dir()]
    total_images = 0
    for d in class_dirs:
        total_images += sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    print(f"[INFO] Found {len(class_dirs)} classes with ~{total_images} total images")
    print("[INFO] Extracting embeddings...")

    processed = 0
    for class_dir in class_dirs:
        label = class_dir.name
        image_files = [f for f in sorted(class_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]

        for image_path in image_files:
            processed += 1
            try:
                img = face_recognition.load_image_file(str(image_path))
                locs = face_recognition.face_locations(img, model=detection_model)
                if not locs:
                    continue
                encs = face_recognition.face_encodings(img, locs)
                if not encs:
                    continue

                embeddings.append(encs[0])
                labels.append(label)

            except Exception as e:
                print(f"[WARN] Failed {image_path.name}: {e}")

            if processed % 50 == 0:
                print(f"  Processed {processed}/{total_images} images... (faces={len(embeddings)})")

    if not embeddings:
        raise ValueError("No face embeddings extracted. Check your dataset and detection model.")

    X = np.vstack(embeddings)
    print(f"[INFO] Done. Extracted {len(X)} embeddings from {processed} images.")
    return X, labels


def evaluate_svm_metrics(
    model,
    encoder: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    add_noise: bool = False,
    noise_level: float = 0.0,
    unknown_threshold: Optional[float] = None,
    min_support: int = 3,
    top_k_confusions: int = 5,
) -> Dict:
    """
    - Computes accuracy/macroF1/weightedF1 on KNOWN ground-truth samples only (y_test != -1)
    - Per-person F1 only for classes with support >= min_support
    - Top confusions (known->known)
    - If unknown_threshold is set, low-confidence predictions become -1 (unknown)
    - If y_test contains -1, computes unknown TPR and known false reject rate
    """

    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D (N,D). Got shape {X_test.shape}")
    if y_test.ndim != 1:
        raise ValueError(f"y_test must be 1D (N,). Got shape {y_test.shape}")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")

    X_eval = X_test.astype(np.float64, copy=True)
    y_test = y_test.astype(int, copy=False)

    if add_noise and noise_level > 0:
        noise = np.random.normal(0, noise_level, X_eval.shape)
        X_noisy = X_eval + noise
        orig_norm = np.linalg.norm(X_eval, axis=1, keepdims=True) + 1e-12
        new_norm = np.linalg.norm(X_noisy, axis=1, keepdims=True) + 1e-12
        X_eval = (X_noisy / new_norm) * orig_norm
        print(f"[INFO] Added noise to test embeddings (noise_level={noise_level})")

    class_names = np.array(list(encoder.classes_), dtype=str)
    K = len(class_names)
    known_labels = list(range(K))

    # ---- Predict (optionally unknown threshold) ----
    used_unknown = unknown_threshold is not None
    if not used_unknown:
        y_pred = model.predict(X_eval).astype(int)
        conf = None
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)
            conf = np.max(proba, axis=1)
            pred = np.argmax(proba, axis=1)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_eval)
            if scores.ndim == 1:
                conf = np.abs(scores)
                pred = (scores > 0).astype(int)
            else:
                conf = np.max(scores, axis=1)
                pred = np.argmax(scores, axis=1)
        else:
            raise ValueError("unknown_threshold set but model has no predict_proba or decision_function")

        y_pred = pred.astype(int)
        y_pred = y_pred.copy()
        y_pred[conf < unknown_threshold] = -1

    # ---- Masks ----
    known_gt_mask = (y_test >= 0) & (y_test < K)   # known ground truth
    unknown_gt_mask = (y_test == -1)               # unknown ground truth

    # ---- Overall metrics (known GT only) ----
    if np.any(known_gt_mask):
        yk_true = y_test[known_gt_mask]
        yk_pred = y_pred[known_gt_mask]

        accuracy_known = float(accuracy_score(yk_true, yk_pred))
        per_class_f1_all = f1_score(
            yk_true, yk_pred, labels=known_labels, average=None, zero_division=0
        )
        macro_f1_known = float(f1_score(
            yk_true, yk_pred, labels=known_labels, average="macro", zero_division=0
        ))
        weighted_f1_known = float(f1_score(
            yk_true, yk_pred, labels=known_labels, average="weighted", zero_division=0
        ))
    else:
        accuracy_known = None
        per_class_f1_all = np.zeros(K, dtype=float)
        macro_f1_known = None
        weighted_f1_known = None

    # ---- Per-person (support-aware) ----
    supports = np.array([int(np.sum(y_test[known_gt_mask] == i)) for i in known_labels], dtype=int)
    reported_idx = [i for i in known_labels if supports[i] >= min_support]

    per_person_f1 = {}
    for i in reported_idx:
        per_person_f1[class_names[i]] = {
            "f1_score": float(per_class_f1_all[i]),
            "support": int(supports[i]),
        }

    reported_f1s = np.array([per_class_f1_all[i] for i in reported_idx], dtype=float)
    per_person_summary = {
        "min_support": int(min_support),
        "num_people_in_encoder": int(K),
        "num_people_in_test": int(np.sum(supports > 0)),
        "num_people_reported": int(len(reported_idx)),
        "min_f1_reported": float(np.min(reported_f1s)) if reported_f1s.size else None,
        "median_f1_reported": float(np.median(reported_f1s)) if reported_f1s.size else None,
    }

    # ---- Confusion matrix (known GT only) ----
    include_unknown_col = used_unknown or np.any(y_pred == -1) or np.any(unknown_gt_mask)
    cm_labels = known_labels + ([-1] if include_unknown_col else [])
    cm = confusion_matrix(y_test[known_gt_mask], y_pred[known_gt_mask], labels=cm_labels)

    cm_names = list(class_names)
    if include_unknown_col:
        cm_names.append("unknown")

    # ---- Top confusions (known->known) ----
    top_confusions: List[Dict] = []
    if cm.shape[0] >= K and cm.shape[1] >= K:
        cm_known = cm[:K, :K].copy()
        np.fill_diagonal(cm_known, 0)
        pairs: List[Tuple[int, int, int]] = []
        for i in range(K):
            for j in range(K):
                v = int(cm_known[i, j])
                if v > 0:
                    pairs.append((v, i, j))
        pairs.sort(reverse=True)
        for v, i, j in pairs[:top_k_confusions]:
            top_confusions.append({"count": int(v), "true": class_names[i], "pred": class_names[j]})

    # ---- Open-set unknown metrics (only meaningful if y_test has -1) ----
    unknown_metrics = {
        "has_unknown_ground_truth": bool(np.any(unknown_gt_mask)),
        "unknown_threshold": unknown_threshold,
        "unknown_tpr": None,              # unknown GT predicted unknown
        "known_false_reject_rate": None,  # known GT predicted unknown
        "unknown_support": int(np.sum(unknown_gt_mask)),
        "known_support": int(np.sum(known_gt_mask)),
    }
    if include_unknown_col:
        if np.any(unknown_gt_mask):
            unk_pred = y_pred[unknown_gt_mask]
            unknown_metrics["unknown_tpr"] = float(np.mean(unk_pred == -1)) if unk_pred.size else None
        if np.any(known_gt_mask):
            known_pred = y_pred[known_gt_mask]
            unknown_metrics["known_false_reject_rate"] = float(np.mean(known_pred == -1)) if known_pred.size else None

    return {
        "accuracy_known": accuracy_known,
        "macro_f1_known": macro_f1_known,
        "weighted_f1_known": weighted_f1_known,
        "per_person_f1": per_person_f1,
        "per_person_summary": per_person_summary,
        "confusion_matrix_labels": cm_names,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions,
        "used_unknown_threshold": bool(used_unknown),
        "unknown_metrics": unknown_metrics,
        "counts": {
            "n_total": int(len(y_test)),
            "n_known_gt": int(np.sum(known_gt_mask)),
            "n_unknown_gt": int(np.sum(unknown_gt_mask)),
            "n_pred_unknown": int(np.sum(y_pred == -1)),
        },
    }


def print_report(metrics: Dict):
    print("\n" + "=" * 80)
    print("SVM FACE RECOGNITION — EVALUATION REPORT")
    print("=" * 80)

    print(f"Known-Only Accuracy : {metrics['accuracy_known']:.4f}" if metrics["accuracy_known"] is not None else "Known-Only Accuracy : N/A")
    print(f"Known-Only Macro F1 : {metrics['macro_f1_known']:.4f}" if metrics["macro_f1_known"] is not None else "Known-Only Macro F1 : N/A")
    print(f"Known-Only Wtd  F1  : {metrics['weighted_f1_known']:.4f}" if metrics["weighted_f1_known"] is not None else "Known-Only Wtd  F1  : N/A")

    s = metrics["per_person_summary"]
    print("\nPeople coverage:")
    print(f"  People in encoder  : {s['num_people_in_encoder']}")
    print(f"  People in test     : {s['num_people_in_test']}")
    print(f"  Reported (support>={s['min_support']}): {s['num_people_reported']}")
    print(f"  Min F1 (reported)  : {s['min_f1_reported']}")
    print(f"  Med F1 (reported)  : {s['median_f1_reported']}")

    um = metrics["unknown_metrics"]
    if metrics["used_unknown_threshold"]:
        print("\nUnknown threshold:")
        print(f"  threshold: {um['unknown_threshold']}")
    if um["has_unknown_ground_truth"]:
        print("\nUnknown metrics (requires y_test = -1 samples):")
        print(f"  unknown support            : {um['unknown_support']}")
        print(f"  unknown TPR (reject unknown): {um['unknown_tpr']}")
        print(f"  known false reject rate     : {um['known_false_reject_rate']}")

    if metrics["top_confusions"]:
        print("\nTop confusions (known→known):")
        for c in metrics["top_confusions"]:
            print(f"  {c['true']} → {c['pred']}: {c['count']}")

    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained SVM face recognition model")
    ap.add_argument("--test-dataset", type=Path, required=True, help="Test dataset dir: test/<person>/*.jpg")
    ap.add_argument("--svm-model", type=Path, default=Path("face_recognition_model.pkl"), help="Path to trained SVM model")
    ap.add_argument("--encoder", type=Path, default=Path("label_encoder.pkl"), help="Path to label encoder")
    ap.add_argument("--detection-model", choices=["hog", "cnn"], default="hog", help="Face detector backend")
    ap.add_argument("--add-noise", action="store_true", help="Add Gaussian noise to embeddings")
    ap.add_argument("--noise-level", type=float, default=0.08, help="Noise level if add-noise")
    ap.add_argument("--unknown-threshold", type=float, default=None, help="If set, low-confidence => unknown (-1)")
    ap.add_argument("--min-support", type=int, default=3, help="Only report per-person if support >= this")
    ap.add_argument("--output-json", type=Path, default=Path("svm_eval_results.json"), help="Save metrics JSON")
    args = ap.parse_args()

    if not args.test_dataset.exists():
        raise FileNotFoundError(f"Test dataset not found: {args.test_dataset}")

    print("[INFO] Loading encoder...")
    encoder: LabelEncoder = joblib.load(args.encoder)

    print(f"[INFO] Loading test dataset: {args.test_dataset}")
    X_test, test_labels = load_dataset(args.test_dataset, args.detection_model)

    # --- Map labels: keep unknown folders as -1 (recommended) ---
    train_classes = set(encoder.classes_)
    y_test_list: List[int] = []
    for lab in test_labels:
        if lab in train_classes:
            y_test_list.append(int(encoder.transform([lab])[0]))
        else:
            y_test_list.append(-1)
    y_test = np.array(y_test_list, dtype=int)

    print("[INFO] Loading SVM model...")
    svm_model = joblib.load(args.svm_model)

    metrics = evaluate_svm_metrics(
        svm_model,
        encoder,
        X_test,
        y_test,
        add_noise=args.add_noise,
        noise_level=args.noise_level,
        unknown_threshold=args.unknown_threshold,
        min_support=args.min_support,
    )

    print_report(metrics)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
