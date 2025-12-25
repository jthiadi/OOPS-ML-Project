import os
import csv
import time
import datetime
import sqlite3

import cv2
import numpy as np
import torch

# Depth Anything (MiDaS-style depth)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# -------------------------------------------------------------------
# PATHS / CONSTANTS
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEFT_DB_PATH = os.path.join(BASE_DIR, "left_items.db")
      # SQLite db for left-behind items
LIVE_LOG_PATH = "live_log.txt"      # text log for live events
SUMMARY_LOG_PATH = "summary_log.txt"
TABLE_ID = "T1"                     # default table ID; you can change per camera if needed


# -------------------------------------------------------------------
# DEPTH ESTIMATOR (MiDaS / Depth-Anything)
# -------------------------------------------------------------------
class DepthEstimator:
    """
    Simple wrapper around LiheYoung/depth-anything-small-hf.

    Usage:
        depth_est = DepthEstimator(device="cuda" or "cpu")
        depth_map = depth_est.predict(bgr_frame)  # returns float32 [0..1]
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        print("📦 Loading Depth Anything (LiheYoung/depth-anything-small-hf)...")
        self.processor = AutoImageProcessor.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        ).to(self.device)
        self.model.eval()
        print("✅ Depth model ready on", self.device)

    @torch.no_grad()
    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Input : BGR uint8 image (OpenCV)
        Output: depth map normalized to [0,1] as float32
        """
        # Convert BGR→RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # [1, H', W']

        # Resize depth to match input size
        depth_resized = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0).detach().cpu().numpy()

        # Normalize to [0,1]
        m, M = depth_resized.min(), depth_resized.max()
        if M - m < 1e-8:
            return np.zeros_like(depth_resized, dtype=np.float32)
        depth_norm = (depth_resized - m) / (M - m + 1e-8)
        return depth_norm.astype(np.float32)


# -------------------------------------------------------------------
# LOGGING HELPERS
# -------------------------------------------------------------------
def _ensure_log_files():
    """Create empty log files if they don't exist yet."""
    if not os.path.exists(LIVE_LOG_PATH):
        with open(LIVE_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("=== LIVE LOG START ===\n")
    if not os.path.exists(SUMMARY_LOG_PATH):
        with open(SUMMARY_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("=== SUMMARY LOG START ===\n")


def log_live(msg: str):
    """
    Append one line into live_log.txt and also print to console.
    """
    _ensure_log_files()
    ts = datetime.datetime.now().isoformat()
    line = f"[{ts}] {msg}\n"
    print(line.strip())
    with open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def log_summary(msg: str):
    _ensure_log_files()
    ts = datetime.datetime.now().isoformat()
    line = f"[{ts}] {msg}\n"
    with open(SUMMARY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


# -------------------------------------------------------------------
# SQLITE: left-behind item DB
# -------------------------------------------------------------------
def init_left_items_db(db_path: str = LEFT_DB_PATH):
    """
    Create SQLite DB + table if not exists.
    Table: left_items
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS left_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            obj_id INTEGER,
            item_name TEXT,
            owner_side TEXT,       -- 'left' / 'right'
            table_id TEXT,
            captured_at TEXT,      -- ISO timestamp
            image_path TEXT,
            grok_desc TEXT,        -- later filled by C# Grok API
            grok_desc_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    log_live(f"SQLite left_items table ready at {db_path}")


def save_left_item_event(
    frame: np.ndarray,
    obj_id: int,
    item_name: str,
    owner_side: str,
    bbox,
    table_id: str = TABLE_ID,
    db_path: str = LEFT_DB_PATH,
):
    """
    Save one 'left-behind' item event:
      - crop image from ORIGINAL frame
      - write metadata into SQLite
    bbox: (x1, y1, x2, y2) in ORIGINAL coordinates
    """

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clamp box to image bounds
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        log_live(f"[WARN] Invalid bbox for obj {obj_id}, skip save_left_item_event.")
        return

    ts = datetime.datetime.now().isoformat()

    # --- Save crop image ---
    os.makedirs("captures", exist_ok=True)
    img_name = f"{ts.replace(':', '-')}_obj{obj_id}.jpg"
    img_path = os.path.join("captures", img_name)

    crop = frame[y1:y2, x1:x2].copy()
    cv2.imwrite(img_path, crop)

    # --- Insert row into SQLite ---
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO left_items
        (obj_id, item_name, owner_side, table_id, captured_at, image_path, grok_desc, grok_desc_at)
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (obj_id, item_name, owner_side, table_id, ts, img_path),
    )
    conn.commit()
    conn.close()

    log_live(
        f"LEFT-BEHIND logged: obj_id={obj_id}, item='{item_name}', "
        f"owner_side={owner_side}, img={img_path}"
    )


# Initialize logs and DB once when module is imported
_ensure_log_files()
init_left_items_db()
log_summary("log_and_depth module imported; DB + logs initialised.")
