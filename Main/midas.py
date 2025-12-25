# ===========================
# midas_table_ownership.py — Original-View Ownership with Depth + best.pt table lock
# - MAIN "Frame" window: ORIGINAL CAMERA with all boxes/labels/left-right lines
# - Mini (top-left of Frame): WARPED RGB table (annotated)
# - "Depth Gradient View": full warped depth colormap
# - Tiny depth thumbnail under the mini on the main Frame
# - Depth runs ONLY on the WARPED TABLE after lock
# - Auto-lock table via best.pt (class "table") if available; otherwise manual 4-clicks
# - Robust person filtering to avoid "table as person"
# - Sticky ownership: spatial lines + depth proximity + hysteresis
# - Logs to CSV
# ===========================

import cv2
import numpy as np
from ultralytics import YOLO
import torch, time, csv, os, sys, datetime
import sqlite3  

# ---------------- SETTINGS ----------------
# Camera source
CAMERA_SRC = "http://192.168.50.62:8080/video"  # your IP cam; change to 0/1/2 for USB cam
PREVIEW_SCALE = 0.85     # scaling for displayed main Frame window
MAX_BOX_RATIO = 0.80     # max object box area fraction *of table area* to accept
AUTO_TABLE_LOCK = True   # try to auto-detect the table with best.pt before manual clicks
TABLE_CLASS_NAME = "table"  # class name in your custom best.pt for the table
POSE_CONF = 0.30         # higher to reduce false "person" on borders/table
OBJ_CONF  = 0.22         # object detector conf
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE_STR}{' - ' + torch.cuda.get_device_name(0) if DEVICE_STR=='cuda' else ''}")

# Warped canvas size (top-down table)
WARP_W, WARP_H = 900, 550

# left/right switching hysteresis (pixels) & depth decisiveness threshold
CROSS_MARGIN = 10
DEPTH_DECISIVE_DELTA = 0.06

# Log file
LOG_PATH = "ownership_log.csv"

LIVE_LOG_PATH = "live_log.txt"
SUMMARY_LOG_PATH = "summary_log.txt"

# create empty logs at start
open(LIVE_LOG_PATH, "w", encoding="utf-8").write("=== LIVE LOG START ===\n")
open(SUMMARY_LOG_PATH, "w", encoding="utf-8").write("=== SUMMARY LOG START ===\n")

# SQLite DB for left-behind items
LEFT_DB_PATH = "left_items.db"   # akan dibuat otomatis di folder yang sama
TABLE_ID = "T1"                  # kalau nanti ada banyak meja, bisa diganti dinamis

# track perubahan status orang di meja (None / Left / Right / Both)
prev_status_text = "None"


# ---------------- MODEL LOADING ----------------
# Object detection for items
obj_model  = YOLO("yolov8x.pt")
# People pose
pose_model = YOLO("yolov8n-pose.pt")
# Custom table detector
try:
    table_model = YOLO("best.pt")
    HAS_TABLE_MODEL = True
    print("✅ Loaded best.pt for table detection.")
except Exception as e:
    HAS_TABLE_MODEL = False
    print("⚠️  best.pt not found/failed to load. Will use manual 4-click table lock. Err:", e)

# Try moving to device (older ultralytics may not expose .to; inference still works without)
for _m in (obj_model, pose_model):
    try:
        _m.to(DEVICE_STR)
    except Exception:
        pass
if HAS_TABLE_MODEL:
    try:
        table_model.to(DEVICE_STR)
    except Exception:
        pass

# Depth Anything from Hugging Face (MiDaS-style usage)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
class DepthEstimator:
    def __init__(self, device):
        self.device = device
        print("📦 Loading Depth Anything (LiheYoung/depth-anything-small-hf)...")
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        ).to(self.device)
        self.model.eval()
        print("✅ Depth model ready.")

    @torch.no_grad()
    def predict(self, bgr):
        # returns normalized depth [0..1]
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # [1, H', W']
        depth_resized = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()
        m, M = depth_resized.min(), depth_resized.max()
        if M - m < 1e-8:
            return np.zeros_like(depth_resized, dtype=np.float32)
        depth_norm = (depth_resized - m) / (M - m + 1e-8)
        return depth_norm.astype(np.float32)

depth_est = DepthEstimator(device=DEVICE_STR)

# ---------------- GLOBALS ----------------
points = []                  # 4 chosen points (original)
table_defined = False
src_pts = None               # np.float32(4,2)
dst_pts = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype=np.float32)
M = None                     # original -> warped
Minv = None                  # warped -> original

# tracking/ownership
object_owners = {}     # {obj_id: (owner, (cx,cy,cls))}
previous_owners = {}
next_object_id = 0
left_line_x = None
right_line_x = None

# selection preview scaling (for correct mouse mapping)
SEL_SX, SEL_SY = 1.0, 1.0

# logging init
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "obj_id", "cls", "cx", "cy", "depth", "owner", "note"])

def log_row(obj_id, cls_name, cx, cy, depth, owner, note=""):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"{time.time():.3f}", obj_id, cls_name, cx, cy, f"{depth:.4f}", owner, note])

# ---------- LEFT-BEHIND DB HELPERS ----------

def init_left_items_db():
    """Create SQLite DB and table if not exists."""
    conn = sqlite3.connect(LEFT_DB_PATH)
    c = conn.cursor()
    c.execute("""
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
    """)
    conn.commit()
    conn.close()


def save_left_item_event(frame, obj_id, cls_id, owner_side, bbox, table_id=TABLE_ID):
    """
    Save one 'left-behind' item:
      - crop photo from ORIGINAL frame
      - insert row into left_items table
    bbox: (x1, y1, x2, y2) in ORIGINAL coords
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return  # box invalid, skip

    ts = datetime.datetime.now().isoformat()
    item_name = obj_model.names[cls_id]

    # crop foto barang
    crop = frame[y1:y2, x1:x2].copy()
    os.makedirs("captures", exist_ok=True)
    img_name = f"{ts.replace(':','-')}_obj{obj_id}.jpg"
    img_path = os.path.join("captures", img_name)
    cv2.imwrite(img_path, crop)

    # simpan ke SQLite
    conn = sqlite3.connect(LEFT_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO left_items
        (obj_id, item_name, owner_side, table_id, captured_at, image_path, grok_desc, grok_desc_at)
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
    """, (obj_id, item_name, owner_side, table_id, ts, img_path))
    conn.commit()
    conn.close()


# ---------------- CAMERA ----------------
# init DB once at start
init_left_items_db()

# ---------------- CAMERA ----------------
# init DB once at start
init_left_items_db()

def init_camera_wide(src_index=0):
    """
    Open a webcam with a 'wide' friendly resolution and no digital zoom.
    src_index = 0 or 1 depending on which camera you want.
    """
    cap = cv2.VideoCapture(src_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Ask for a wide-ish resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Try to reset any digital zoom (may be ignored by some drivers)
    cap.set(cv2.CAP_PROP_ZOOM, 0)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read from webcam")

    h, w = frame.shape[:2]
    print(f"✅ Webcam using resolution: {w} x {h}")
    return cap, frame

cap, first_frame = init_camera_wide(1) 
# ---------------- UTILS ----------------
def order_corners(pts4):
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_image(img):  # original -> warped
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)

def warp_point(x, y):  # original -> warped
    pt = np.array([[[x, y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, M)[0,0]
    return int(res[0]), int(res[1])

def in_table_polygon(x, y):  # ORIGINAL space
    return cv2.pointPolygonTest(src_pts.astype(np.int32), (x, y), False) >= 0

def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes_array = np.array(boxes)
    x1 = boxes_array[:,0]; y1 = boxes_array[:,1]
    x2 = boxes_array[:,2]; y2 = boxes_array[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def sample_depth(depth_map, x, y, k=3):  # WARPED space
    h, w = depth_map.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    x1, x2 = max(0, x - k), min(w, x + k + 1)
    y1, y2 = max(0, y - k), min(h, y + k + 1)
    patch = depth_map[y1:y2, x1:x2]
    return float(np.median(patch)) if patch.size > 0 else 0.0

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    return inter / union

def expand_box(box, dx, dy):
    x1,y1,x2,y2 = box
    return (int(x1-dx), int(y1-dy), int(x2+dx), int(y2+dy))

# tracking
def assign_ids_and_track(old_objs, new_objs, threshold=50):
    global next_object_id
    new_with_ids = []
    for (x1, y1, x2, y2, cls, conf) in new_objs:
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        matched_id = None
        min_dist = float('inf')
        for obj_id, (ox, oy, ocls) in old_objs.items():
            dist = np.hypot(ox - cx, oy - cy)
            if dist < threshold and cls == ocls and dist < min_dist:
                min_dist = dist
                matched_id = obj_id
        if matched_id is not None:
            new_with_ids.append((matched_id, x1, y1, x2, y2, cls, conf, cx))
        else:
            obj_id = next_object_id
            next_object_id += 1
            new_with_ids.append((obj_id, x1, y1, x2, y2, cls, conf, cx))
    new_dict = {obj_id: ((x1 + x2)//2, (y1 + y2)//2, cls) for (obj_id, x1, y1, x2, y2, cls, conf, cx) in new_with_ids}
    return new_with_ids, new_dict

# left/right helper drawing in ORIGINAL space
def left_person(display_frame, boxes, person_box=None):
    if len(boxes) == 0 or src_pts is None:
        return display_frame, None
    right_most = max(boxes, key=lambda b: b[2])
    x2 = right_most[2]
    upper_y = int(src_pts[:,1].min())
    lower_y = int(src_pts[:,1].max())
    if person_box is not None:
        px1, py1, px2, py2 = person_box
        cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
    cv2.line(display_frame, (x2, upper_y), (x2, lower_y), (255, 255, 0), 2)
    return display_frame, x2

def right_person(display_frame, boxes, person_box=None):
    if len(boxes) == 0 or src_pts is None:
        return display_frame, None
    left_most = min(boxes, key=lambda b: b[0])
    x1 = left_most[0]
    upper_y = int(src_pts[:,1].min())
    lower_y = int(src_pts[:,1].max())
    if person_box is not None:
        px1, py1, px2, py2 = person_box
        cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
    cv2.line(display_frame, (x1, upper_y), (x1, lower_y), (0, 255, 255), 2)
    return display_frame, x1

def clamp_lines(left_x, right_x):
    if left_x is not None and right_x is not None and left_x >= right_x:
        left_x = right_x - 1
    return left_x, right_x

def update_territory_lines(left_items, right_items, left_x, right_x):
    if left_items:
        left_x = max((x2 for (x1,y1,x2,y2,_,_) in left_items), default=left_x)
    if right_items:
        right_x = min((x1 for (x1,y1,x2,y2,_,_) in right_items), default=right_x)
    return clamp_lines(left_x, right_x)

def person_depth_signature(kpts_warp, depth_map, box_warp):
    samples = []
    if kpts_warp is not None and len(kpts_warp) >= 16:
        for idx in [7,8,9,10]:  # elbows + wrists
            x, y = kpts_warp[idx]
            if x >= 0 and y >= 0:
                samples.append(sample_depth(depth_map, int(x), int(y), k=2))
    if not samples and box_warp is not None:
        x1,y1,x2,y2 = box_warp
        cx, cy = (x1+x2)//2, (y1+y2)//2
        samples.append(sample_depth(depth_map, cx, cy, k=3))
    return float(np.median(samples)) if samples else 0.0

def decide_owner_with_depth(cx, depth, left_anchor, right_anchor, left_line_x, right_line_x, mid_x):
    if left_line_x is not None and cx <= left_line_x - 5:  return "left"
    if right_line_x is not None and cx >= right_line_x + 5: return "right"
    l_depth = left_anchor.get("depth", None)
    r_depth = right_anchor.get("depth", None)
    if l_depth is None and r_depth is None: return "left" if cx < mid_x else "right"
    if l_depth is None: return "right"
    if r_depth is None: return "left"
    dl = abs(depth - l_depth); dr = abs(depth - r_depth)
    if abs(dl - dr) < 0.02: return "left" if cx < mid_x else "right"
    return "left" if dl < dr else "right"

# ---------------- TABLE LOCK: AUTO via best.pt, else manual 4-click ----------------
def auto_lock_table(frame):
    """
    Try to detect the largest 'table' box from best.pt and convert to 4 corner points.
    Returns points list [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] or None
    """
    try:
        res = table_model(frame, conf=0.35, verbose=False)
    except Exception:
        return None
    if len(res) == 0:
        return None
    r0 = res[0]
    best = None
    best_area = 0
    for b in r0.boxes:
        cls = int(b.cls[0])
        name = table_model.names.get(cls, "").lower()
        if name != TABLE_CLASS_NAME:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        area = max(0, x2-x1) * max(0, y2-y1)
        if area > best_area:
            best_area = area
            best = (x1,y1,x2,y2)
    if best is None:
        return None
    x1,y1,x2,y2 = best
    return [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]

def click_event(event, x, y, flags, param):
    # x,y are already in ORIGINAL pixel coords because we show 1:1
    global points, table_defined
    if event == cv2.EVENT_LBUTTONDOWN and not table_defined:
        if len(points) < 4:
            points.append((int(x), int(y)))
            print(f"Point {len(points)}: {(int(x), int(y))}")

def do_table_lock(cap):
    global table_defined, src_pts, M, Minv, points
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Cannot read from camera for table lock")

    if AUTO_TABLE_LOCK and HAS_TABLE_MODEL:
        auto_pts = auto_lock_table(first)
        if auto_pts is not None:
            points = auto_pts
            table_defined = True
            print("✅ Table auto-locked via best.pt.")
        else:
            print("⚠️  best.pt could not detect table; falling back to manual clicks.")

    if not table_defined:
       cv2.namedWindow("Frame: Select Table", cv2.WINDOW_AUTOSIZE)  # 1:1 exact pixels
       cv2.setMouseCallback("Frame: Select Table", click_event)
       cv2.resizeWindow("Frame: Select Table", first.shape[1], first.shape[0])

       print("🟢 Click 4 corners of the TABLE (clockwise or CCW). Press ESC to cancel.")

       while True:
            ret, frm = cap.read()
            if not ret: break

            disp = frm.copy()
            for i, pt in enumerate(points):
                cv2.circle(disp, pt, 6, (0, 255, 255), -1)
                cv2.putText(disp, f"{i+1}", (pt[0]+6, pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(disp, points[i-1], points[i], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(disp, points[3], points[0], (0, 255, 0), 2)
                table_defined = True

            # show exactly the original frame – no scaling → no offset
            cv2.imshow("Frame: Select Table", disp)

            k = cv2.waitKey(1)
            if k == 27 or table_defined:
                break

        # Only try to destroy the window if it was actually created
    try:
        cv2.destroyWindow("Frame: Select Table")
    except cv2.error:
        pass

    if not table_defined or len(points) < 4:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("❌ Table was not locked.")


    # finalize transform
    src = order_corners(points)
    _M = cv2.getPerspectiveTransform(src, dst_pts)
    _Minv = cv2.getPerspectiveTransform(dst_pts, src)
    return src, _M, _Minv

src_pts, M, Minv = do_table_lock(cap)
print("✅ Table locked. MAIN Frame shows ORIGINAL camera; mini shows WARPED table.")

# ---------------- WINDOWS ----------------
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)                # MAIN: original camera
cv2.namedWindow("Depth Gradient View", cv2.WINDOW_NORMAL)  # full depth gradient (warped)

# ---------------- MAIN LOOP ----------------
status_text = "None"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    main_disp = frame.copy()

    # Build warped table for depth & table annotations (mini)
    warped = warp_image(frame)  # (WARP_H, WARP_W, 3)
    table_disp = warped.copy()

    # Depth on warped
    depth_map = depth_est.predict(warped)
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

    # Mini sizes
    mini_w = max(260, WARP_W // 3)
    mini_h = int(mini_w * WARP_H / WARP_W)
    # Tiny depth thumbnail (aligned width)
    dm_thumb = cv2.resize(depth_vis, (mini_w, int(mini_w * depth_vis.shape[0] / depth_vis.shape[1])))

    # Show full depth window
    cv2.imshow("Depth Gradient View", cv2.resize(depth_vis, (WARP_W, WARP_H)))

    # ---------------- OBJECT DETECTION (original), restrict to table polygon ----------------
    detected_boxes = []  # ORIGINAL coords (x1,y1,x2,y2,cls,conf)
    det_res = obj_model(frame, conf=OBJ_CONF, verbose=False)
    if len(det_res) > 0:
        r0 = det_res[0]
        table_area = cv2.contourArea(src_pts.astype(np.int32))
        for det in r0.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            name = obj_model.names.get(cls, "").lower()
            # skip explicit unwanted classes
            if name in {"chair"}:
                continue
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            box_area = max(0, x2-x1) * max(0, y2-y1)
            if in_table_polygon(cx, cy) and box_area <= table_area * MAX_BOX_RATIO:
                detected_boxes.append((x1, y1, x2, y2, cls, conf))

    # TRACK objects
    detected_with_ids, tracked_objects = assign_ids_and_track(
        {obj_id: data for obj_id, (owner, data) in object_owners.items()},
        detected_boxes
    )

    # ---------------- POSE (original) → filter valid table seaters; also produce warped anchors for depth -----------
    person_boxes = []       # ORIGINAL person ROIs used for drawing lines
    person_boxes_warp = []  # WARPED boxes for anchors (depth)
    persons_kpts_warp = []  # WARPED keypoints arrays

    # derive table bbox (original) for quick geometric filters
    x_min_tbl = int(src_pts[:,0].min()); x_max_tbl = int(src_pts[:,0].max())
    y_min_tbl = int(src_pts[:,1].min()); y_max_tbl = int(src_pts[:,1].max())
    table_box = (x_min_tbl, y_min_tbl, x_max_tbl, y_max_tbl)
    wide_near = expand_box(table_box, dx=120, dy=140)  # allow near-table margin

    pose_res = pose_model(frame, conf=POSE_CONF, verbose=False)
    for pr in pose_res:
        kobj = getattr(pr, "keypoints", None)
        if kobj is None: continue
        kxy = getattr(kobj, "xy", None)
        if kxy is None: continue

        for k in kxy:
            ko = k.detach().cpu().numpy().astype(float)
            if ko.ndim != 2 or ko.shape[1] != 2:
                continue
            ko = ko[np.isfinite(ko).all(axis=1)]
            if ko.size == 0:
                continue

            # build bbox in ORIGINAL
            px1, py1 = int(ko[:,0].min()), int(ko[:,1].min())
            px2, py2 = int(ko[:,0].max()), int(ko[:,1].max())
            pbox = (px1, py1, px2, py2)

            # FILTERS to avoid "table as person"
            # - require overlap with horizontal span of table and near vertically
            # - also require not almost entirely inside the table (people are outside)
            # - require minimum keypoints
            if len(ko) < 8:
                continue
            # Must overlap horizontally and be vertically close to the table
            if not (px2 > x_min_tbl - 80 and px1 < x_max_tbl + 80 and
                    py2 > y_min_tbl - 150 and py1 < y_max_tbl + 180):
                continue
            # If person box is 90% inside the table box, reject (that's likely table false-positive)
            inter_iou = iou_xyxy(pbox, table_box)
            if inter_iou > 0.80 and (px2-px1)*(py2-py1) < 1.1*(x_max_tbl-x_min_tbl)*(y_max_tbl-y_min_tbl):
                continue

            # Accept this person
            person_boxes.append(pbox)

            # Warp the keypoints for depth anchors & draw on table_disp (warped)
            kw_list = []
            for (x, y) in ko:
                xw, yw = warp_point(x, y)
                kw_list.append([xw, yw])
            kw = np.asarray(kw_list, dtype=int)
            if kw.ndim != 2 or kw.shape[0] == 0:
                continue

            x1w, y1w = int(kw[:,0].min()), int(kw[:,1].min())
            x2w, y2w = int(kw[:,0].max()), int(kw[:,1].max())
            person_boxes_warp.append((x1w, y1w, x2w, y2w))
            persons_kpts_warp.append(kw)

            # draw skeleton on warped mini
            skeleton = [
                (5,7),(7,9),(6,8),(8,10),
                (11,13),(13,15),(12,14),(14,16),
                (5,6),(11,12),(5,11),(6,12)
            ]
            for (a,b) in skeleton:
                a -= 1; b -= 1
                if 0 <= a < len(kw) and 0 <= b < len(kw):
                    cv2.line(table_disp, tuple(kw[a]), tuple(kw[b]), (0,255,0), 2)
            for (kx,ky) in kw:
                cv2.circle(table_disp, (int(kx),int(ky)), 3, (0,255,255), -1)

    person_boxes_warp = remove_duplicate_boxes(person_boxes_warp, iou_threshold=0.4)

    # ---------------- ANCHORS + LINE UPDATES (use warped for depth anchors; draw lines on original) ----------------
    left_items = []   # ORIGINAL coords for boundary calc on main frame
    right_items = []
    table_mid_x = (x_min_tbl + x_max_tbl) // 2

    left_anchor = {}
    right_anchor = {}

    if person_boxes_warp:
        # choose left/right by warped center x
        centers = [(((b[0]+b[2])//2), idx) for idx,b in enumerate(person_boxes_warp)]
        left_idx  = min(centers, key=lambda t: t[0])[1]
        right_idx = max(centers, key=lambda t: t[0])[1]
        left_person_box_w  = person_boxes_warp[left_idx]
        right_person_box_w = person_boxes_warp[right_idx]
        left_k  = persons_kpts_warp[left_idx]  if left_idx  < len(persons_kpts_warp) else None
        right_k = persons_kpts_warp[right_idx] if right_idx < len(persons_kpts_warp) else None

        left_depth  = person_depth_signature(left_k,  depth_map, left_person_box_w)
        right_depth = person_depth_signature(right_k, depth_map, right_person_box_w)
        left_anchor  = {"box": left_person_box_w,  "depth": left_depth}
        right_anchor = {"box": right_person_box_w, "depth": right_depth}

    # ---------------- OWNERSHIP with depth + sticky; DRAW ON ORIGINAL ----------------
    # First, for each object, compute warped center to sample depth
    obj_draw_cache = []  # (obj_id, x1,y1,x2,y2, cls, conf, cx, cy, d_warp, final_owner)

    # provisional group for line update (ORIGINAL)
    for (obj_id, x1,y1,x2,y2,cls,conf,cx) in detected_with_ids:
        # warped center
        cxw, cyw = warp_point(cx, (y1+y2)//2)
        d = sample_depth(depth_map, cxw, cyw, k=2)

        prev_owner = object_owners.get(obj_id, ("none", None))[0]
        owner = decide_owner_with_depth(cx, d, left_anchor, right_anchor, left_line_x, right_line_x, table_mid_x)

        # accumulate for line updates
        if owner == "left":
            left_items.append((x1,y1,x2,y2,cls,conf))
        else:
            right_items.append((x1,y1,x2,y2,cls,conf))

        obj_draw_cache.append((obj_id, x1,y1,x2,y2, cls, conf, cx, (y1+y2)//2, d, owner, prev_owner))

    # Update the vertical boundaries (on ORIGINAL x)
    if len(person_boxes) > 0:
        left_line_x, right_line_x = update_territory_lines(left_items, right_items, left_line_x, right_line_x)
    # Else keep previous lines unchanged

    # Draw person boxes and boundary lines on ORIGINAL
    if person_boxes:
        left_person_box_o  = min(person_boxes, key=lambda b: (b[0]+b[2])//2)
        right_person_box_o = max(person_boxes, key=lambda b: (b[0]+b[2])//2)
        main_disp, left_line_x  = left_person(main_disp, left_items,  person_box=left_person_box_o)
        main_disp, right_line_x = right_person(main_disp, right_items, person_box=right_person_box_o)
        if len(person_boxes) == 1:
            status_text = "Left" if (left_person_box_o[0]+left_person_box_o[2])//2 < table_mid_x else "Right"
        else:
            status_text = "Both"
    else:
        status_text = "None"

    # Final sticky ownership + draw on ORIGINAL + log
    for (obj_id, x1,y1,x2,y2, cls, conf, cx, cy, d, owner, prev_owner) in obj_draw_cache:
        final_owner = owner

        # sticky: only switch if both boundary crossing + decisive depth advantage
        if prev_owner != "none" and final_owner != prev_owner:
            lD = left_anchor.get("depth", None)
            rD = right_anchor.get("depth", None)
            decisive = False
            if lD is not None and rD is not None:
                decisive = abs(abs(d - lD) - abs(d - rD)) > DEPTH_DECISIVE_DELTA
            crossed = ((left_line_x  is not None and prev_owner=="left"  and cx > left_line_x  + CROSS_MARGIN) or
                       (right_line_x is not None and prev_owner=="right" and cx < right_line_x - CROSS_MARGIN))
            if not (decisive and crossed):
                final_owner = prev_owner

        object_owners[obj_id] = (final_owner, (cx, cy, cls))

        color = (255,255,0) if final_owner=="left" else ((0,255,255) if final_owner=="right" else (0,0,255))
        cv2.rectangle(main_disp, (x1,y1), (x2,y2), color, 2)
        cv2.putText(main_disp, f"{obj_model.names[cls]} {conf:.2f} ({final_owner}) d={d:.2f}",
                    (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Log only when owner changes / new
        if obj_id not in previous_owners or previous_owners[obj_id] != final_owner:
            timestamp = datetime.datetime.now().isoformat()

            with open(LIVE_LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(
                    f"[{timestamp}] OBJ {obj_id} ({obj_model.names[cls]}) "
                    f"at ({cx},{cy}) depth={d:.2f} owner={final_owner} "
                    f"NOTE={'' if prev_owner==final_owner else f'CHANGED:{prev_owner}->{final_owner}'}\n"
                )

            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([timestamp, obj_id, obj_model.names[cls], cx, cy,
                                        f"{d:.4f}", final_owner,
                                        "" if prev_owner==final_owner else f"owner_change:{prev_owner}->{final_owner}"])

        previous_owners[obj_id] = final_owner


    # keep only tracked
    object_owners = {oid: (own, data) for oid,(own,data) in object_owners.items()
                     if oid in {oid for (oid, *_rest) in detected_with_ids}}
    # detect lost items
    current_ids = {oid for (oid, *_r) in detected_with_ids}
    previous_ids = set(previous_owners.keys())

    lost_ids = previous_ids - current_ids
    for lid in lost_ids:
        ts = datetime.datetime.now().isoformat()
        with open(LIVE_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(f"[{ts}] OBJ {lid} LOST from camera (last_owner={previous_owners[lid]})\n")

    # ---------- LEFT-BEHIND DETECTION (status change) ----------

    left_went_away  = (prev_status_text in ("Left", "Both")  and status_text in ("None", "Right"))
    right_went_away = (prev_status_text in ("Right", "Both") and status_text in ("None", "Left"))

    if left_went_away or right_went_away:
        # untuk tiap object yang masih terdeteksi dan punya owner yang baru saja "pergi"
        for (obj_id, x1, y1, x2, y2, cls, conf, cx, cy, d, owner, prev_owner) in obj_draw_cache:
            own_now = object_owners.get(obj_id, ("none", None))[0]
            if left_went_away and own_now == "left":
                save_left_item_event(frame, obj_id, cls, "left", (x1, y1, x2, y2))
            if right_went_away and own_now == "right":
                save_left_item_event(frame, obj_id, cls, "right", (x1, y1, x2, y2))

    # update status lama untuk loop berikutnya
    prev_status_text = status_text


    # ----- COMPOSE MAIN WINDOW (original camera + mini overlays) -----
    # 1) paste mini warped (annotated RGB)
    mini_warp = cv2.resize(table_disp, (mini_w, mini_h))
    y0, x0 = 10, 10
    y1, x1p = y0 + mini_h, x0 + mini_w
    main_disp[y0:y1, x0:x1p] = mini_warp

    # 2) paste tiny depth thumbnail UNDER the mini warped
    y2 = y1 + 8
    dth = min(dm_thumb.shape[0], max(1, main_disp.shape[0] - y2 - 10))
    dtw = min(dm_thumb.shape[1], mini_w)
    dm_thumb_resized = cv2.resize(dm_thumb, (dtw, dth))
    main_disp[y2:y2+dth, x0:x0+dtw] = dm_thumb_resized

    # 3) status text
    cv2.putText(main_disp, f"Person Detected: {status_text}", (10, main_disp.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show MAIN
    out_w = int(main_disp.shape[1] * PREVIEW_SCALE)
    out_h = int(main_disp.shape[0] * PREVIEW_SCALE)
    cv2.imshow("Frame", cv2.resize(main_disp, (out_w, out_h)))

    if cv2.waitKey(1) == 27:
        break
# WRITE SUMMARY LOG AT END
with open(SUMMARY_LOG_PATH, "a", encoding="utf-8") as sf:
    sf.write("\n=== FINAL SUMMARY ===\n")

    if len(object_owners) == 0:
        sf.write("No items left on table.\n")
    else:
        for oid, (own, (cx, cy, cls)) in object_owners.items():
            sf.write(
                f"OBJ {oid} ({obj_model.names[cls]}) "
                f"owner={own} last_pos=({cx},{cy})\n"
            )

cap.release()
cv2.destroyAllWindows()
