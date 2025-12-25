import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.errors import GEOSException
import torch
import face_recognition
from pathlib import Path
import joblib
import time
import os
import datetime
import copy
import psutil
import sys
import json
from dual_model_inference import run_dual_inference
from decision_model.predict_data import predict_status

from log_and_depth import (
    DepthEstimator,
    init_left_items_db,
    save_left_item_event,
    log_live,
)

print("CUDA available:", torch.cuda.is_available())
print("Num GPUs:", torch.cuda.device_count())
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu" #prioritize cuda

# Path untuk live frame (1 file saja, di folder yang sama dengan main.py)
BASE_DIR = Path(__file__).resolve().parent
FRAME_OUTPUT_PATH = BASE_DIR / "live_frame.jpg"
STATUS_OUTPUT_PATH = BASE_DIR / "live_status.json"

polygon = None #placeholder for table polygon (4 points)
points = [] #list for mouse-clicked points when defining table
table_defined = False #boolean if table has been defined

pose_model = YOLO("yolo11n-pose.pt") #pose model

if torch.cuda.is_available(): #move models to GPU if available
    pose_model.model = pose_model.model.to(DEVICE_STR)

init_left_items_db() #initialize left items database
log_live("=== APP START (new main.py) ===")

object_owners = {}  # obj_id -> "left" or "right"
object_centers = {}  # obj_id -> {"cx","left","right","top","bottom"}
object_names = {} # obj_id -> obj name
next_object_id = 1
prev_objects = {}     # obj_id -> {"xyxy": [x1,y1,x2,y2], "name": ...}
IOU_TRACK_THRESHOLD = 0.5

LEFT_TERRITORY_X = None  # vertical boundary separating left owner territory
RIGHT_TERRITORY_X = None # vertical boundary separating right owner territory
SMOOTHING_ALPHA = 0.9

# track if we've already logged left_behind items for a given absence period
already_logged = {"left": False, "right": False}
seat_label = {
    "left": "unknown",   # "seated" | "temporary_leave" | "left_behind" | "unknown"
    "right": "unknown",
}

# NEW: storage for debug-capture mode
final_objects = {}
# For debug exit-capture
exit_objects = {}
exit_clean_frame = None

# DEBUG
DEBUG_FORCE_CAPTURE = False
USE_LEFT_BEHIND_MODEL = True # SET False = tes capture teken Q

#face recognition model load
MODEL_PATH = Path("face_recognition_model.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

knn_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

PROBABILITY_THRESHOLD = 0.75 #threhold for face recognition
PROBABILITY_MARGIN = 0.2 
RESIZE_SCALE = 0.25 #scale image faces down to speed up face recognition
DETECTION_MODEL = "hog" #backend model
FRAME_SKIP = 6

# Track warnings so we only print once per unseen item
UNSEEN_ITEM_WARNING = set()

def iou(a, b):
    # a, b are [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def debug_capture_all_items_on_table(clean_frame, table_polygon):
    if table_polygon is None:
        print("Table polygon not defined, cannot capture.")
        return

    if not object_centers:
        print("No objects in object_centers, nothing to capture.")
        return

    log_live("Manual Q pressed → capturing ALL table items (model ignored).")

    for obj_id, cen in object_centers.items():
        name = object_names.get(obj_id, "Unknown")
        owner = object_owners.get(obj_id)

        if owner not in ("left", "right"):
            continue

        x1 = cen["left"]
        x2 = cen["right"]
        y1 = cen["top"]
        y2 = cen["bottom"]

        inside, percent = bbox_inside_polygon_percentage(
            (x1, y1, x2, y2), table_polygon, min_percent=30
        )
        if not inside:
            continue

        print(
            f"Saving obj {obj_id}: {name}, "
            f"owner={owner}, inside={percent:.1f}%"
        )

        save_left_item_event(
            frame=clean_frame,   
            obj_id=obj_id,
            item_name=name,
            owner_side=owner,
            bbox=(x1, y1, x2, y2),
        )

    print("DONE manual capture.\n")

#placing overlay for face recognition frames
def overlay_frame(base, overlay, x, y):
    h, w, _ = overlay.shape
    base[y : y + h, x : x + w] = overlay

#handle mouse clicking for table definition
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print("Point added:", (x, y))

#compute how much of the bbox is inside table polygon
def bbox_inside_polygon_percentage(box, table_polygon, min_percent=40):
    x1, y1, x2, y2 = box #object bbox coords
    bbox_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) #object polygon bbox

    table_poly = Polygon(table_polygon) #table polygon
    if not table_poly.is_valid:
        table_poly = table_poly.buffer(0)

    try:
        inter = bbox_poly.intersection(table_poly)
        inter_area = inter.area #intersection area between table polygon and object bbox polygon
    except GEOSException as e:
        print(f"Shapely intersection failed: {e}")
        return False, 0

    object_area = bbox_poly.area #area of object bbox
    if object_area == 0:
        return False, 0

    inside_percent = (inter_area / object_area) * 100.0 #percentage of object bbox inside table polygon
    return inside_percent >= min_percent, inside_percent

#extract pose keypoints from tensor shapes
def safe_extract_keypoints(kpts_tensor):
    arr = kpts_tensor.cpu().numpy()
    if arr.ndim == 2 and arr.shape == (17, 3):
        return arr[:, :2], arr[:, 2]
    if arr.ndim == 3 and arr.shape[1:] == (17, 3):
        return arr[0, :, :2], arr[0, :, 2]
    if arr.ndim == 1:
        if arr.shape[0] == 51:
            resh = arr.reshape(17, 3)
            return resh[:, :2], resh[:, 2]
        if arr.shape[0] == 34:
            resh = arr.reshape(17, 2)
            return resh, np.ones(17) * 0.5
    if arr.ndim == 2 and arr.shape == (17, 2):
        return arr, np.ones(17) * 0.5
    return np.zeros((17, 2)), np.zeros(17) #return keypoints and confidences of keypoints

#compute chest midpoint from keypoints
def get_chest_midpoint(kpts, conf):
    LEFT_SHOULDER = 5 #left shoulder keypoint index
    RIGHT_SHOULDER = 6 #right shoulder keypoint index
    lx, ly = kpts[LEFT_SHOULDER] #coordinates of left shoulder
    rx, ry = kpts[RIGHT_SHOULDER] #coordinates of right shoulder
    if conf[LEFT_SHOULDER] > 0.1 and conf[RIGHT_SHOULDER] > 0.1: #both shoulder above confidence threshold
        return int((lx + rx) / 2), int((ly + ry) / 2) #return midpoint coordinates
    #pick bottommost point of the points given
    valid_points = [(x, y, c) for (x, y), c in zip(kpts, conf) if c > 0.001]
    if valid_points:
        bottom = max(valid_points, key=lambda p: p[1])
        return int(bottom[0]), int(bottom[1])
    return None

#euclidean distance between chest midpoint and a reference table point
def is_near_table(chest, table_point, max_dist=200):
    fx, fy = chest
    tx, ty = table_point
    dist = np.sqrt((fx - tx) ** 2 + (fy - ty) ** 2)
    return dist < max_dist, dist #true if within max distance

#compute center of bbox
def compute_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

#euclidean distance between two points
def distance(p1, p2):
    if p1 is None or p2 is None:
        return 1e9 #return large distance if either point is None
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

#recomputes territory lines based on current object positions
def update_territory_lines():
    global LEFT_TERRITORY_X, RIGHT_TERRITORY_X #territory lines positions

    #list of right edges positions of objects owned by left side
    left_edges = [
        data["right"]
        for oid, data in object_centers.items()
        if object_owners.get(oid) == "left"
    ]

    #list of left edges positions of objects owned by right side
    right_edges = [
        data["left"]
        for oid, data in object_centers.items()
        if object_owners.get(oid) == "right"
    ]

    raw_left = max(left_edges) if left_edges else None #rightmost edge of left-owned objects
    raw_right = min(right_edges) if right_edges else None #leftmost edge of right-owned objects

    if raw_left is not None:
        LEFT_TERRITORY_X = smooth_value(LEFT_TERRITORY_X, raw_left) #smooth left territory line

    if raw_right is not None:
        RIGHT_TERRITORY_X = smooth_value(RIGHT_TERRITORY_X, raw_right) #smooth right territory line

#smoothing function for territory lines updates
def smooth_value(old, new, alpha=SMOOTHING_ALPHA):
    if old is None:
        return new
    return int(old * (1 - alpha) + new * alpha)

#face recognition function for a single frame
def run_face_recognition_frame(frame):
    small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE) #resize frame for faster processing
    face_locations = face_recognition.face_locations(small, model=DETECTION_MODEL)

    #nothing detected
    if len(face_locations) == 0:
        return frame, []

    face_encodings = face_recognition.face_encodings(small, face_locations) #encode detected faces
    results = [] #list to store recognition results

    for encoding, loc in zip(face_encodings, face_locations):
        if hasattr(knn_model, "predict_proba"):
            probs = knn_model.predict_proba([encoding])[0] #get probability scores for each class
            idx = probs.argmax() #index of highest confidence score
            prob = probs[idx] #highest confidence score
            label = label_encoder.inverse_transform([idx])[0] #get corresponding label (name)
        else:
            label = knn_model.predict([encoding])[0] #predict label directly
            prob = 1.0 #assume full confidence if no probability scores

        if prob >= PROBABILITY_THRESHOLD - PROBABILITY_MARGIN:
            results.append((label, loc, prob)) #append recognized name, location, and probability if confidence is high enough
        else:
            results.append(("Unknown", loc, prob)) #append as Unknown if confidence is low

    #draw rectangles and labels on frame
    for name, loc, prob in results:
        top, right, bottom, left = [int(v / RESIZE_SCALE) for v in loc]
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            f"{name} {prob:.2f}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame, results


EMPTY_SMOOTH_SEC = 3.0  # how long we wait before saying "seat is really empty"

#update seat state with smoothing
import time
import copy

EMPTY_SMOOTH_SEC = 5  # seconds of smoothing

def update_seat_state(side, current_person, seat_state, prev_seat_state, person):
    now = time.time()
    seat = seat_state[side]
    temp = seat["is_present"]

    recognized_name = str(person.get(str(current_person), "Unknown"))
    prev_name = prev_seat_state[side].get("owner_name")

    # CASE 1 — no person detected
    if current_person is None:
        seat["start_sit_time"] = None
        if seat["current_owner"] is None:
            seat["is_present"] = False
            seat["last_seen_time"] = None
            return
        if seat["last_seen_time"] is None:
            seat["last_seen_time"] = now
        gone_sec = now - seat["last_seen_time"]
        seat["is_present"] = gone_sec < EMPTY_SMOOTH_SEC
        seat["new_person_present"]=True
        return

    # CASE 2 — recognized name matches previous valid name
    if prev_name not in (None, "Unknown") and recognized_name == prev_name:
        temp_start_sit_time = seat_state[side].get("start_sit_time")
        temp_last_return_time = seat_state[side].get("last_return_time")
        seat_state[side] = copy.deepcopy(prev_seat_state[side])
        seat_state[side]["is_present"]=True
        seat_state[side]["start_sit_time"] = temp_start_sit_time
        seat_state[side]["num_returns"]+=1
        seat_state[side]["last_return_time"]=temp_last_return_time
        seat_state[side]["new_person_present"]=False
        seat["current_owner"] = current_person  # update tracker ID
        seat["is_present"] = True
        if not temp and seat["is_present"]:
            seat["start_sit_time"] = now
        seat["last_seen_time"] = None
        # Update prev_state to reflect latest tracker ID
        prev_seat_state[side] = copy.deepcopy(seat_state[side])
        return

    # CASE 3 — same tracker ID as before
    if seat["current_owner"] == current_person:
        if seat["last_seen_time"] is not None and now - seat["last_seen_time"] >= EMPTY_SMOOTH_SEC:
            seat["num_returns"] += 1
            seat["last_return_time"] = now
        seat["is_present"] = True
        if not temp and seat["is_present"]:
            seat["start_sit_time"] = now
        seat["last_seen_time"] = None
        seat["new_person_present"]=False
        if recognized_name not in (None, "Unknown"):
            prev_seat_state[side] = copy.deepcopy(seat_state[side])
        return

    # CASE 4 — new person (different tracker ID / different name)
    seat["current_owner"] = current_person
    seat["owner_name"] = recognized_name
    seat["session_start_time"] = now
    seat["start_sit_time"] = now
    seat["last_seen_time"] = None
    seat["is_present"] = True
    seat["num_returns"] = 0
    seat["last_return_time"] = now
    seat["new_person_present"]=True
    # DO NOT update prev_seat_state here! Only update when a valid name continues a session
    if recognized_name not in (None, "Unknown"):
        prev_seat_state[side] = copy.deepcopy(seat_state[side])

#check if chest are above table level
def is_chest_above_table(kpts, kconf, table_polygon):
    #extract ankles keypoints
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    valid_pts = [] #list to store valid shoulder points

    #shoulders confidence check
    if kconf[LEFT_SHOULDER] > 0.1:
        valid_pts.append(kpts[LEFT_SHOULDER])
    if kconf[RIGHT_SHOULDER] > 0.1:
        valid_pts.append(kpts[RIGHT_SHOULDER])

    #no shoulders detected
    if not valid_pts:
        return False

    #table top y-coordinate
    top_y = min(p[1] for p in table_polygon)

    #table horizontal span
    xs = [p[0] for p in table_polygon]
    min_x, max_x = min(xs), max(xs)

    # Every valid shoulder must satisfy BOTH:
    # 1) horizontally inside table bounds
    # 2) vertically above table top
    for (ax, ay) in valid_pts:
        #must be above table top
        if ay > top_y:
            return False
        #must be inside horizontal span
        if ax < min_x or ax > max_x:
            return False

    return True

#prepare left behind prediction model parameters
def compute_seat_features(side, seat_state):
    now = time.time() #current time
    seat = seat_state[side] #seat state for the given side

    is_present = seat["is_present"]

    #filter items so only model-supported ones go in
    item_list = []
    for obj_id, owner in object_owners.items():
        if owner != side:
            continue

        raw_name = object_names.get(obj_id, "Unknown")
        item_list.append(raw_name)

    owner_side = side #"left" or "right"
    weekday = datetime.datetime.now().strftime("%a") #current weekday + formatting

    #time since last seen person
    if seat["last_seen_time"] is None:
        time_since_person = 0.0
    else:
        time_since_person = now - seat["last_seen_time"]

    nowDT = datetime.datetime.now()
    time_of_day = nowDT.hour + nowDT.minute / 60 + nowDT.second / 3600 #curernt time in seconds

    #new session
    if seat["session_start_time"] is None:
        current_sit_minutes = 0.0
        total_session_minutes = 0.0
    #update session times
    else:
        total_session_minutes = (now - seat["session_start_time"]) / 60.0
        if is_present and seat["start_sit_time"] is not None:
            current_sit_minutes = (now - seat["start_sit_time"]) / 60.0
        else:
            current_sit_minutes = 0.0

    num_previous_returns = seat["num_returns"]
    seat_now_occupied = 1 if is_present else 0

    # NEW PERSON RULE: 1 if someone arrived/returned very recently
    if seat["new_person_present"] and (now - seat["last_return_time"] < 3.0):
        new_person_present = 1
    else:
        new_person_present = 0

    return {
        "name": "",
        "item_list": item_list,
        "owner_side": owner_side,
        "weekday": weekday,
        "time_since_person": time_since_person,
        "time_of_day": time_of_day,
        "current_sit_minutes": current_sit_minutes,
        "total_session_minutes": total_session_minutes,
        "num_previous_returns": num_previous_returns,
        "seat_now_occupied": seat_now_occupied,
        "new_person_present": new_person_present,
    }

cap = cv2.VideoCapture("FILL_ME") #main camera
cap_cam2 = cv2.VideoCapture("FILL_ME") #left face camera
cap_cam3 = cv2.VideoCapture(2, cv2.CAP_DSHOW) #right face camera
cv2.namedWindow("OOPS")
cv2.setMouseCallback("OOPS", mouse_callback)

frame_id = 0 #frame counter
owner_left = None #current left seat owner
owner_right = None #current right seat owner
name_cam2 = "Unknown" #left camera recognized name
name_cam3 = "Unknown" #right camera recognized name
person = {} #map of pose track IDs to recognized names

#seat state tracking for left and right seats
seat_state = {
    "left": {
        "current_owner": None,
        "owner_name": None,
        "last_seen_time": None,
        "session_start_time": None,
        "start_sit_time": None,
        "last_return_time": None,
        "num_returns": 0,
        "is_present": False,
        "new_person_present": False,
    },
    "right": {
        "current_owner": None,
        "owner_name": None,
        "last_seen_time": None,
        "session_start_time": None,
        "start_sit_time": None,
        "last_return_time": None,
        "num_returns": 0,
        "is_present": False,
        "new_person_present": False,
    },
}

prev_seat_state = {
    "left": {
        "current_owner": None,
        "owner_name": None,
        "last_seen_time": None,
        "session_start_time": None,
        "start_sit_time": None,
        "last_return_time": None,
        "num_returns": 0,
        "is_present": False,
        "new_person_present": False,
    },
    "right": {
        "current_owner": None,
        "owner_name": None,
        "last_seen_time": None,
        "session_start_time": None,
        "start_sit_time": None,
        "last_return_time": None,
        "num_returns": 0,
        "is_present": False,
        "new_person_present": False,
    },
}

predict_every = 200 #how often to run left-behind item prediction
current_left_person = None #current left seat person
current_right_person = None #current right seat person

#-----MAIN LOOP-----
while True:
    try:
        parent_pid = psutil.Process(os.getpid()).ppid()
        if parent_pid == 1:
            print("Parent died → UI STOP pressed → exiting cleanly.")
            break
    except Exception:
        # If psutil fails, also exit to be safe
        print("psutil check failed → exiting.")
        break

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) #flip frame horizontally

    frame_id += 1 #increment frame counter
    if frame_id % 3 !=0: #skip every 3 frames 
        continue

    frame = cv2.resize(frame, (960, 540)) #resize frame
    raw_frame = frame.copy()

    #read face cameras
    ret2, frame2 = cap_cam2.read()
    ret3, frame3 = cap_cam3.read()

    #resize face camera frames for overlay
    if ret2:
        frame2_small = cv2.resize(frame2, (160, 100))
    if ret3:
        frame3_small = cv2.resize(frame3, (160, 100))

    #draw table points
    for p in points:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)

    #4 points clicked → define table polygon
    if len(points) == 4:
        pts = np.array(points, dtype=np.float32)

        #sort into: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        polygon = np.array([tl, tr, br, bl], dtype=np.int32)

        #draw rectangle 
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        if not table_defined:
            print("TABLE DEFINED:", polygon)
            table_defined = True #mark table as defined
    elif len(points) >= 2:
        #while user is still clicking (<4 points), just connect them
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

    #table already defined
    if table_defined:
        #-----POSE DETECTION-----
        #pose detection results with tracking
        pose_results = pose_model.track(
            frame, device=DEVICE_STR, persist=True, iou=0.1, verbose=False
        )
        chest_positions = {} #chest positions by track ID
        wrist_positions = {} #wrist positions by track ID
        kpts_by_track = {} #keypoints by track ID

        left_candidates, right_candidates = [], [] #candidates for left and right seat owners

        table_center_x = (polygon[0][0] + polygon[2][0]) / 2 #table center x-coordinate
        table_top_y = min(p[1] for p in polygon) #table top y-coordinate
        left_compare_pt = (int(table_center_x - 130), table_top_y) #left reference point
        right_compare_pt = (int(table_center_x + 130), table_top_y) #right reference point

        #for each pose detected
        for i, box in enumerate(pose_results[0].boxes):
            track_id = int(box.id.cpu().item()) if box.id is not None else -1
            raw = pose_results[0].keypoints[i].data[0]
            kpts, kconf = safe_extract_keypoints(raw) #keypoints and confidences
            kpts_by_track[track_id] = (kpts, kconf) #store keypoints and confidences by track ID

            #check if chest above table (valid seating person)
            if not is_chest_above_table(kpts, kconf, polygon):
                continue

            chest = get_chest_midpoint(kpts, kconf) #get chest midpoint
            if chest is None:
                continue
            fx, fy = chest
            chest_positions[track_id] = (fx, fy) #store chest position by track ID

            LEFT_WRIST_IDX = 9
            RIGHT_WRIST_IDX = 10
            lw = (
                (int(kpts[LEFT_WRIST_IDX][0]), int(kpts[LEFT_WRIST_IDX][1]))
                if kconf[LEFT_WRIST_IDX] > 0.2
                else None
            )
            rw = (
                (int(kpts[RIGHT_WRIST_IDX][0]), int(kpts[RIGHT_WRIST_IDX][1]))
                if kconf[RIGHT_WRIST_IDX] > 0.2
                else None
            )
            wrist_positions[track_id] = (lw, rw) #store wrist positions by track ID

            #determine proximity to table reference points
            compare_pt = left_compare_pt if fx < table_center_x else right_compare_pt
            near, dist = is_near_table((fx, fy), compare_pt, max_dist=200)
            #skip if not near table
            if not near:
                continue
            
            #determine left/right candidates based on chest x-coordinate
            if fx < table_center_x:
                left_candidates.append((track_id, fy))
            else:
                right_candidates.append((track_id, fy))

            x_vals = kpts[:, 0]
            y_vals = kpts[:, 1]

            # ignore invalid keypoints (0,0 or too far)
            valid_mask = (x_vals > 0) & (y_vals > 0) & (kconf > 0.1)
            #draw bbox for valid persons
            if valid_mask.sum() >= 5:
                x1 = int(x_vals[valid_mask].min())
                y1 = int(y_vals[valid_mask].min())
                x2 = int(x_vals[valid_mask].max())
                y2 = int(y_vals[valid_mask].max())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 200, 255), 1)
                cv2.putText(frame, f"ID:{track_id} {person.get(str(track_id), '')}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 2)

        #determine left/right seat owners based on closest candidates (closest to top of table)
        owner_left = min(left_candidates, key=lambda x: x[1])[0] if left_candidates else None
        #set owner name to "Unknown" if recognized for the first time
        if owner_left is not None and person.get(f"{owner_left}") is None:
            person[f"{owner_left}"] = "Unknown"

        owner_right = min(right_candidates, key=lambda x: x[1])[0] if right_candidates else None
        if owner_right is not None and person.get(f"{owner_right}") is None:
            person[f"{owner_right}"] = "Unknown"

        #draw table reference points (left and right)
        cv2.circle(frame, left_compare_pt, 7, (255, 0, 0), -1)
        cv2.circle(frame, right_compare_pt, 7, (255, 0, 255), -1)

        #draw wrist keypoints
        for tid, (kpts, kconf) in kpts_by_track.items():
            lw = (int(kpts[9][0]), int(kpts[9][1]))
            rw = (int(kpts[10][0]), int(kpts[10][1]))
            if kconf[9] > 0.2:
                cv2.circle(frame, lw, 5, (0, 0, 255), -1)
            if kconf[10] > 0.2:
                cv2.circle(frame, rw, 5, (0, 0, 255), -1)

        #display current seat owners
        if owner_left is not None:
            cv2.putText(
                frame,
                f"LEFT OWNER: {owner_left} {person[f'{owner_left}']}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 137),
                2,
            )
        if owner_right is not None:
            cv2.putText(
                frame,
                f"RIGHT OWNER: {owner_right} {person[f'{owner_right}']}",
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 137),
                2,
            )

        #-----OBJECT DETECTION-----
        #object detection results with tracking
        if frame_id % 3 == 0:
            merged_dets = run_dual_inference(raw_frame)
            current_objects = {}
            used_prev_ids = set()

            current_obj_ids = set() #set of current object IDs in frame
            #for each object detected
            for det in merged_dets:
                box = det["xyxy"]
                obj_name = det["name"]
                conf = det["conf"]
                print(obj_name)

                best_iou = 0
                best_id = None

                # match with previous object IDs
                for oid, pdata in prev_objects.items():
                    i = iou(box, pdata["xyxy"])
                    if i > best_iou:
                        best_iou = i
                        best_id = oid

                # if sufficiently similar → keep same ID
                if best_iou > IOU_TRACK_THRESHOLD:
                    obj_id = best_id
                    used_prev_ids.add(best_id)
                else:
                    # assign new ID
                    obj_id = next_object_id
                    next_object_id += 1

                # store current object
                current_objects[obj_id] = {
                    "xyxy": box,
                    "name": det["name"],
                    "conf": det["conf"],
                }
                current_obj_ids.add(obj_id)

            prev_objects = current_objects.copy()

            # obj_id now comes from the tracker
            for obj_id, data in current_objects.items():
                x1, y1, x2, y2 = data["xyxy"]
                obj_name = data["name"]
                conf = data["conf"]

            # Skip unwanted classes
                if obj_name in ["chair", "person", "dining table"]:
                    continue

                cx, cy = compute_center(x1, y1, x2, y2) #center of object bbox

                #store object bbox positions
                object_centers[obj_id] = {
                    "cx": cx,
                    "left": x1,
                    "right": x2,
                    "top": y1,
                    "bottom": y2,
                }
                final_objects[obj_id] = {
                    "name": obj_name,
                    "bbox": (x1, y1, x2, y2),
                    "owner": object_owners.get(obj_id, None),
                }
                exit_objects = final_objects.copy()
                exit_clean_frame = frame.copy()

                object_names[obj_id] = obj_name #store object name by tracker ID

                #check if object bbox is sufficiently inside table polygon
                inside, percent = bbox_inside_polygon_percentage(
                    (x1, y1, x2, y2), polygon, min_percent=60
                )
                if not inside:
                    continue

                #assign object owner if not already assigned
                if obj_id not in object_owners:
                    assigned = None

                    #very far distance for comparison of left/right wrists to the object
                    left_d = right_d = 1e9

                    #if owner of seat detected
                    if owner_left is not None:
                        print("OWNER LEFT:", owner_left)
                        lw_pair = wrist_positions.get(owner_left) #get wrist position of the owner
                        if lw_pair:
                            candidates = [p for p in lw_pair if p is not None] #left/right wrist
                            if candidates:
                                #decide distance of nearest wrist to object center
                                nearest_l = min(
                                    candidates, key=lambda p: distance((cx, cy), p)
                                )
                                left_d = distance((cx, cy), nearest_l)
                                print(f"{obj_id} left_d:", left_d)

                    if owner_right is not None:
                        print("OWNER RIGHT:", owner_right)
                        rw_pair = wrist_positions.get(owner_right)
                        if rw_pair:
                            candidates = [p for p in rw_pair if p is not None]
                            if candidates:
                                nearest_r = min(
                                    candidates, key=lambda p: distance((cx, cy), p)
                                )
                                right_d = distance((cx, cy), nearest_r)
                                print(f"{obj_id} right_d:", right_d)
                    #if both territory lines defined
                    if LEFT_TERRITORY_X is not None and RIGHT_TERRITORY_X is not None:
                        #if inside left territory, assign left
                        if cx <= LEFT_TERRITORY_X:
                            assigned = "left"
                            print(f"{obj_id} inside left territory")
                        #if inside right territory, assign right
                        elif cx >= RIGHT_TERRITORY_X:
                            assigned = "right"
                            print(f"{obj_id} inside right territory")
                        #if between both territory lines, assign based on wrist distances
                        else:
                            assigned = "left" if left_d <= right_d else "right"
                            print(f"{obj_id} assigned to:", assigned)
                    #if only left territory line defined
                    elif LEFT_TERRITORY_X is not None:
                        #assigned = "left" if cx <= LEFT_TERRITORY_X else "right"
                        #assign left
                        if cx <= LEFT_TERRITORY_X:
                            assigned = "left"
                        else:
                            assigned = "left" if left_d <= right_d else "right"
                            print(f"{obj_id} assigned to:", assigned)
                    #if only right territory line defined
                    elif RIGHT_TERRITORY_X is not None:
                        #assigned = "right" if cx >= RIGHT_TERRITORY_X else "left"
                        #assign right 
                        if cx >= RIGHT_TERRITORY_X:
                            assigned = "right"
                        else:
                            assigned = "left" if left_d <= right_d else "right"
                            print(f"{obj_id} assigned to:", assigned)
                    #both territory lines undefined
                    elif left_d or right_d:
                        #assign based on wrist distances
                        assigned = "left" if left_d <= right_d else "right"
                    #both territory lines and wrist distances undefined
                    else:
                        # Last fallback: table center
                        assigned = "left" if cx < table_center_x else "right"

                    object_owners[obj_id] = assigned #assign owner (left/right) to object

                owner = object_owners.get(obj_id) #get object owner
                #set color based on owner side
                if owner == "left":
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)
                #draw object bbox and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    frame,
                    f"{obj_name}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            #remove objects no longer detected
            for oid in list(object_centers.keys()):
                if oid not in current_obj_ids:
                    object_centers.pop(oid)
                    object_owners.pop(oid, None)
                    object_names.pop(oid, None)

            update_territory_lines() #update territory lines

            #draw territory lines
            if LEFT_TERRITORY_X is not None:
                cv2.line(
                    frame,
                    (LEFT_TERRITORY_X, polygon[:, 1].min()),
                    (LEFT_TERRITORY_X, polygon[:, 1].max()),
                    (255, 0, 0),
                    3,
                )
            if RIGHT_TERRITORY_X is not None:
                cv2.line(
                    frame,
                    (RIGHT_TERRITORY_X, polygon[:, 1].min()),
                    (RIGHT_TERRITORY_X, polygon[:, 1].max()),
                    (0, 255, 255),
                    3,
                )

            #draw chest positions and distances to table reference points
            for track_id, (fx, fy) in chest_positions.items():
                compare_pt = left_compare_pt if fx < table_center_x else right_compare_pt
                near, dist = is_near_table((fx, fy), compare_pt)
                cv2.circle(frame, (fx, fy), 6, (0, 255, 255), -1)
                cx2, cy2 = compare_pt
                cv2.line(frame, (fx, fy), (cx2, cy2), (255, 255, 255), 1)
                label_x, label_y = (fx + cx2) // 2, (fy + cy2) // 2 - 5
                cv2.putText(
                    frame,
                    f"{int(dist)}px",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        #-----FACE RECOGNITION-----
        #left side camera
        if ret2:
            frame2, results2 = run_face_recognition_frame(frame2)

            if results2:
                # we found a face → use name (may be Unknown)
                name_cam2 = results2[0][0]
            else:
                # no face → treat as empty
                name_cam2 = ""

            col2 = (0, 0, 255) if name_cam2 == "Unknown" else (0, 255, 0)
            frame2_small = cv2.resize(frame2, (160, 100))

        #right side camera
        if ret3:
            frame3, results3 = run_face_recognition_frame(frame3)

            if results3:
                # we found a face → use name (may be Unknown)
                name_cam3 = results3[0][0]
            else:
                # no face → treat as empty
                name_cam3 = ""   

            col3 = (0,0,255) if name_cam3 == "Unknown" else (0,255,0)
            frame3_small = cv2.resize(frame3, (160, 100))

        # Overlay face camera frames
        H, W, _ = frame.shape
        if ret2:
            overlay_frame(frame, frame2_small, W - 160, 30)
            cv2.putText(
                frame,
                f"LEFT {name_cam2}",
                (W - 155, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col2,
                2,
            )
        if ret3:
            overlay_frame(frame, frame3_small, W - 160, 170)
            cv2.putText(
                frame,
                f"RIGHT {name_cam3}",
                (W - 155, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col3,
                2,
            )

        #left side
        if owner_left is None:
            #no body near left seat → treat as empty
            current_left_person = None
        else:
            #there IS a pose track for left
            if ret2 and 'results2' in locals() and results2:
                face_name, face_loc, prob = results2[0]
                if face_name != "Unknown": #update only if recognized
                    print("LEFT OWNER RECOGNIZED AS:", face_name)
                    person[f'{owner_left}'] = face_name #map track ID to name
                    current_left_person  = person.get(str(owner_left), "Unknown") if owner_left is not None else None #current left person name
            else:
                # no face detected but pose still there → reuse last known name (or None)
                current_left_person = person.get(f"{owner_left}", None)

        #right side
        if owner_right is None:
            # no body near right seat → treat as empty
            current_right_person = None
        else:
            if ret3 and 'results3' in locals() and results3:
                face_name, face_loc, prob = results3[0]
                if face_name != "Unknown":
                    print("RIGHT OWNER RECOGNIZED AS:", face_name)
                    person[f'{owner_right}'] = face_name
                    current_right_person = person.get(str(owner_right), "Unknown") if owner_right is not None else None
            else:
                current_right_person = person.get(f"{owner_right}", None)


        #-----DECISION + LOGGING (ML model)-----
        #predict if table is defined and at prediction interval
        if frame_id % predict_every == 0 and table_defined: 
            #update seat states with current persons
            update_seat_state("left", owner_left, seat_state, prev_seat_state, person)
            update_seat_state("right", owner_right, seat_state, prev_seat_state, person)

            #for each side (left and right)
            for side in ("left", "right"):
                print(f"--- Checking side: {side} ---")
                seat = seat_state[side]
                feats = compute_seat_features(side, seat_state) #parameters
                ml_label = "TEMPORARY_LEAVE"
                prob = None
                thr = None
                if USE_LEFT_BEHIND_MODEL:
                    if not feats["item_list"]:
                        print(f"[{side}] no ML items, skip ML")
                    else:
                        prob, ml_label, thr = predict_status(
                            item_types=feats["item_list"],
                            weekday=feats["weekday"],
                            time_since_person=feats["time_since_person"],
                            time_of_day=feats["time_of_day"],
                            current_sit_minutes=feats["current_sit_minutes"],
                            total_session_minutes=feats["total_session_minutes"],
                            num_previous_returns=feats["num_previous_returns"],
                            seat_now_occupied=feats["seat_now_occupied"],
                            new_person_present=feats["new_person_present"]
                        )
                        print(
                            f"[{side}] ML Prediction: {ml_label} "
                            f"(prob={prob[ml_label]:.3f}, thr={thr})"
                        )
                else:
                    print(f"[{side}] MODEL DISABLED")

                if seat["is_present"]:
                    seat_label[side] = "seated"
                else:
                    if ml_label == "LEFT_BEHIND":
                        seat_label[side] = "left_behind"
                    else:
                        seat_label[side] = "temporary_leave"

                if USE_LEFT_BEHIND_MODEL:
                    if seat["is_present"] and not DEBUG_FORCE_CAPTURE:
                        already_logged[side] = False
                        continue
                    should_capture = (ml_label == "LEFT_BEHIND") or DEBUG_FORCE_CAPTURE
                else:
                    if seat["is_present"] and not DEBUG_FORCE_CAPTURE:
                        already_logged[side] = False
                        continue

                    should_capture = (not seat["is_present"]) or DEBUG_FORCE_CAPTURE

                if should_capture and not already_logged[side]:
                    log_live(
                        f"[{side.upper()}] Saving left-behind items "
                        f"(model={USE_LEFT_BEHIND_MODEL}, ml_label={ml_label}, "
                        f"items={feats['item_list']})"
                    )

                    for obj_id, owner in object_owners.items():
                        if owner != side:
                            continue
                        cen = object_centers.get(obj_id)
                        if not cen:
                            continue

                        name = object_names.get(obj_id, "Unknown")
                        x1 = cen["left"]
                        x2 = cen["right"]
                        y1 = cen["top"]
                        y2 = cen["bottom"]

                        save_left_item_event(
                            frame=raw_frame,
                            obj_id=obj_id,
                            item_name=name,
                            owner_side=side,
                            bbox=(x1, y1, x2, y2),
                        )

                    already_logged[side] = True

            # TULIS STATUS UNTUK FLUTTER (bullets)
            try:
                with open(STATUS_OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "left": {
                                "status": seat_label["left"],
                                "is_present": seat_state["left"]["is_present"],
                            },
                            "right": {
                                "status": seat_label["right"],
                                "is_present": seat_state["right"]["is_present"],
                            },
                            "timestamp": time.time(),
                        },
                        f,
                    )
            except Exception as e:
                log_live(f"[WARN] failed writing live_status.json: {e}")

        if frame_id % 3 == 0:
            try:
                cv2.imwrite(str(FRAME_OUTPUT_PATH), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            except Exception as e:
                print("")

    cv2.imshow("OOPS", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("r"):
        points = []
        table_defined = False
        print("Table RESET")

    # Q = MANUAL CAPTURE (paksa save semua barang di meja, TIDAK exit)
    if key == ord("q"):
        if table_defined and polygon is not None:
            print("\ncapturing ALL items on table\n")
            debug_capture_all_items_on_table(raw_frame, polygon)  # 👉 pakai raw_frame
        else:
            print("Table not defined yet, skip capture.")
        continue

    if key == 27:  # ESC
        try:
            if FRAME_OUTPUT_PATH.exists():
                os.remove(FRAME_OUTPUT_PATH)
        except Exception as e:
            print("Failed to remove live_frame.jpg on exit:", e)
        break

cap.release()
cv2.destroyAllWindows()

