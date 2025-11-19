"""
face_detection_plus.py
All-in-one: Face (blur/eyes/smile), Hand Tracking, Object Detection (optional),
QR detection, simple expression (smile) analysis, and a Tkinter control panel.

Usage:
    python face_detection_plus.py

Notes:
- Optional MobileNet-SSD model files (for object detection) should be placed in the same folder:
    - MobileNetSSD_deploy.prototxt
    - MobileNetSSD_deploy.caffemodel
  If not found, the script will skip object detection gracefully.
- Demo sample path (user-uploaded): /mnt/data/5a28592e-27e9-4df5-b049-23dc5adfff9c.png
"""

import cv2
import time
import threading
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

# Try to import mediapipe for hand tracking (optional but recommended)
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    HAVE_MEDIAPIPE = False

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).parent.resolve()
DEMO_IMAGE_PATH = "/mnt/data/5a28592e-27e9-4df5-b049-23dc5adfff9c.png"  # uploaded demo sample
MOBILENET_PROTO = BASE_DIR / "MobileNetSSD_deploy.prototxt"
MOBILENET_MODEL = BASE_DIR / "MobileNetSSD_deploy.caffemodel"

# ---------------- DEFAULT STATE ----------------
state = {
    "use_blur": True,
    "use_eyes": True,
    "use_smile": True,
    "use_object": False,      # enable if you put model files in folder
    "use_hands": True if HAVE_MEDIAPIPE else False,
    "use_qr": True,
    "show_fps": True,
    "camera_index": 0,
    "running": False
}

# ---------------- LOAD CASCADES ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# ---------------- OPTIONAL: Object Detection (MobileNet-SSD) ----------------
object_net = None
OBJECT_CLASSES = []
if MOBILENET_PROTO.exists() and MOBILENET_MODEL.exists():
    try:
        object_net = cv2.dnn.readNetFromCaffe(str(MOBILENET_PROTO), str(MOBILENET_MODEL))
        # Standard MobileNet-SSD class names
        OBJECT_CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]
        state["use_object"] = True
        print("[INFO] MobileNet-SSD model loaded — Object detection enabled")
    except Exception as e:
        print("[WARN] Failed to load MobileNet-SSD model:", e)

# ---------------- OPTIONAL: Hand Tracking via MediaPipe ----------------
mp_hands = None
hands_detector = None
if HAVE_MEDIAPIPE and state["use_hands"]:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    print("[INFO] MediaPipe Hands available — Hand tracking enabled")
elif not HAVE_MEDIAPIPE and state["use_hands"]:
    print("[WARN] mediapipe not installed — hand tracking disabled")

# ---------------- QR Detector ----------------
qr_detector = cv2.QRCodeDetector()

# ---------------- UTILS ----------------
def lock_camera_props(cap):
    """Apply stable camera settings to reduce flicker."""
    # Lower resolution for stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # Try locking exposure (works on many webcams)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    except Exception:
        pass

def draw_label(img, text, tl):
    """Draw a filled label with text at top-left tl=(x,y)."""
    x, y = tl
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x-2, y-2), (x + w + 6, y + h + 6), (0,0,0), -1)
    cv2.putText(img, text, (x+2, y+h+2 - 6), font, scale, (255,255,255), thickness, cv2.LINE_AA)

# ---------------- VIDEO LOOP ----------------
def video_loop(status_label, canvas_label):
    cap = cv2.VideoCapture(state["camera_index"])
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot open camera index %s" % state["camera_index"])
        state["running"] = False
        return

    lock_camera_props(cap)

    prev = time.time()
    fps = 0
    counter = 0

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert grayscale for cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- FACE DETECTION ----------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            gray_roi = gray[y:y+h, x:x+w]
            label = "Face"

            # Blur face (privacy)
            if state["use_blur"]:
                k = (w//7)|1
                blurred = cv2.GaussianBlur(face_roi, (k, k), 0)
                frame[y:y+h, x:x+w] = blurred
            else:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # Eyes detection
            if state["use_eyes"]:
                eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255,255,0), 1)

            # Smile detection / basic expression
            if state["use_smile"]:
                smiles = smile_cascade.detectMultiScale(gray_roi, 1.7, 22)
                if len(smiles) > 0:
                    label = "Smiling"
                    # draw a small filled label
                    draw_label(frame, label, (x, y-20))

            # add name or face label if not shown already
            if label == "Face":
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # ---------------- OBJECT DETECTION (OPTIONAL) ----------------
        if state["use_object"] and object_net is not None:
            # prepare blob and forward
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            object_net.setInput(blob)
            detections = object_net.forward()
            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(OBJECT_CLASSES):
                        label = OBJECT_CLASSES[idx]
                    else:
                        label = f"obj_{idx}"
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 128, 255), 2)
                    draw_label(frame, f"{label} {confidence:.2f}", (startX, startY-18))

        # ---------------- HAND TRACKING (MediaPipe) ----------------
        if HAVE_MEDIAPIPE and state["use_hands"] and hands_detector is not None:
            # convert to RGB as mediapipe expects
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb)
            if res.multi_hand_landmarks:
                for hlandmarks in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlandmarks, mp_hands.HAND_CONNECTIONS)

        # ---------------- QR CODE DETECTION ----------------
        if state["use_qr"]:
            data, pts, _ = qr_detector.detectAndDecode(frame)
            if data:
                draw_label(frame, f"QR: {data}", (10, 10))

        # ---------------- FPS ----------------
        if state["show_fps"]:
            counter += 1
            if counter >= 10:
                now = time.time()
                fps = 10 / (now - prev)
                prev = now
                counter = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # ---------------- SHOW (OpenCV window) ----------------
        cv2.imshow("Face+Extras - Press Q to quit", frame)
        # also show demo image when paused or small preview (optional)
        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # save a snapshot
            p = BASE_DIR / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(p), frame)
            print("[SAVE] Snapshot saved to", p)

    cap.release()
    cv2.destroyAllWindows()
    state["running"] = False

# ---------------- GUI: simple control panel (launches video thread) ----------------
def start_stop(status_label, _):
    if not state["running"]:
        state["running"] = True
        t = threading.Thread(target=video_loop, args=(status_label, None), daemon=True)
        t.start()
        status_label.config(text="Status: Running")
    else:
        state["running"] = False
        status_label.config(text="Status: Stopping...")

def build_gui():
    root = tk.Tk()
    root.title("Face Detection Plus — Controls")
    root.geometry("420x260")

    top = ttk.Frame(root, padding=8)
    top.pack(fill=tk.BOTH, expand=True)

    status_label = ttk.Label(top, text="Status: Idle")
    status_label.pack(anchor="w")

    start_btn = ttk.Button(top, text="Start / Stop (Q to quit)", command=lambda: start_stop(status_label, None))
    start_btn.pack(fill=tk.X, pady=6)

    # toggles
    def make_chk(text, key):
        var = tk.IntVar(value=1 if state.get(key) else 0)
        chk = ttk.Checkbutton(top, text=text, variable=var, command=lambda: state.update({key: bool(var.get())}))
        chk.pack(anchor="w", pady=2)
    make_chk("Blur Faces", "use_blur")
    make_chk("Detect Eyes", "use_eyes")
    make_chk("Detect Smile", "use_smile")
    if object_net is not None:
        make_chk("Object Detection (MobileNet-SSD)", "use_object")
    else:
        lbl = ttk.Label(top, text="Obj detection: model not found (optional)", foreground="gray")
        lbl.pack(anchor="w", pady=2)
    if HAVE_MEDIAPIPE:
        make_chk("Hand Tracking (mediapipe)", "use_hands")
    else:
        lbl2 = ttk.Label(top, text="Hand tracking: mediapipe not installed (optional)", foreground="gray")
        lbl2.pack(anchor="w", pady=2)
    make_chk("QR Detection", "use_qr")
    make_chk("Show FPS", "show_fps")

    # Demo image preview button (opens the uploaded demo if exists)
    def show_demo():
        p = Path(DEMO_IMAGE_PATH)
        if p.exists():
            demo = cv2.imread(str(p))
            if demo is not None:
                cv2.imshow("Demo Sample", demo)
        else:
            messagebox.showinfo("Demo", f"Demo file not found: {DEMO_IMAGE_PATH}")

    demo_btn = ttk.Button(top, text="Show Demo Sample", command=show_demo)
    demo_btn.pack(fill=tk.X, pady=6)

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
    root.mainloop()

def on_close(root):
    state["running"] = False
    time.sleep(0.2)
    root.destroy()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("[INFO] Face Detection Plus — starting")
    build_gui()
