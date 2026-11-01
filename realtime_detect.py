# realtime_detect.py - Real-time Bangladeshi banknote recognition with OpenCV
# Keys:  q = quit | r = toggle guide box / full frame | s = save screenshot
#        + / - = raise / lower confidence threshold | m = mirror on/off

import time, argparse
from collections import deque

import cv2
import numpy as np
import torch
import timm

# ----------------------------- ARGS -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="best_banknote_model.pth")
ap.add_argument("--cam", type=int, default=0, help="camera index (0 = default webcam)")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--threshold", type=float, default=0.75)
ap.add_argument("--smooth", type=int, default=6, help="frames averaged for stable output")
args = ap.parse_args()

# ----------------------------- MODEL -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
CLASSES = ckpt["class_names"]
IMG = ckpt["img_size"]
MEAN = np.array(ckpt["mean"], dtype=np.float32)
STD = np.array(ckpt["std"], dtype=np.float32)

model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(device)
print(f"Loaded {ckpt['model_name']} | classes: {CLASSES} | device: {device}")

@torch.inference_mode()
def predict(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()

# warm-up (first inference is slow)
predict(np.zeros((IMG, IMG, 3), np.uint8))

# ----------------------------- CAMERA -----------------------------
cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") and cv2.os.name == "nt" \
    else cv2.VideoCapture(args.cam)
if not cap.isOpened():
    cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Try --cam 1 or close other apps using the camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always get the latest frame (low latency)

# ----------------------------- STATE -----------------------------
threshold = args.threshold
use_roi, mirror = True, True
history = deque(maxlen=args.smooth)
fps, t_prev, shot_id = 0.0, time.time(), 0
WIN = "Banknote Detector (q to quit)"

def draw_text(img, text, org, scale=0.8, color=(255, 255, 255), bg=None, thick=2):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

print("Camera on. Hold a note inside the box. Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera frame not received."); break
    if mirror:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # region of interest (note should fill this box)
    if use_roi:
        bw, bh = int(W * 0.55), int(H * 0.55)
        x1, y1 = (W - bw) // 2, (H - bh) // 2
        x2, y2 = x1 + bw, y1 + bh
    else:
        x1, y1, x2, y2 = 0, 0, W, H

    probs = predict(frame[y1:y2, x1:x2])
    history.append(probs)
    avg = np.mean(history, axis=0)                  # temporal smoothing
    order = avg.argsort()[::-1]
    top, conf = int(order[0]), float(avg[order[0]])
    confident = conf >= threshold and len(history) >= max(2, args.smooth // 2)

    color = (0, 200, 0) if confident else (0, 140, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    if confident:
        draw_text(frame, f"{CLASSES[top]} Taka  {conf*100:.0f}%", (x1, max(40, y1 - 14)),
                  scale=1.1, bg=color)
    else:
        draw_text(frame, "Show a note inside the box", (x1, max(40, y1 - 14)), bg=color)

    # top-3 panel
    for i, idx in enumerate(order[:3]):
        draw_text(frame, f"{CLASSES[idx]:>5}: {avg[idx]*100:5.1f}%", (15, 90 + i * 32), scale=0.7,
                  color=(0, 255, 0) if i == 0 else (220, 220, 220))

    # HUD
    now = time.time(); fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)); t_prev = now
    draw_text(frame, f"FPS {fps:.0f} | thr {threshold:.2f} | {'BOX' if use_roi else 'FULL'}",
              (15, 35), scale=0.65, color=(255, 255, 0))

    cv2.imshow(WIN, frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q") or k == 27:
        break
    elif k == ord("r"):
        use_roi = not use_roi; history.clear()
    elif k == ord("m"):
        mirror = not mirror; history.clear()
    elif k in (ord("+"), ord("=")):
        threshold = min(0.99, threshold + 0.05)
    elif k in (ord("-"), ord("_")):
        threshold = max(0.30, threshold - 0.05)
    elif k == ord("s"):
        name = f"capture_{shot_id:03d}.jpg"; cv2.imwrite(name, frame); shot_id += 1
        print("Saved", name)

cap.release()
cv2.destroyAllWindows()