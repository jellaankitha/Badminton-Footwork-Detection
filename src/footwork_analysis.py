from ultralytics import YOLO
import cv2
import os

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "weights/best.pt"
INPUT_VIDEO = "videos/new_match.mp4"
OUTPUT_VIDEO = "output/new_match_footwork.mp4"

CONFIDENCE = 0.30

# -----------------------------
# FOOTWORK LABELS
# -----------------------------
LABELS = {
    "front_left": "Backhand Front Corner",
    "front_right": "Forehand Front Corner",
    "side_left": "Backhand Side",
    "side_right": "Forehand Side",
    "back_left": "Backhand Back Court Corner",
    "back_right": "Forehand Back Court Corner",
}

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Video:", width, "x", height)
print("FPS:", fps)

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("output", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------
    # TRACK ALL PLAYERS
    # --------------------------------------------
    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    detections = results[0].boxes

    # --------------------------------------------
    # PROCESS EVERY DETECTED PLAYER
    # --------------------------------------------
    if detections is not None:

        for box in detections:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            confidence = float(box.conf[0])

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # ----------------------------------------
            # TRACKING ID
            # ----------------------------------------
            if box.id is not None:
                player_id = int(box.id[0])
            else:
                player_id = 0

            # ----------------------------------------
            # CENTER OF PLAYER
            # ----------------------------------------
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # ----------------------------------------
            # NORMALIZED POSITION
            # ----------------------------------------
            nx = center_x / width
            ny = center_y / height

            # ----------------------------------------
            # DETERMINE FOOTWORK
            # ----------------------------------------
            # FRONT
            if ny < 0.40:

                if nx < 0.50:
                    footwork = LABELS["front_left"]
                else:
                    footwork = LABELS["front_right"]

            # BACK
            elif ny > 0.67:

                if nx < 0.50:
                    footwork = LABELS["back_left"]
                else:
                    footwork = LABELS["back_right"]

            # SIDE
            else:

                if nx < 0.50:
                    footwork = LABELS["side_left"]
                else:
                    footwork = LABELS["side_right"]

            # ----------------------------------------
            # DRAW PLAYER BOX
            # ----------------------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ----------------------------------------
            # PLAYER LABEL
            # ----------------------------------------
            text1 = "Player " + str(player_id)

            text2 = "Footwork: " + footwork

            text3 = "Confidence: " + str(round(confidence, 2))

            # ----------------------------------------
            # DISPLAY PLAYER
            # ----------------------------------------
            cv2.putText(
                frame,
                text1,
                (x1, max(y1 - 55, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ----------------------------------------
            # DISPLAY FOOTWORK
            # ----------------------------------------
            cv2.putText(
                frame,
                text2,
                (x1, max(y1 - 30, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # ----------------------------------------
            # DISPLAY CONFIDENCE
            # ----------------------------------------
            cv2.putText(
                frame,
                text3,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    # --------------------------------------------
    # SAVE FRAME
    # --------------------------------------------
    out.write(frame)

    # Progress in TERMINAL only
    if frame_number % 100 == 0:
        print("Processed:", frame_number)

# -----------------------------
# FINISH
# -----------------------------
cap.release()
out.release()

print()
print("DONE!")
print("Output saved to:")
print(OUTPUT_VIDEO)