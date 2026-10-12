import cv2
import csv
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
model_path = "weights/best.pt"
video_path = "videos/videoplayback (1) copy.mp4"

output_video = "output/footwork_test.mp4"
output_csv = "output/footwork_data.csv"

# -----------------------------
# Load model
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Resolution:", width, "x", height)

# -----------------------------
# Test settings
# Start at 60 seconds
# Process only 30 seconds
# -----------------------------
start_frame = int(fps * 60)
max_frames = int(fps * 30)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# -----------------------------
# Court polygon
# This is only for this camera view.
# -----------------------------
court_polygon = np.array([
    (0, 150),
    (500, 150),
    (410, 360),
    (0, 360)
], dtype=np.int32)

# -----------------------------
# Output video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

# -----------------------------
# Output CSV
# -----------------------------
csv_file = open(
    output_csv,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time",
    "player_id",
    "center_x",
    "center_y",
    "confidence"
])

# -----------------------------
# Process video
# -----------------------------
frame_number = 0

while frame_number < max_frames:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    # -----------------------------
    # Track players
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.5,
        verbose=False
    )

    result = results[0]

    # -----------------------------
    # Process detections
    # -----------------------------
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, player_id, confidence in zip(
            boxes,
            ids,
            confidences
        ):

            x1, y1, x2, y2 = box

            # Calculate player center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # -----------------------------
            # Court filtering
            # -----------------------------
            inside_court = cv2.pointPolygonTest(
                court_polygon,
                (float(center_x), float(center_y)),
                False
            )

            # Ignore detections outside the court
            if inside_court < 0:
                continue

            # -----------------------------
            # Save player movement data
            # -----------------------------
            csv_writer.writerow([
                start_frame + frame_number,
                round(
                    (start_frame + frame_number) / fps,
                    2
                ),
                int(player_id),
                round(center_x, 2),
                round(center_y, 2),
                round(float(confidence), 2)
            ])

            # -----------------------------
            # Draw player bounding box
            # -----------------------------
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Draw player ID
            # -----------------------------
            cv2.putText(
                frame,
                f"Player {int(player_id)}",
                (
                    int(x1),
                    max(20, int(y1) - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # Draw court polygon
    # -----------------------------
    cv2.polylines(
        frame,
        [court_polygon],
        True,
        (255, 0, 0),
        2
    )

    # -----------------------------
    # Write output frame
    # -----------------------------
    writer.write(frame)

    frame_number += 1

    if frame_number % 100 == 0:
        print(
            f"Processed {frame_number}/{max_frames}"
        )

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
writer.release()
csv_file.close()

print()
print("DONE!")
print("Video:", output_video)
print("CSV:", output_csv)


            