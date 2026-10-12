import cv2
from ultralytics import YOLO

model = YOLO("weights/best.pt")

video_path = "videos/videoplayback (1) copy.mp4"
output_path = "output/tracking_test_30sec.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

max_frames = int(fps * 30)

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_number = 0

while cap.isOpened() and frame_number < max_frames:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()
    writer.write(annotated_frame)

    if frame_number % 100 == 0:
        print(f"Processed {frame_number}/{max_frames} frames")

    frame_number += 1

cap.release()
writer.release()

print("Tracking test completed.")
print(f"Saved video to: {output_path}")