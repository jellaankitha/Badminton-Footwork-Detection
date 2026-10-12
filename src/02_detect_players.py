from ultralytics import YOLO
import cv2
import os
import glob

print("Starting YOLO player detection...")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Find frames
frame_files = sorted(glob.glob("frames/*.jpg"))

print("Frames found:", len(frame_files))

# Create output folder
output_folder = "output/player_test"
os.makedirs(output_folder, exist_ok=True)

# Test only 20 frames
test_frames = frame_files[:20]

print("Testing:", len(test_frames), "frames")

for i, frame_path in enumerate(test_frames):

    print(f"Processing {i + 1}/20: {os.path.basename(frame_path)}")

    # Run YOLO
    results = model(frame_path, conf=0.25, verbose=False)

    # Read image
    image = cv2.imread(frame_path)

    # Draw detections
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            # COCO class 0 = person
            if class_id != 0:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Player {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Save image
    filename = os.path.basename(frame_path)

    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(output_path, image)

print()
print("--------------------------------")
print("Detection completed!")
print("Images saved:", len(test_frames))
print("Location:", output_folder)
print("--------------------------------")