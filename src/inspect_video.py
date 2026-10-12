import cv2
import os

video_path = "videos/videoplayback (1) copy.mp4"
output_dir = "output/court_samples"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", total_frames)
print("Duration:", round(total_frames / fps, 2), "seconds")

# Sample every 5 minutes
sample_times = [
    0,
    60,
    300,
    600,
    900,
    1200,
    1800,
    2400,
    3000,
    3600,
    4200,
    4800
]

for seconds in sample_times:

    frame_number = int(seconds * fps)

    if frame_number >= total_frames:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print("Could not read:", seconds, "seconds")
        continue

    filename = f"{output_dir}/frame_{seconds}s.jpg"

    cv2.imwrite(filename, frame)

    print("Saved:", filename)

cap.release()

print()
print("DONE")