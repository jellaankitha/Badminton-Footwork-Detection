import cv2
import os

# Put the EXACT name of one video from the videos folder here
video_path = r"videos/videoplayback (1) copy.mp4"

# Frames will be saved in the frames folder
output_folder = r"frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("Check the video filename.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Total frames:", total_frames)

frame_number = 0
saved_frames = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Save every 10th frame
    if frame_number % 10 == 0:
        filename = os.path.join(
            output_folder,
            f"frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved_frames += 1

    frame_number += 1

cap.release()

print("--------------------------------")
print("Frame extraction completed!")
print("Total frames:", frame_number)
print("Frames saved:", saved_frames)
print("--------------------------------")