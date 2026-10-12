import csv
import os
import glob

# Annotation CSV
csv_file = "annotations/videoplayback (1) copy_dataframe.csv"

# Folders
frames_folder = "frames"
selected_folder = "output/annotation_frames"

os.makedirs(selected_folder, exist_ok=True)

# Get available frames
frame_files = glob.glob(os.path.join(frames_folder, "frame_*.jpg"))

print("Total extracted frames:", len(frame_files))

selected = set()

# Read annotations
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        start = int(row["start"])
        end = int(row["end"])

        # Select frames every 10 frames inside the annotation
        for frame_number in range(start, end + 1):
            if frame_number % 10 == 0:
                selected.add(frame_number)

print("Frames selected:", len(selected))

# Copy matching frames
count = 0

for frame_number in sorted(selected):

    filename = f"frame_{frame_number:06d}.jpg"
    source = os.path.join(frames_folder, filename)

    if os.path.exists(source):
        destination = os.path.join(selected_folder, filename)

        with open(source, "rb") as src:
            with open(destination, "wb") as dst:
                dst.write(src.read())

        count += 1

print("--------------------------------")
print("Selected frames copied:", count)
print("Location:", selected_folder)
print("--------------------------------")