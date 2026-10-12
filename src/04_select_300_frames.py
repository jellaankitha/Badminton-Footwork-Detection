import os
import glob
import shutil

# Input folder
input_folder = "output/annotation_frames"

# Output folder
output_folder = "output/roboflow_300"

os.makedirs(output_folder, exist_ok=True)

# Get all selected frames
frames = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

print("Available frames:", len(frames))

# Number of frames we want
number_to_select = 300

# Make sure we don't request more than available
number_to_select = min(number_to_select, len(frames))

# Select frames evenly across the whole dataset
step = len(frames) / number_to_select

selected_frames = []

for i in range(number_to_select):
    index = int(i * step)
    selected_frames.append(frames[index])

# Copy selected frames
for frame in selected_frames:
    filename = os.path.basename(frame)

    destination = os.path.join(output_folder, filename)

    shutil.copy2(frame, destination)

print("--------------------------------")
print("Selected frames:", len(selected_frames))
print("Saved to:", output_folder)
print("--------------------------------")