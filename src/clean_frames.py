from ultralytics import YOLO
import cv2
import os
import shutil
import csv
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

# Project folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "frames"
OUTPUT_FOLDER = BASE_DIR / "frames_cleaned"

# Pretrained YOLO model for detecting PEOPLE.
# This is different from your 7-class footwork model.
PERSON_MODEL_PATH = "yolov8n.pt"

CONFIDENCE = 0.30
BLUR_THRESHOLD = 40
MIN_PLAYERS = 2

# Print progress after every 500 frames
PROGRESS_INTERVAL = 500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==========================================
# CHECK FOLDERS
# ==========================================

if not INPUT_FOLDER.is_dir():
    raise FileNotFoundError(
        f"Input folder not found: {INPUT_FOLDER}\n"
        "Make sure 'frames' is beside this Python file."
    )

# Prevent accidentally copying output into the input folder.
if INPUT_FOLDER.resolve() == OUTPUT_FOLDER.resolve():
    raise ValueError("Input and output folders must be different.")

# Create a fresh output folder for this run.
# The existing output is preserved by renaming it.
if OUTPUT_FOLDER.exists():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = OUTPUT_FOLDER.with_name(
        f"{OUTPUT_FOLDER.name}_backup_{timestamp}"
    )

    OUTPUT_FOLDER.rename(backup_folder)
    print(f"Previous output preserved at: {backup_folder}")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_PATH = BASE_DIR / "cleaning_report.csv"

# ==========================================
# LOAD PERSON DETECTOR
# ==========================================

print("Loading pretrained YOLO person detector...")
print("The first run may download yolov8n.pt.")

model = YOLO(PERSON_MODEL_PATH)

print("Person detector loaded!")

# ==========================================
# GET ALL IMAGE FILES
# ==========================================

files = sorted(
    path for path in INPUT_FOLDER.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

total = len(files)

print(f"\nTotal frames found: {total}")
print("Cleaning started...\n")

# ==========================================
# COUNTERS
# ==========================================

kept = 0
no_players = 0
one_player = 0
blurry = 0
bad_images = 0
detection_errors = 0

# ==========================================
# PROCESS FRAMES
# ==========================================

with open(
    REPORT_PATH, "w", newline="", encoding="utf-8"
) as report_file:

    writer = csv.writer(report_file)
    writer.writerow([
        "filename",
        "status",
        "reason",
        "person_count",
        "blur_score"
    ])

    for i, image_path in enumerate(files, start=1):

        person_count = 0
        blur_value = ""

        try:
            image = cv2.imread(str(image_path))

            # ----------------------------------
            # INVALID IMAGE
            # ----------------------------------

            if image is None:
                bad_images += 1
                writer.writerow([
                    image_path.name, "REJECTED",
                    "Unreadable image", "", ""
                ])
                continue

            # ----------------------------------
            # BLUR CHECK
            # ----------------------------------

            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            blur_value = round(
                cv2.Laplacian(
                    gray, cv2.CV_64F
                ).var(), 2
            )

            if blur_value < BLUR_THRESHOLD:
                blurry += 1
                writer.writerow([
                    image_path.name, "REJECTED",
                    "Too blurry", "", blur_value
                ])
                continue

            # ----------------------------------
            # DETECT PEOPLE, NOT FOOTWORK CLASSES
            # ----------------------------------

            results = model.predict(
                source=image,
                conf=CONFIDENCE,
                classes=[0],  # COCO class 0 = person
                verbose=False
            )

            boxes = results[0].boxes

            if boxes is not None:
                person_count = len(boxes)

            # ----------------------------------
            # NO PERSON DETECTED
            # ----------------------------------

            if person_count == 0:
                no_players += 1
                writer.writerow([
                    image_path.name, "REJECTED",
                    "No people detected",
                    person_count, blur_value
                ])
                continue

            # ----------------------------------
            # ONLY ONE PERSON DETECTED
            # ----------------------------------

            if person_count < MIN_PLAYERS:
                one_player += 1
                writer.writerow([
                    image_path.name, "REJECTED",
                    "Fewer than 2 people detected",
                    person_count, blur_value
                ])
                continue

            # ----------------------------------
            # KEEP FRAME
            # ----------------------------------

            output_path = OUTPUT_FOLDER / image_path.name

            shutil.copy2(image_path, output_path)

            kept += 1

            writer.writerow([
                image_path.name, "KEPT",
                "At least 2 people detected",
                person_count, blur_value
            ])

        except Exception as error:
            detection_errors += 1

            writer.writerow([
                image_path.name, "ERROR",
                str(error), person_count, blur_value
            ])

        # --------------------------------------
        # PROGRESS
        # --------------------------------------

        if i % PROGRESS_INTERVAL == 0 or i == total:
            print(
                f"Checked: {i}/{total} | "
                f"Kept: {kept} | "
                f"No people: {no_players} | "
                f"One person: {one_player} | "
                f"Blurry: {blurry} | "
                f"Bad images: {bad_images} | "
                f"Errors: {detection_errors}"
            )

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 45)
print("             CLEANING COMPLETE")
print("=" * 45)

print(f"Original frames : {total}")
print(f"Frames kept     : {kept}")

print("\nRejected frames:")
print(f"No people       : {no_players}")
print(f"One person      : {one_player}")
print(f"Blurry          : {blurry}")
print(f"Unreadable      : {bad_images}")
print(f"Errors          : {detection_errors}")

print(f"\nClean frames: {OUTPUT_FOLDER}")
print(f"Full report : {REPORT_PATH}")

print("\nOriginal frames have NOT been deleted.")
print("=" * 45)