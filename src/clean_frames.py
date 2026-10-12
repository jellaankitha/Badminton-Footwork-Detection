from ultralytics import YOLO
import cv2
import os
import shutil

# ==========================================
# SETTINGS
# ==========================================

INPUT_FOLDER = "frames"
OUTPUT_FOLDER = "frames_cleaned"

MODEL_PATH = "weights/best.pt"

CONFIDENCE = 0.30

# Blur threshold
BLUR_THRESHOLD = 40

# Similar-frame threshold
DIFFERENCE_THRESHOLD = 8

# Minimum number of players required
MIN_PLAYERS = 2


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded!")


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# GET ALL FRAMES
# ==========================================

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

files.sort()

print()
print("Total frames found:", len(files))
print("Cleaning started...")
print()


# ==========================================
# COUNTERS
# ==========================================

kept = 0
no_players = 0
one_player = 0
blurry = 0
duplicates = 0
bad_images = 0

previous_gray = None


# ==========================================
# PROCESS FRAMES
# ==========================================

for i, filename in enumerate(files):

    input_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(input_path)

    # --------------------------------------
    # INVALID IMAGE
    # --------------------------------------

    if image is None:
        bad_images += 1
        continue


    # ======================================
    # BLUR CHECK
    # ======================================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_value = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_value < BLUR_THRESHOLD:

        blurry += 1
        continue


    # ======================================
    # YOLO PLAYER DETECTION
    # ======================================

    results = model.predict(
        image,
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes


    player_count = 0

    if boxes is not None:
        player_count = len(boxes)


    # ======================================
    # NO PLAYERS
    # ======================================

    if player_count == 0:

        no_players += 1
        continue


    # ======================================
    # ONLY ONE PLAYER
    # ======================================

    if player_count < MIN_PLAYERS:

        one_player += 1
        continue


    # ======================================
    # DUPLICATE / VERY SIMILAR FRAME
    # ======================================

    resized_current = cv2.resize(
        gray,
        (160, 90)
    )

    if previous_gray is not None:

        resized_previous = cv2.resize(
            previous_gray,
            (160, 90)
        )

        difference = cv2.absdiff(
            resized_current,
            resized_previous
        ).mean()

        if difference < DIFFERENCE_THRESHOLD:

            duplicates += 1
            continue


    # ======================================
    # KEEP FRAME
    # ======================================

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    shutil.copy2(
        input_path,
        output_path
    )

    previous_gray = gray

    kept += 1


    # ======================================
    # PROGRESS
    # ======================================

    if (i + 1) % 500 == 0:

        print(
            f"Checked: {i + 1} | "
            f"Kept: {kept} | "
            f"No players: {no_players} | "
            f"One player: {one_player} | "
            f"Blurry: {blurry} | "
            f"Duplicates: {duplicates}"
        )


# ==========================================
# FINAL RESULT
# ==========================================

print()
print("==========================================")
print("          CLEANING COMPLETE")
print("==========================================")

print("Original frames :", len(files))
print("Frames kept     :", kept)

print()
print("Removed because:")
print("No players      :", no_players)
print("Only one player :", one_player)
print("Blurry          :", blurry)
print("Duplicates      :", duplicates)
print("Bad images      :", bad_images)

print()
print("Clean frames saved in:")
print(OUTPUT_FOLDER)

print("==========================================")