import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# FILE PATHS
# ============================================================

model_path = "weights/best.pt"
video_path = "videos/videoplayback (1) copy.mp4"

output_video = "output/court_tracking_test.mp4"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(model_path)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {video_path}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = total_frames / fps


print()
print("VIDEO INFORMATION")
print("------------------")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")


# ============================================================
# TEST SECTION
# ============================================================

# Start at 80 seconds
start_seconds = 80

# Process only 30 seconds
test_seconds = 30

start_frame = int(
    start_seconds * fps
)

max_frames = int(
    test_seconds * fps
)


print()
print("TEST")
print("----")
print("Start:", start_seconds, "seconds")
print("Length:", test_seconds, "seconds")
print("Frames:", max_frames)


# Move video to starting frame
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)


# ============================================================
# ACTUAL COURT COORDINATES
# ============================================================

# Coordinates selected from frame_4800s.jpg
#
# 1 = top-left
# 2 = top-right
# 3 = bottom-right
# 4 = bottom-left

court_polygon = np.array(
    [
        (203, 135),
        (440, 134),
        (475, 330),
        (159, 329)
    ],
    dtype=np.int32
)


# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_count = 0

print()
print("Processing...")
print()


while frame_count < max_frames:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break


    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.5,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # DRAW COURT
    # --------------------------------------------------------

    cv2.polylines(
        frame,
        [court_polygon],
        True,
        (255, 0, 0),
        2
    )


    # --------------------------------------------------------
    # PLAYER DETECTIONS
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # PROCESS EACH PLAYER
        # ----------------------------------------------------

        for box, player_id, confidence in zip(
            boxes,
            ids,
            confidences
        ):

            x1, y1, x2, y2 = box


            # ------------------------------------------------
            # PLAYER FOOT POSITION
            # ------------------------------------------------

            # Bottom-center of bounding box.
            # This is a better approximation of
            # where the player's feet touch the court.

            foot_x = (x1 + x2) / 2
            foot_y = y2


            # ------------------------------------------------
            # CHECK WHETHER FEET ARE INSIDE COURT
            # ------------------------------------------------

            inside_court = cv2.pointPolygonTest(
                court_polygon,
                (
                    float(foot_x),
                    float(foot_y)
                ),
                False
            )


            # Ignore detections whose feet
            # are outside the court.

            if inside_court < 0:
                continue


            # ------------------------------------------------
            # DRAW PLAYER BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    int(x1),
                    int(y1)
                ),
                (
                    int(x2),
                    int(y2)
                ),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW FOOT POINT
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    int(foot_x),
                    int(foot_y)
                ),
                5,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # PLAYER ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Player {int(player_id)}",
                (
                    int(x1),
                    max(
                        20,
                        int(y1) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Conf: {float(confidence):.2f}",
                (
                    int(x1),
                    min(
                        height - 5,
                        int(y2) + 18
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(frame)


    frame_count += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_count % 100 == 0:

        print(
            f"Processed "
            f"{frame_count}/{max_frames}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


# ============================================================
# DONE
# ============================================================

print()
print("==============================")
print("DONE")
print("==============================")
print()
print("Output video:")
print(output_video)
print()
print("Frames processed:", frame_count)