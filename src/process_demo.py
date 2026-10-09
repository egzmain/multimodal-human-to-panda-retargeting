import os
import csv
import math
import cv2
import numpy as np
import OpenEXR
import mediapipe as mp


# ============================================================
# DEMO 03 PATHS
# ============================================================

BASE = (
    r"C:\Users\kewin\humanoid_challenge\data"
    r"\demo_03\2026-10-09--14-59-53\EXR_RGBD"
)

RGB_DIR = os.path.join(BASE, "rgb")
DEPTH_DIR = os.path.join(BASE, "depth")

OUTPUT = (
    r"C:\Users\kewin\humanoid_challenge\data"
    r"\demo_03\hand_trajectory.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 60.0

mp_hands = mp.solutions.hands


# ============================================================
# LOAD DEPTH
# ============================================================

def load_depth(path):

    exr = OpenEXR.File(path)

    depth = np.array(
        exr.channels()["R"].pixels
    )

    return depth.astype(
        np.float32
    )


# ============================================================
# SORT FRAME FILES NUMERICALLY
# ============================================================

def numeric_files(
    folder,
    extension
):

    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith(extension)
    ]

    return sorted(
        files,
        key=lambda x: int(
            os.path.splitext(x)[0]
        )
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isdir(RGB_DIR):
    raise RuntimeError(
        f"RGB directory not found:\n{RGB_DIR}"
    )

if not os.path.isdir(DEPTH_DIR):
    raise RuntimeError(
        f"Depth directory not found:\n{DEPTH_DIR}"
    )


# ============================================================
# GET RGB FRAMES
# ============================================================

rgb_files = numeric_files(
    RGB_DIR,
    ".jpg"
)

print()
print("==============================")
print("PROCESSING DEMO 03")
print("==============================")
print()

print("RGB frames:", len(rgb_files))
print("RGB directory:")
print(RGB_DIR)

print()
print("Processing...")
print()


# ============================================================
# PROCESS
# ============================================================

rows = []

detected_hands = 0
valid_depth_values = 0


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for i, rgb_name in enumerate(
        rgb_files
    ):

        frame_number = int(
            os.path.splitext(
                rgb_name
            )[0]
        )

        rgb_path = os.path.join(
            RGB_DIR,
            rgb_name
        )

        depth_path = os.path.join(
            DEPTH_DIR,
            f"{frame_number}.exr"
        )


        # ----------------------------------------------------
        # Missing depth frame
        # ----------------------------------------------------

        if not os.path.exists(
            depth_path
        ):
            continue


        # ----------------------------------------------------
        # Load RGB
        # ----------------------------------------------------

        image = cv2.imread(
            rgb_path
        )

        if image is None:
            continue


        h, w = image.shape[:2]


        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(
            image_rgb
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        time_s = (
            frame_number / FPS
        )


        # Default values when no hand detected
        hand_x = math.nan
        hand_y = math.nan
        lidar_z = math.nan
        grasp = math.nan


        # ----------------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------------

        if result.multi_hand_landmarks:

            detected_hands += 1

            landmarks = (
                result
                .multi_hand_landmarks[0]
                .landmark
            )


            # Wrist
            wrist = landmarks[0]

            hand_x = wrist.x
            hand_y = wrist.y


            # Thumb tip
            thumb = landmarks[4]


            # Index fingertip
            index = landmarks[8]


            # ------------------------------------------------
            # GRASP FEATURE
            #
            # Normalized image-space distance between
            # thumb tip and index fingertip.
            # ------------------------------------------------

            grasp = math.sqrt(
                (thumb.x - index.x) ** 2
                +
                (thumb.y - index.y) ** 2
            )


            # ------------------------------------------------
            # LiDAR / DEPTH
            # ------------------------------------------------

            depth = load_depth(
                depth_path
            )

            dh, dw = depth.shape[:2]


            # Convert normalized MediaPipe wrist coordinates
            # into depth-image coordinates.
            px = int(
                np.clip(
                    wrist.x * dw,
                    0,
                    dw - 1
                )
            )

            py = int(
                np.clip(
                    wrist.y * dh,
                    0,
                    dh - 1
                )
            )


            depth_value = float(
                depth[py, px]
            )


            if (
                np.isfinite(depth_value)
                and
                depth_value > 0
            ):

                lidar_z = depth_value
                valid_depth_values += 1


        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        rows.append([
            frame_number,
            time_s,
            hand_x,
            hand_y,
            lidar_z,
            grasp
        ])


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if i % 100 == 0:

            print(
                f"{i}/{len(rgb_files)}"
            )


# ============================================================
# WRITE CSV
# ============================================================

with open(
    OUTPUT,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "video_time_s",
        "hand_x",
        "hand_y",
        "lidar_z",
        "grasp"
    ])

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

total_frames = len(rows)

if total_frames > 0:

    detection_percent = (
        detected_hands
        / total_frames
        * 100.0
    )

else:

    detection_percent = 0.0


print()
print("==============================")
print("DEMO 03 FINISHED")
print("==============================")
print()

print(
    "Frames processed:",
    total_frames
)

print(
    "Hand detections:",
    detected_hands
)

print(
    "Hand detection rate:",
    f"{detection_percent:.1f}%"
)

print(
    "Valid depth values:",
    valid_depth_values
)

print()
print("Saved:")
print(OUTPUT)