import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\kewin\humanoid_challenge"

EPISODE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "demo_03",
    "episodes"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "demo_03",
    "ml_dataset"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

TRAIN_EPISODES = list(range(1, 17))
TEST_EPISODES = list(range(17, 21))


# ============================================================
# PANDA WORKSPACE
#
# Same approximate workspace used by our working retargeting.
# ============================================================

ROBOT_X_MIN = 0.35
ROBOT_X_MAX = 0.50

ROBOT_Y_MIN = -0.12
ROBOT_Y_MAX = 0.12

ROBOT_Z_MIN = 0.30
ROBOT_Z_MAX = 0.46


# ============================================================
# LOAD EPISODES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            EPISODE_DIR,
            "episode_*.csv"
        )
    )
)

if len(files) != 20:
    print(
        f"WARNING: expected 20 episodes, "
        f"found {len(files)}"
    )

episodes = {}

for file in files:

    df = pd.read_csv(file)

    episode_number = int(
        df["episode"].iloc[0]
    )

    episodes[episode_number] = df


# ============================================================
# DETECT FEATURE COLUMN NAMES
# ============================================================

example = episodes[1]

print()
print("Available columns:")
print(list(example.columns))
print()


def find_column(possible_names):

    for name in possible_names:
        if name in example.columns:
            return name

    raise RuntimeError(
        "Could not find any of these columns: "
        + str(possible_names)
    )


HAND_X = find_column([
    "hand_x",
    "x",
    "wrist_x"
])

HAND_Y = find_column([
    "hand_y",
    "y",
    "wrist_y"
])

DEPTH = find_column([
    "lidar_z",
    "depth",
    "hand_z",
    "z"
])

GRASP = find_column([
    "grasp",
    "grasp_signal",
    "pinch"
])

CAP = find_column([
    "cap_smooth",
    "filtered",
    "capacitance"
])


print("Using columns:")
print("  hand X :", HAND_X)
print("  hand Y :", HAND_Y)
print("  depth  :", DEPTH)
print("  grasp  :", GRASP)
print("  cap    :", CAP)
print()


# ============================================================
# CLEAN EACH EPISODE
#
# Interpolate missing MediaPipe/depth measurements.
# ============================================================

for episode_number, df in episodes.items():

    df = df.copy()

    clean_columns = [
        HAND_X,
        HAND_Y,
        DEPTH,
        GRASP,
        CAP
    ]

    for column in clean_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = (
            df[column]
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

    episodes[episode_number] = df


# ============================================================
# CALIBRATION
#
# IMPORTANT:
# Only TRAIN episodes determine normalization.
# Test episodes do NOT influence these values.
# ============================================================

train_calibration = pd.concat(
    [
        episodes[i]
        for i in TRAIN_EPISODES
    ],
    ignore_index=True
)


def robust_range(series):

    values = series.dropna().to_numpy()

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    return float(low), float(high)


HAND_X_MIN, HAND_X_MAX = robust_range(
    train_calibration[HAND_X]
)

HAND_Y_MIN, HAND_Y_MAX = robust_range(
    train_calibration[HAND_Y]
)

DEPTH_MIN, DEPTH_MAX = robust_range(
    train_calibration[DEPTH]
)


print("====================================")
print("TRAINING CALIBRATION")
print("====================================")

print(
    f"Hand X: {HAND_X_MIN:.4f} "
    f"to {HAND_X_MAX:.4f}"
)

print(
    f"Hand Y: {HAND_Y_MIN:.4f} "
    f"to {HAND_Y_MAX:.4f}"
)

print(
    f"Depth : {DEPTH_MIN:.4f} "
    f"to {DEPTH_MAX:.4f}"
)

print()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value, minimum, maximum):

    denominator = maximum - minimum

    if abs(denominator) < 1e-8:
        return np.zeros_like(value)

    normalized = (
        value - minimum
    ) / denominator

    return np.clip(
        normalized,
        0.0,
        1.0
    )


# ============================================================
# RETARGET EACH EPISODE
# ============================================================

processed = []


for episode_number in sorted(episodes):

    df = episodes[episode_number].copy()

    nx = normalize(
        df[HAND_X].to_numpy(),
        HAND_X_MIN,
        HAND_X_MAX
    )

    ny = normalize(
        df[HAND_Y].to_numpy(),
        HAND_Y_MIN,
        HAND_Y_MAX
    )

    nz = normalize(
        df[DEPTH].to_numpy(),
        DEPTH_MIN,
        DEPTH_MAX
    )


    # --------------------------------------------------------
    # HUMAN -> PANDA MAPPING
    # --------------------------------------------------------

    robot_x = (
        ROBOT_X_MIN
        + nx * (
            ROBOT_X_MAX
            - ROBOT_X_MIN
        )
    )

    robot_y = (
        ROBOT_Y_MIN
        + ny * (
            ROBOT_Y_MAX
            - ROBOT_Y_MIN
        )
    )

    # LiDAR depth gets SMALLER when hand moves upward.
    # Therefore invert Z mapping.

    robot_z = (
        ROBOT_Z_MAX
        - nz * (
            ROBOT_Z_MAX
            - ROBOT_Z_MIN
        )
    )


    # --------------------------------------------------------
    # VELOCITY FEATURES
    # --------------------------------------------------------

    hand_x = df[HAND_X].to_numpy()
    hand_y = df[HAND_Y].to_numpy()
    depth = df[DEPTH].to_numpy()

    vx = np.gradient(hand_x)
    vy = np.gradient(hand_y)
    vz = np.gradient(depth)


    # --------------------------------------------------------
    # CREATE ML DATA
    # --------------------------------------------------------

    output = pd.DataFrame({

        "episode":
            episode_number,

        "split":
            (
                "train"
                if episode_number in TRAIN_EPISODES
                else "test"
            ),

        "episode_time_s":
            df["episode_time_s"],

        # RGB-D observations
        "hand_x":
            hand_x,

        "hand_y":
            hand_y,

        "lidar_z":
            depth,

        "grasp":
            df[GRASP].to_numpy(),

        # Motion context
        "hand_vx":
            vx,

        "hand_vy":
            vy,

        "hand_vz":
            vz,

        # Capacitive observation
        "cap_smooth":
            df[CAP].to_numpy(),

        # Manual semantic ground truth
        "object_held":
            df["object_held"].astype(int),

        "phase":
            df["phase"].astype(int),

        # Behaviour-cloning targets
        "target_robot_x":
            robot_x,

        "target_robot_y":
            robot_y,

        "target_robot_z":
            robot_z,
    })


    # --------------------------------------------------------
    # TARGET ROBOT DELTAS
    # --------------------------------------------------------

    output["target_dx"] = np.gradient(
        robot_x
    )

    output["target_dy"] = np.gradient(
        robot_y
    )

    output["target_dz"] = np.gradient(
        robot_z
    )


    processed.append(output)


# ============================================================
# COMBINE
# ============================================================

dataset = pd.concat(
    processed,
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

full_path = os.path.join(
    OUTPUT_DIR,
    "dataset.csv"
)

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)


dataset.to_csv(
    full_path,
    index=False
)

dataset[
    dataset["split"] == "train"
].to_csv(
    train_path,
    index=False
)

dataset[
    dataset["split"] == "test"
].to_csv(
    test_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

train = dataset[
    dataset["split"] == "train"
]

test = dataset[
    dataset["split"] == "test"
]


print("====================================")
print("ML DATASET READY")
print("====================================")

print(
    f"Total frames: {len(dataset)}"
)

print(
    f"Training frames: {len(train)}"
)

print(
    f"Test frames: {len(test)}"
)

print()

print(
    "Train episodes:",
    sorted(
        train["episode"].unique()
    )
)

print(
    "Test episodes:",
    sorted(
        test["episode"].unique()
    )
)

print()

print("Robot target ranges:")

print(
    "X:",
    dataset["target_robot_x"].min(),
    "to",
    dataset["target_robot_x"].max()
)

print(
    "Y:",
    dataset["target_robot_y"].min(),
    "to",
    dataset["target_robot_y"].max()
)

print(
    "Z:",
    dataset["target_robot_z"].min(),
    "to",
    dataset["target_robot_z"].max()
)

print()

print("Saved:")
print(full_path)
print(train_path)
print(test_path)