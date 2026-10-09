import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DEMO_DIR = r"C:\Users\kewin\humanoid_challenge\data\demo_03"

HAND_FILE = os.path.join(
    DEMO_DIR,
    "hand_trajectory.csv"
)

CAP_FILE = os.path.join(
    DEMO_DIR,
    "capacitance.csv"
)

OUTPUT_FILE = os.path.join(
    DEMO_DIR,
    "synchronized_demo.csv"
)

OUTPUT_PLOT = os.path.join(
    DEMO_DIR,
    "synchronized_demo.png"
)


# ============================================================
# SYNCHRONIZATION MARKER
# ============================================================

# DaVinci Resolve:
# first deliberate touch = 00:00:02:58
# video is 60 FPS

FPS = 60.0

VIDEO_SYNC_TIME = 2.0 + 58.0 / FPS

# First major deliberate capacitive interaction.
# Determined from the beginning of the recorded capacitance data.

CAP_SYNC_TIME = 4.432

# Therefore:
#
# cap_time = video_time + TIME_OFFSET

TIME_OFFSET = CAP_SYNC_TIME - VIDEO_SYNC_TIME


print()
print("====================================")
print("DEMO 03 SYNCHRONIZATION")
print("====================================")

print(f"Video sync time: {VIDEO_SYNC_TIME:.4f} s")
print(f"Cap sync time:   {CAP_SYNC_TIME:.4f} s")
print(f"Time offset:     {TIME_OFFSET:.4f} s")


# ============================================================
# LOAD DATA
# ============================================================

hand = pd.read_csv(HAND_FILE)
cap = pd.read_csv(CAP_FILE)

print()
print(f"Video/hand rows: {len(hand)}")
print(f"Cap samples:     {len(cap)}")


# ============================================================
# CLEAN CAPACITANCE
# ============================================================

cap = cap.sort_values(
    "pc_time_s"
).reset_index(drop=True)

# Median filtering removes individual noisy spikes while
# preserving sustained interaction events.

cap["cap_smooth"] = (
    cap["filtered"]
    .rolling(
        window=9,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# MAP VIDEO TIME -> CAPACITANCE TIME
# ============================================================

hand["cap_time_s"] = (
    hand["video_time_s"]
    + TIME_OFFSET
)


# ============================================================
# INTERPOLATE CAPACITANCE ONTO 60-HZ VIDEO TIMELINE
# ============================================================

cap_times = cap["pc_time_s"].to_numpy()

cap_raw = cap["filtered"].to_numpy(
    dtype=float
)

cap_smooth = cap["cap_smooth"].to_numpy(
    dtype=float
)

query_times = hand["cap_time_s"].to_numpy()


hand["cap_filtered"] = np.interp(
    query_times,
    cap_times,
    cap_raw,
    left=np.nan,
    right=np.nan
)

hand["cap_smooth"] = np.interp(
    query_times,
    cap_times,
    cap_smooth,
    left=np.nan,
    right=np.nan
)


# ============================================================
# INTERACTION FEATURE
# ============================================================

# Lower MPR121 filtered values correspond to stronger
# interaction / near-contact in this recording.
#
# This is an empirical interaction cue, NOT a calibrated
# distance or capacitance measurement.

CAP_THRESHOLD = 6.0

hand["interaction"] = (
    hand["cap_smooth"] <= CAP_THRESHOLD
).astype(int)


# ============================================================
# SAVE
# ============================================================

hand.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

valid_cap = hand["cap_smooth"].notna()

interaction_frames = int(
    hand["interaction"].sum()
)

print()
print("====================================")
print("RESULT")
print("====================================")

print(
    f"Valid synchronized cap frames: "
    f"{valid_cap.sum()} / {len(hand)}"
)

print(
    f"Interaction frames: "
    f"{interaction_frames}"
)

print(
    f"Interaction duration: "
    f"{interaction_frames / FPS:.2f} s"
)

print()
print("Saved:")
print(OUTPUT_FILE)


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 10),
    sharex=True
)


# ------------------------------------------------------------
# HAND XY
# ------------------------------------------------------------

axes[0].plot(
    hand["video_time_s"],
    hand["hand_x"],
    label="hand_x"
)

axes[0].plot(
    hand["video_time_s"],
    hand["hand_y"],
    label="hand_y"
)

axes[0].set_ylabel("Image position")
axes[0].legend()
axes[0].grid(alpha=0.3)


# ------------------------------------------------------------
# LIDAR DEPTH
# ------------------------------------------------------------

axes[1].plot(
    hand["video_time_s"],
    hand["lidar_z"]
)

axes[1].set_ylabel("LiDAR depth (m)")
axes[1].grid(alpha=0.3)


# ------------------------------------------------------------
# THUMB-INDEX / GRASP FEATURE
# ------------------------------------------------------------

axes[2].plot(
    hand["video_time_s"],
    hand["grasp"]
)

axes[2].set_ylabel("Thumb-index")
axes[2].grid(alpha=0.3)


# ------------------------------------------------------------
# CAPACITIVE SIGNAL
# ------------------------------------------------------------

axes[3].plot(
    hand["video_time_s"],
    hand["cap_smooth"],
    label="Capacitive signal"
)

axes[3].axhline(
    CAP_THRESHOLD,
    linestyle="--",
    label="Interaction threshold"
)

# Shade interaction regions.

axes[3].fill_between(
    hand["video_time_s"],
    0,
    1,
    where=hand["interaction"].astype(bool),
    alpha=0.2,
    transform=axes[3].get_xaxis_transform(),
    label="Detected interaction"
)

axes[3].set_ylabel("MPR121 filtered")
axes[3].set_xlabel("Video time (s)")
axes[3].legend()
axes[3].grid(alpha=0.3)


fig.suptitle(
    "Demo 03 — Synchronized RGB-D + Capacitive Data"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=160
)

print()
print("Saved diagnostic plot:")
print(OUTPUT_PLOT)

plt.show()