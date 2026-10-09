import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

DEMO_DIR = r"C:\Users\kewin\humanoid_challenge\data\demo_03"

INPUT_FILE = os.path.join(
    DEMO_DIR,
    "synchronized_demo.csv"
)

EPISODE_DIR = os.path.join(
    DEMO_DIR,
    "episodes"
)

SUMMARY_FILE = os.path.join(
    DEMO_DIR,
    "episodes_summary.csv"
)

os.makedirs(
    EPISODE_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 60.0


# ============================================================
# HELPER: DAVINCI SS:FF -> SECONDS
# ============================================================

def tcr(seconds, frames):
    return seconds + frames / FPS


# ============================================================
# MANUAL ANNOTATIONS
#
# first_contact:
#     object touched immediately before pickup
#
# second_contact:
#     object returned to surface
#
# First two synchronization touches are NOT included.
# ============================================================

annotations = [
    (tcr(6, 53),  tcr(8, 27)),
    (tcr(11, 14), tcr(12, 57)),
    (tcr(15, 44), tcr(18, 10)),
    (tcr(20, 8),  tcr(23, 3)),
    (tcr(24, 40), tcr(25, 35)),
    (tcr(26, 25), tcr(26, 49)),
    (tcr(27, 48), tcr(28, 12)),
    (tcr(29, 17), tcr(29, 46)),
    (tcr(30, 16), tcr(30, 46)),
    (tcr(31, 11), tcr(31, 39)),
    (tcr(32, 29), tcr(33, 29)),
    (tcr(35, 10), tcr(36, 33)),
    (tcr(37, 47), tcr(38, 31)),
    (tcr(39, 28), tcr(40, 35)),
    (tcr(41, 39), tcr(42, 35)),
    (tcr(44, 2),  tcr(44, 40)),
    (tcr(45, 37), tcr(46, 28)),
    (tcr(47, 16), tcr(47, 57)),
    (tcr(48, 45), tcr(50, 20)),
    (tcr(51, 8),  tcr(51, 55)),
]


# ============================================================
# LOAD SYNCHRONIZED DATA
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)

video_start = float(
    data["video_time_s"].min()
)

video_end = float(
    data["video_time_s"].max()
)


# ============================================================
# CALCULATE NON-OVERLAPPING EPISODE BOUNDARIES
# ============================================================

episode_bounds = []

for i, (first_contact, second_contact) in enumerate(
    annotations
):

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    if i == 0:

        # Give first episode some approach context,
        # but do not include synchronization markers.

        start = max(
            5.5,
            first_contact - 1.0
        )

    else:

        previous_second_contact = (
            annotations[i - 1][1]
        )

        # Midpoint between previous placement and
        # current pickup contact.

        start = (
            previous_second_contact
            + first_contact
        ) / 2.0


    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    if i == len(annotations) - 1:

        end = min(
            video_end,
            second_contact + 1.0
        )

    else:

        next_first_contact = (
            annotations[i + 1][0]
        )

        # Midpoint between current placement and
        # next pickup contact.

        end = (
            second_contact
            + next_first_contact
        ) / 2.0


    episode_bounds.append(
        (
            start,
            first_contact,
            second_contact,
            end
        )
    )


# ============================================================
# EXTRACT EPISODES
# ============================================================

summary = []

print()
print("====================================")
print("DEMO 03 EPISODE EXTRACTION")
print("====================================")
print()


for episode_index, values in enumerate(
    episode_bounds,
    start=1
):

    (
        start,
        first_contact,
        second_contact,
        end
    ) = values


    mask = (
        (data["video_time_s"] >= start)
        &
        (data["video_time_s"] <= end)
    )

    episode = (
        data.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


    if len(episode) == 0:
        print(
            f"WARNING: Episode {episode_index} "
            f"contains no frames."
        )
        continue


    # --------------------------------------------------------
    # EPISODE-RELATIVE TIME
    # --------------------------------------------------------

    episode["episode_time_s"] = (
        episode["video_time_s"]
        - episode["video_time_s"].iloc[0]
    )


    # --------------------------------------------------------
    # SEMANTIC PHASE LABEL
    #
    # 0 = approach
    # 1 = object manipulation / lift
    # 2 = after return / retreat
    # --------------------------------------------------------

    episode["phase"] = 0

    episode.loc[
        episode["video_time_s"]
        >= first_contact,
        "phase"
    ] = 1

    episode.loc[
        episode["video_time_s"]
        >= second_contact,
        "phase"
    ] = 2


    # --------------------------------------------------------
    # OBJECT HELD LABEL
    #
    # This is derived from YOUR manual video annotation.
    # It is not claimed to come directly from capacitance.
    # --------------------------------------------------------

    episode["object_held"] = (
        (
            episode["video_time_s"]
            >= first_contact
        )
        &
        (
            episode["video_time_s"]
            < second_contact
        )
    ).astype(int)


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    episode.insert(
        0,
        "episode",
        episode_index
    )

    episode.insert(
        1,
        "session",
        "demo_03"
    )


    # --------------------------------------------------------
    # SAVE EPISODE
    # --------------------------------------------------------

    output_file = os.path.join(
        EPISODE_DIR,
        f"episode_{episode_index:02d}.csv"
    )

    episode.to_csv(
        output_file,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    interaction_frames = int(
        episode["interaction"].sum()
    )

    held_frames = int(
        episode["object_held"].sum()
    )

    duration = (
        episode["episode_time_s"].iloc[-1]
    )


    summary.append(
        {
            "episode": episode_index,
            "start_s": start,
            "first_contact_s": first_contact,
            "second_contact_s": second_contact,
            "end_s": end,
            "duration_s": duration,
            "frames": len(episode),
            "interaction_frames": interaction_frames,
            "object_held_frames": held_frames,
        }
    )


    print(
        f"Episode {episode_index:02d} | "
        f"{start:6.2f} -> {end:6.2f} s | "
        f"FC={first_contact:6.2f} | "
        f"SC={second_contact:6.2f} | "
        f"{len(episode):3d} frames"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("====================================")
print("RESULT")
print("====================================")

print(
    f"Episodes extracted: {len(summary_df)}"
)

print(
    f"Total episode frames: "
    f"{summary_df['frames'].sum()}"
)

print(
    f"Mean episode duration: "
    f"{summary_df['duration_s'].mean():.2f} s"
)

print(
    f"Min episode duration: "
    f"{summary_df['duration_s'].min():.2f} s"
)

print(
    f"Max episode duration: "
    f"{summary_df['duration_s'].max():.2f} s"
)

print()
print("Episodes saved to:")
print(EPISODE_DIR)

print()
print("Summary saved to:")
print(SUMMARY_FILE)