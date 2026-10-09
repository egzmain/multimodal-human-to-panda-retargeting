import os
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\kewin\humanoid_challenge"

PANDA_DIR = (
    r"C:\Users\kewin\mujoco_menagerie"
    r"\franka_emika_panda"
)

MODEL_PATH = os.path.join(
    PANDA_DIR,
    "scene.xml"
)

TEMP_MODEL_PATH = os.path.join(
    PANDA_DIR,
    "scene_learned_cube.xml"
)

PREDICTIONS_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "demo_03",
    "ml_dataset",
    "test_predictions_xyz_gripper.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EPISODE = 19

PLAYBACK_FPS = 60.0

XYZ_SMOOTH_WINDOW = 9
GRIPPER_SMOOTH_WINDOW = 7

GRIPPER_CLOSE_THRESHOLD = 0.60
GRIPPER_OPEN_THRESHOLD = 0.40

LOOP_PAUSE_SECONDS = 1.5


# ============================================================
# PANDA SETTINGS
# ============================================================

EE_BODY_NAME = "hand"

REFERENCE_Q = np.array([
    0.0,
    -0.65,
    0.0,
    -2.10,
    0.0,
    1.55,
    0.75,
])

DAMPING = 0.08
MAX_DQ = 0.035
POSTURE_GAIN = 0.12

IK_STEPS_PER_FRAME = 8

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


# ============================================================
# CUBE SETTINGS
# ============================================================

CUBE_HALF_SIZE = 0.018

# Position cube slightly below the finger geometry at the
# beginning of each replay.
CUBE_VERTICAL_OFFSET = np.array([
    0.0,
    0.0,
    -0.055,
])


# The cube is released only after the learned gripper has
# remained continuously OPEN for this many frames.
#
# 24 frames at 60 FPS = 0.4 seconds.
#
# This affects only the visual object attachment.
# The Panda fingers still execute every learned prediction.
CUBE_RELEASE_FRAMES = 24


# ============================================================
# CREATE SCENE WITH RED CUBE
# ============================================================

with open(
    MODEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    xml = f.read()


object_xml = """

<body
    name="demo_object"
    mocap="true"
    pos="0.4 0.0 0.3">

    <geom
        name="demo_object_geom"
        type="box"
        size="0.018 0.018 0.018"
        rgba="0.9 0.2 0.1 1"
        contype="0"
        conaffinity="0"
    />

</body>

"""


if "</worldbody>" not in xml:

    raise RuntimeError(
        "Could not find </worldbody> "
        "in Panda scene.xml"
    )


xml = xml.replace(
    "</worldbody>",
    object_xml
    + "\n</worldbody>"
)


with open(
    TEMP_MODEL_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(xml)


print()
print(
    "Created temporary cube scene:"
)

print(
    TEMP_MODEL_PATH
)


# ============================================================
# LOAD LEARNED POLICY OUTPUT
# ============================================================

df = pd.read_csv(
    PREDICTIONS_FILE
)


df = (
    df[df["episode"] == EPISODE]
    .copy()
    .reset_index(drop=True)
)


if len(df) == 0:

    raise RuntimeError(
        f"No predictions found for "
        f"Episode {EPISODE}"
    )


required_columns = [
    "pred_robot_x",
    "pred_robot_y",
    "pred_robot_z",
    "pred_gripper_probability",
]


for column in required_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Missing required column: "
            f"{column}"
        )


# ============================================================
# RAW LEARNED XYZ
# ============================================================

raw_targets = df[
    [
        "pred_robot_x",
        "pred_robot_y",
        "pred_robot_z",
    ]
].to_numpy(
    dtype=float
)


# ============================================================
# SMOOTH XYZ FOR EXECUTION
# ============================================================

targets = np.zeros_like(
    raw_targets
)


for axis in range(3):

    targets[:, axis] = (
        pd.Series(
            raw_targets[:, axis]
        )
        .rolling(
            window=XYZ_SMOOTH_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )


# ============================================================
# LEARNED GRIPPER PROBABILITY
# ============================================================

raw_gripper_probability = df[
    "pred_gripper_probability"
].to_numpy(
    dtype=float
)


gripper_probability = (
    pd.Series(
        raw_gripper_probability
    )
    .rolling(
        window=GRIPPER_SMOOTH_WINDOW,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    TEMP_MODEL_PATH
)

data = mujoco.MjData(
    model
)


# ============================================================
# BODY IDS
# ============================================================

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    EE_BODY_NAME
)


left_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_finger"
)


right_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "right_finger"
)


object_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "demo_object"
)


if hand_id < 0:

    raise RuntimeError(
        "Could not find Panda hand."
    )


if left_id < 0:

    raise RuntimeError(
        "Could not find left finger."
    )


if right_id < 0:

    raise RuntimeError(
        "Could not find right finger."
    )


if object_id < 0:

    raise RuntimeError(
        "Could not find demo cube."
    )


# ============================================================
# CUBE MOCAP
# ============================================================

mocap_id = model.body_mocapid[
    object_id
]


if mocap_id < 0:

    raise RuntimeError(
        "Cube does not have a mocap ID."
    )


def set_cube_position(
    position
):

    data.mocap_pos[
        mocap_id
    ] = position


# ============================================================
# INITIAL ROBOT STATE
# ============================================================

data.qpos[:7] = (
    REFERENCE_Q.copy()
)


for i in range(7):

    data.ctrl[i] = (
        REFERENCE_Q[i]
    )


data.ctrl[7] = (
    GRIPPER_OPEN
)


mujoco.mj_forward(
    model,
    data
)


# ============================================================
# GRIPPER CONTROL
# ============================================================

def set_gripper(
    closed
):

    if closed:

        data.ctrl[7] = (
            GRIPPER_CLOSED
        )

    else:

        data.ctrl[7] = (
            GRIPPER_OPEN
        )


# ============================================================
# HOLD ARM
# ============================================================

def hold_arm():

    for i in range(7):

        data.ctrl[i] = (
            data.qpos[i]
        )


# ============================================================
# ADVANCE GRIPPER PHYSICS
# ============================================================

def step_gripper_physics():

    # Preserve the IK-controlled arm configuration.
    arm_qpos = (
        data.qpos[:7].copy()
    )


    # Hold the seven arm actuators at the IK configuration.
    for i in range(7):

        data.ctrl[i] = (
            arm_qpos[i]
        )


    # Allow MuJoCo to advance the gripper actuator.
    mujoco.mj_step(
        model,
        data
    )


    # Restore exact IK arm pose.
    # Finger joints are intentionally NOT restored.
    data.qpos[:7] = (
        arm_qpos
    )


    # Prevent arm velocity accumulation.
    data.qvel[:7] = 0.0


    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# POSITION-ONLY IK
# ============================================================

def move_toward_target(
    target_position
):

    for _ in range(
        IK_STEPS_PER_FRAME
    ):

        mujoco.mj_forward(
            model,
            data
        )


        # ----------------------------------------------------
        # CURRENT END-EFFECTOR POSITION
        # ----------------------------------------------------

        current_position = (
            data.xpos[
                hand_id
            ].copy()
        )


        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        error = (
            target_position
            - current_position
        )


        # ----------------------------------------------------
        # JACOBIAN
        # ----------------------------------------------------

        jacobian_position = np.zeros(
            (3, model.nv)
        )


        jacobian_rotation = np.zeros(
            (3, model.nv)
        )


        mujoco.mj_jacBody(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            hand_id
        )


        J = (
            jacobian_position[
                :,
                :7
            ]
        )


        # ----------------------------------------------------
        # DAMPED LEAST-SQUARES IK
        # ----------------------------------------------------

        A = (
            J @ J.T
            + (DAMPING ** 2)
            * np.eye(3)
        )


        dq_task = (
            J.T
            @ np.linalg.solve(
                A,
                error
            )
        )


        # ----------------------------------------------------
        # NULL-SPACE POSTURE CONTROL
        # ----------------------------------------------------

        J_pinv = (
            J.T
            @ np.linalg.solve(
                A,
                np.eye(3)
            )
        )


        nullspace = (
            np.eye(7)
            - J_pinv @ J
        )


        posture_error = (
            REFERENCE_Q
            - data.qpos[:7]
        )


        dq_posture = (
            POSTURE_GAIN
            * nullspace
            @ posture_error
        )


        # ----------------------------------------------------
        # TOTAL JOINT UPDATE
        # ----------------------------------------------------

        dq = (
            dq_task
            + dq_posture
        )


        dq = np.clip(
            dq,
            -MAX_DQ,
            MAX_DQ
        )


        data.qpos[:7] += dq


        # ----------------------------------------------------
        # JOINT LIMITS
        # ----------------------------------------------------

        for joint_index in range(7):

            lower = (
                model.jnt_range[
                    joint_index,
                    0
                ]
            )

            upper = (
                model.jnt_range[
                    joint_index,
                    1
                ]
            )


            data.qpos[
                joint_index
            ] = np.clip(
                data.qpos[
                    joint_index
                ],
                lower,
                upper
            )


        hold_arm()


        mujoco.mj_forward(
            model,
            data
        )


# ============================================================
# FIND FINGER GEOMETRY CENTRE
# ============================================================

def get_finger_center():

    mujoco.mj_forward(
        model,
        data
    )


    left_positions = []
    right_positions = []


    for geom_id in range(
        model.ngeom
    ):

        body_id = (
            model.geom_bodyid[
                geom_id
            ]
        )


        if body_id == left_id:

            left_positions.append(
                data.geom_xpos[
                    geom_id
                ].copy()
            )


        elif body_id == right_id:

            right_positions.append(
                data.geom_xpos[
                    geom_id
                ].copy()
            )


    # --------------------------------------------------------
    # USE GEOMETRY CENTRES WHEN AVAILABLE
    # --------------------------------------------------------

    if (
        len(left_positions) > 0
        and
        len(right_positions) > 0
    ):

        left_center = np.mean(
            left_positions,
            axis=0
        )


        right_center = np.mean(
            right_positions,
            axis=0
        )


        return (
            left_center
            + right_center
        ) / 2.0


    # --------------------------------------------------------
    # FALLBACK: BODY CENTRES
    # --------------------------------------------------------

    left = (
        data.xpos[
            left_id
        ].copy()
    )


    right = (
        data.xpos[
            right_id
        ].copy()
    )


    return (
        left + right
    ) / 2.0


# ============================================================
# INITIAL CUBE POSITION
# ============================================================

def get_initial_cube_position():

    finger_center = (
        get_finger_center()
    )


    return (
        finger_center
        + CUBE_VERTICAL_OFFSET
    )


# ============================================================
# INFORMATION
# ============================================================

print()
print(
    "===================================="
)

print(
    "LEARNED XYZ + GRIPPER + CUBE"
)

print(
    "===================================="
)

print(
    f"Held-out Episode: {EPISODE}"
)

print(
    f"Frames: {len(df)}"
)


if "position_error_m" in df.columns:

    print(
        f"Raw mean XYZ error: "
        f"{df['position_error_m'].mean() * 100:.2f} cm"
    )


print()

print(
    "XYZ source: LEARNED"
)

print(
    "Gripper source: LEARNED"
)

print(
    "Manual object_held labels are NOT "
    "used for robot control."
)

print()

print(
    "Cube visualization:"
)

print(
    "event-conditioned mocap attachment"
)

print(
    "Physical cube contact: DISABLED"
)

print()

print(
    "Cube release debounce:"
)

print(
    f"{CUBE_RELEASE_FRAMES} "
    f"continuous open frames"
)

print()

print(
    "Raw gripper probability range:"
)

print(
    f"{raw_gripper_probability.min():.3f}"
    f" -> "
    f"{raw_gripper_probability.max():.3f}"
)

print()

print(
    "Smoothed gripper probability range:"
)

print(
    f"{gripper_probability.min():.3f}"
    f" -> "
    f"{gripper_probability.max():.3f}"
)

print()

print(
    "Close threshold:",
    GRIPPER_CLOSE_THRESHOLD
)

print(
    "Open threshold:",
    GRIPPER_OPEN_THRESHOLD
)

print()


# ============================================================
# MOVE PANDA TO START
# ============================================================

print(
    "Moving Panda to starting position..."
)


set_gripper(
    False
)


for _ in range(150):

    move_toward_target(
        targets[0]
    )

    step_gripper_physics()


# ============================================================
# INITIAL CUBE POSITION
# ============================================================

initial_cube_position = (
    get_initial_cube_position()
)


set_cube_position(
    initial_cube_position
)


mujoco.mj_forward(
    model,
    data
)


print(
    "Cube start position:"
)

print(
    initial_cube_position
)

print()


# ============================================================
# VIEWER
# ============================================================

frame_duration = (
    1.0
    / PLAYBACK_FPS
)


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    time.sleep(
        1.0
    )


    loop_number = 0


    # ========================================================
    # REPLAY LOOP
    # ========================================================

    while viewer.is_running():

        loop_number += 1


        print()
        print(
            "===================================="
        )

        print(
            f"LOOP {loop_number}"
        )

        print(
            "===================================="
        )


        # ====================================================
        # RESET EPISODE STATE
        # ====================================================

        gripper_closed = False

        object_grasped = False

        object_released = False

        cube_open_counter = 0


        set_gripper(
            False
        )


        # ====================================================
        # RETURN ROBOT TO START
        # ====================================================

        print(
            "Returning to start..."
        )


        for _ in range(100):

            if not viewer.is_running():
                break


            move_toward_target(
                targets[0]
            )


            set_gripper(
                False
            )


            step_gripper_physics()


            viewer.sync()


            time.sleep(
                0.005
            )


        if not viewer.is_running():
            break


        # ====================================================
        # RESET CUBE
        # ====================================================

        initial_cube_position = (
            get_initial_cube_position()
        )


        set_cube_position(
            initial_cube_position
        )


        mujoco.mj_forward(
            model,
            data
        )


        # ====================================================
        # MAKE SURE FINGERS BEGIN OPEN
        # ====================================================

        for _ in range(80):

            if not viewer.is_running():
                break


            set_gripper(
                False
            )


            step_gripper_physics()


            viewer.sync()


            time.sleep(
                model.opt.timestep
            )


        if not viewer.is_running():
            break


        print(
            "START: GRIPPER OPEN"
        )

        print(
            "Cube ready."
        )


        # ====================================================
        # RUN LEARNED POLICY
        # ====================================================

        for frame_index in range(
            len(targets)
        ):

            if not viewer.is_running():
                break


            frame_start = (
                time.perf_counter()
            )


            # =================================================
            # LEARNED GRIPPER PROBABILITY
            # =================================================

            probability = (
                gripper_probability[
                    frame_index
                ]
            )


            new_state = (
                gripper_closed
            )


            # -------------------------------------------------
            # LEARNED CLOSE
            # -------------------------------------------------

            if (
                not gripper_closed
                and
                probability
                >= GRIPPER_CLOSE_THRESHOLD
            ):

                new_state = True


            # -------------------------------------------------
            # LEARNED OPEN
            # -------------------------------------------------

            elif (
                gripper_closed
                and
                probability
                <= GRIPPER_OPEN_THRESHOLD
            ):

                new_state = False


            # =================================================
            # LEARNED STATE TRANSITION
            # =================================================

            if (
                new_state
                != gripper_closed
            ):

                gripper_closed = (
                    new_state
                )


                # ---------------------------------------------
                # CLOSE
                # ---------------------------------------------

                if gripper_closed:

                    print(
                        f"Frame "
                        f"{frame_index:03d}: "
                        f"LEARNED CLOSE "
                        f"(p={probability:.3f})"
                    )


                    # First learned close picks up the cube.
                    if (
                        not object_grasped
                        and
                        not object_released
                    ):

                        object_grasped = True

                        cube_open_counter = 0


                        print(
                            ">>> CUBE PICKED UP"
                        )


                # ---------------------------------------------
                # OPEN
                # ---------------------------------------------

                else:

                    print(
                        f"Frame "
                        f"{frame_index:03d}: "
                        f"LEARNED OPEN "
                        f"(p={probability:.3f})"
                    )


            # =================================================
            # CUBE RELEASE DEBOUNCE
            #
            # IMPORTANT:
            # This does NOT modify the robot's learned
            # gripper command.
            #
            # It only determines when the visual cube
            # attachment is released.
            # =================================================

            if object_grasped:

                # ---------------------------------------------
                # GRIPPER CURRENTLY CLOSED
                # ---------------------------------------------

                if gripper_closed:

                    # Any close prediction resets the visual
                    # release counter.
                    cube_open_counter = 0


                # ---------------------------------------------
                # GRIPPER CURRENTLY OPEN
                # ---------------------------------------------

                else:

                    cube_open_counter += 1


                    # -----------------------------------------
                    # SUSTAINED OPEN -> RELEASE CUBE
                    # -----------------------------------------

                    if (
                        cube_open_counter
                        >= CUBE_RELEASE_FRAMES
                    ):

                        object_grasped = False

                        object_released = True


                        print(
                            f">>> CUBE RELEASED "
                            f"after "
                            f"{CUBE_RELEASE_FRAMES} "
                            f"continuous open frames"
                        )


            # =================================================
            # APPLY ACTUAL LEARNED GRIPPER ACTION
            # =================================================

            set_gripper(
                gripper_closed
            )


            # =================================================
            # APPLY LEARNED XYZ ACTION
            # =================================================

            move_toward_target(
                targets[
                    frame_index
                ]
            )


            # =================================================
            # ADVANCE PHYSICS FOR PANDA FINGERS
            # =================================================

            for _ in range(4):

                set_gripper(
                    gripper_closed
                )


                step_gripper_physics()


            # =================================================
            # VISUAL CUBE ATTACHMENT
            # =================================================

            if object_grasped:

                cube_position = (
                    get_finger_center()
                )


                set_cube_position(
                    cube_position
                )


                mujoco.mj_forward(
                    model,
                    data
                )


            # If the cube has been released, nothing is done.
            # Because it is a mocap body, it remains at its
            # final release position.


            # =================================================
            # VIEWER
            # =================================================

            viewer.sync()


            # =================================================
            # 60 FPS PLAYBACK TIMING
            # =================================================

            elapsed = (
                time.perf_counter()
                - frame_start
            )


            remaining = (
                frame_duration
                - elapsed
            )


            if remaining > 0:

                time.sleep(
                    remaining
                )


        if not viewer.is_running():
            break


        # ====================================================
        # EPISODE COMPLETE
        # ====================================================

        print(
            "Episode complete."
        )


        if object_released:

            print(
                "Cube remained at learned "
                "release location."
            )

        elif object_grasped:

            print(
                "Cube still attached at "
                "end of episode."
            )


        # ====================================================
        # PAUSE BEFORE REPLAY
        # ====================================================

        pause_end = (
            time.time()
            + LOOP_PAUSE_SECONDS
        )


        while (
            viewer.is_running()
            and
            time.time()
            < pause_end
        ):

            set_gripper(
                gripper_closed
            )


            step_gripper_physics()


            # Keep cube attached during pause if the episode
            # ended before a sustained release.
            if object_grasped:

                cube_position = (
                    get_finger_center()
                )


                set_cube_position(
                    cube_position
                )


                mujoco.mj_forward(
                    model,
                    data
                )


            viewer.sync()


            time.sleep(
                model.opt.timestep
            )


print()
print(
    "Viewer closed."
)