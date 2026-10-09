import mujoco
import mujoco.viewer
import pandas as pd
import numpy as np
import time


# ============================================================
# FILES
# ============================================================

MODEL_PATH = (
    r"C:\Users\kewin\mujoco_menagerie"
    r"\franka_emika_panda\scene.xml"
)

TRAJECTORY = (
    r"C:\Users\kewin\humanoid_challenge"
    r"\data\demo_01\action_01_robot.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 60.0

GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0

# From recorded human demonstration
PICKUP_TIME = 1.03
PLACE_TIME = 2.77

# IK settings
DAMPING = 0.08
MAX_DQ = 0.035

# How strongly we prefer a natural Panda posture
POSTURE_GAIN = 0.12


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

traj = pd.read_csv(TRAJECTORY)

targets = traj[
    ["robot_x", "robot_y", "robot_z"]
].to_numpy()


# ============================================================
# PANDA HAND
# ============================================================

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

if hand_id == -1:
    raise RuntimeError(
        "Could not find Panda hand body."
    )


# ============================================================
# SAFE REFERENCE POSTURE
#
# A conventional bent-elbow Panda configuration.
# This gives IK a posture to prefer instead of letting
# the arm fold into arbitrary configurations.
# ============================================================

REFERENCE_Q = np.array([
    0.0,
    -0.65,
    0.0,
    -2.10,
    0.0,
    1.55,
    0.75
])


# ============================================================
# INITIALIZE ROBOT NEAR REFERENCE POSTURE
# ============================================================

data.qpos[:7] = REFERENCE_Q

data.ctrl[:7] = REFERENCE_Q
data.ctrl[7] = GRIPPER_OPEN

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# POSITION IK + POSTURE REGULARIZATION
#
# IMPORTANT:
# We deliberately DO NOT impose a hard orientation.
#
# Instead:
#   1. XYZ follows human data
#   2. null-space term keeps Panda near natural posture
#
# This should be dramatically more stable.
# ============================================================

def move_toward(target_position, iterations=5):

    for _ in range(iterations):

        mujoco.mj_forward(
            model,
            data
        )

        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        current_position = (
            data.xpos[hand_id].copy()
        )

        position_error = (
            target_position
            - current_position
        )


        # ----------------------------------------------------
        # POSITION JACOBIAN
        # ----------------------------------------------------

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            hand_id
        )

        J = jacp[:, :7]


        # ----------------------------------------------------
        # DAMPED PSEUDOINVERSE
        # ----------------------------------------------------

        JJ = (
            J @ J.T
            + (DAMPING ** 2)
            * np.eye(3)
        )

        J_pinv = (
            J.T
            @ np.linalg.inv(JJ)
        )


        # ----------------------------------------------------
        # PRIMARY TASK:
        # FOLLOW HUMAN XYZ
        # ----------------------------------------------------

        dq_position = (
            J_pinv
            @ position_error
        )


        # ----------------------------------------------------
        # SECONDARY TASK:
        # STAY NEAR NATURAL PANDA POSTURE
        # ----------------------------------------------------

        q_current = (
            data.qpos[:7].copy()
        )

        posture_error = (
            REFERENCE_Q
            - q_current
        )


        # Null-space projector
        null_space = (
            np.eye(7)
            - J_pinv @ J
        )

        dq_posture = (
            POSTURE_GAIN
            * null_space
            @ posture_error
        )


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        dq = (
            dq_position
            + dq_posture
        )

        dq = np.clip(
            dq,
            -MAX_DQ,
            MAX_DQ
        )

        desired_q = (
            q_current + dq
        )


        # ----------------------------------------------------
        # RESPECT JOINT LIMITS
        # ----------------------------------------------------

        for j in range(7):

            low = (
                model.actuator_ctrlrange[j, 0]
            )

            high = (
                model.actuator_ctrlrange[j, 1]
            )

            desired_q[j] = np.clip(
                desired_q[j],
                low,
                high
            )


        # ----------------------------------------------------
        # COMMAND ARM
        # ----------------------------------------------------

        data.ctrl[:7] = desired_q


        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        for _ in range(5):

            mujoco.mj_step(
                model,
                data
            )


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print()
    print(
        "===================================="
    )
    print(
        "STABLE HUMAN -> PANDA REPLAY"
    )
    print(
        "===================================="
    )
    print()

    print(
        "Moving Panda to first human target..."
    )


    # ========================================================
    # WARM UP TO FIRST TARGET
    # ========================================================

    data.ctrl[7] = GRIPPER_OPEN

    for _ in range(300):

        if not viewer.is_running():
            break

        data.ctrl[7] = GRIPPER_OPEN

        move_toward(
            targets[0],
            iterations=1
        )

        viewer.sync()

        time.sleep(0.004)


    replay_number = 1


    # ========================================================
    # REPLAY LOOP
    # ========================================================

    while viewer.is_running():

        print()
        print(
            f"========== REPLAY {replay_number} =========="
        )

        errors = []

        data.ctrl[7] = GRIPPER_OPEN


        # ====================================================
        # HUMAN TRAJECTORY
        # ====================================================

        for i, target in enumerate(targets):

            if not viewer.is_running():
                break

            action_time = (
                i / FPS
            )


            # ------------------------------------------------
            # GRIPPER
            # ------------------------------------------------

            if action_time < PICKUP_TIME:

                data.ctrl[7] = (
                    GRIPPER_OPEN
                )

                gripper_state = "OPEN"

            elif action_time < PLACE_TIME:

                data.ctrl[7] = (
                    GRIPPER_CLOSED
                )

                gripper_state = "CLOSED"

            else:

                data.ctrl[7] = (
                    GRIPPER_OPEN
                )

                gripper_state = "OPEN"


            # ------------------------------------------------
            # FOLLOW HUMAN XYZ
            # ------------------------------------------------

            move_toward(
                target,
                iterations=5
            )

            mujoco.mj_forward(
                model,
                data
            )


            # ------------------------------------------------
            # TRACKING ERROR
            # ------------------------------------------------

            actual = (
                data.xpos[hand_id].copy()
            )

            error = np.linalg.norm(
                target - actual
            )

            errors.append(error)


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if i % 20 == 0:

                contact = int(
                    traj.iloc[i][
                        "interaction"
                    ]
                )

                print(
                    f"Frame {i:03d} | "
                    f"time={action_time:.2f}s | "
                    f"error={error:.3f}m | "
                    f"contact={contact} | "
                    f"gripper={gripper_state}"
                )


            viewer.sync()

            time.sleep(
                1.0 / FPS
            )


        # ====================================================
        # FINISHED TRAJECTORY
        # ====================================================

        if not viewer.is_running():
            break

        if errors:

            print()

            print(
                f"Mean tracking error: "
                f"{np.mean(errors):.4f} m"
            )

            print(
                f"Max tracking error: "
                f"{np.max(errors):.4f} m"
            )


        print()
        print(
            "Replay complete."
        )

        time.sleep(2)


        # ====================================================
        # RETURN TO START
        # ====================================================

        if not viewer.is_running():
            break

        print(
            "Returning to start..."
        )

        data.ctrl[7] = (
            GRIPPER_OPEN
        )

        for _ in range(200):

            if not viewer.is_running():
                break

            data.ctrl[7] = (
                GRIPPER_OPEN
            )

            move_toward(
                targets[0],
                iterations=1
            )

            viewer.sync()

            time.sleep(0.004)


        time.sleep(1)

        replay_number += 1


print()
print(
    "MuJoCo viewer closed."
)