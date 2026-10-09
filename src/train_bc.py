import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\kewin\humanoid_challenge"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "demo_03",
    "ml_dataset"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.csv"
)

MODEL_FILE = os.path.join(
    DATA_DIR,
    "bc_policy_xyz_gripper.pt"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "test_predictions_xyz_gripper.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 150
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Relative importance of gripper classification.
GRIPPER_LOSS_WEIGHT = 0.5


# ============================================================
# INPUT FEATURES
# ============================================================

FEATURES = [
    "hand_x",
    "hand_y",
    "lidar_z",
    "grasp",
    "hand_vx",
    "hand_vy",
    "hand_vz",
    "cap_smooth",
]


# ============================================================
# XYZ TARGETS
# ============================================================

XYZ_TARGETS = [
    "target_robot_x",
    "target_robot_y",
    "target_robot_z",
]


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


X_train = train_df[
    FEATURES
].to_numpy(
    dtype=np.float32
)

X_test = test_df[
    FEATURES
].to_numpy(
    dtype=np.float32
)


Y_xyz_train = train_df[
    XYZ_TARGETS
].to_numpy(
    dtype=np.float32
)

Y_xyz_test = test_df[
    XYZ_TARGETS
].to_numpy(
    dtype=np.float32
)


Y_gripper_train = train_df[
    "object_held"
].to_numpy(
    dtype=np.float32
).reshape(
    -1,
    1
)

Y_gripper_test = test_df[
    "object_held"
].to_numpy(
    dtype=np.float32
).reshape(
    -1,
    1
)


# ============================================================
# CHECK DATA
# ============================================================

arrays_to_check = [
    X_train,
    X_test,
    Y_xyz_train,
    Y_xyz_test,
    Y_gripper_train,
    Y_gripper_test,
]

for array in arrays_to_check:

    if not np.isfinite(array).all():

        raise RuntimeError(
            "Dataset contains NaN or Inf."
        )


# ============================================================
# NORMALIZATION
#
# Training data ONLY.
# ============================================================

x_mean = X_train.mean(
    axis=0
)

x_std = X_train.std(
    axis=0
)

x_std[
    x_std < 1e-8
] = 1.0


xyz_mean = Y_xyz_train.mean(
    axis=0
)

xyz_std = Y_xyz_train.std(
    axis=0
)

xyz_std[
    xyz_std < 1e-8
] = 1.0


X_train_n = (
    X_train - x_mean
) / x_std


X_test_n = (
    X_test - x_mean
) / x_std


Y_xyz_train_n = (
    Y_xyz_train - xyz_mean
) / xyz_std


# ============================================================
# TORCH DATASET
# ============================================================

train_x_tensor = torch.tensor(
    X_train_n,
    dtype=torch.float32
)

train_xyz_tensor = torch.tensor(
    Y_xyz_train_n,
    dtype=torch.float32
)

train_gripper_tensor = torch.tensor(
    Y_gripper_train,
    dtype=torch.float32
)


dataset = TensorDataset(
    train_x_tensor,
    train_xyz_tensor,
    train_gripper_tensor
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("====================================")
print("XYZ + GRIPPER BEHAVIOUR CLONING")
print("====================================")

print(
    "Device:",
    device
)

print(
    "Training frames:",
    len(train_df)
)

print(
    "Held-out test frames:",
    len(test_df)
)

print(
    "Training episodes:",
    sorted(
        train_df["episode"].unique()
    )
)

print(
    "Test episodes:",
    sorted(
        test_df["episode"].unique()
    )
)

print()

print(
    "Input features:",
    FEATURES
)

print()

print(
    "Training CLOSED frames:",
    int(
        Y_gripper_train.sum()
    )
)

print(
    "Training OPEN frames:",
    int(
        len(Y_gripper_train)
        - Y_gripper_train.sum()
    )
)

print()


# ============================================================
# MODEL
#
# Shared representation followed by:
#
# XYZ head     -> 3 continuous values
# Gripper head -> 1 binary logit
# ============================================================

class BCPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(

            nn.Linear(
                len(FEATURES),
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),
        )


        self.xyz_head = nn.Linear(
            32,
            3
        )


        self.gripper_head = nn.Linear(
            32,
            1
        )


    def forward(self, x):

        features = self.shared(
            x
        )

        xyz = self.xyz_head(
            features
        )

        gripper_logit = (
            self.gripper_head(
                features
            )
        )

        return (
            xyz,
            gripper_logit
        )


model = BCPolicy().to(
    device
)


# ============================================================
# LOSSES
# ============================================================

xyz_loss_function = (
    nn.MSELoss()
)

gripper_loss_function = (
    nn.BCEWithLogitsLoss()
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss_sum = 0.0
    xyz_loss_sum = 0.0
    gripper_loss_sum = 0.0


    for (
        batch_x,
        batch_xyz,
        batch_gripper
    ) in loader:

        batch_x = batch_x.to(
            device
        )

        batch_xyz = batch_xyz.to(
            device
        )

        batch_gripper = (
            batch_gripper.to(
                device
            )
        )


        optimizer.zero_grad()


        (
            predicted_xyz,
            predicted_gripper_logit
        ) = model(
            batch_x
        )


        xyz_loss = (
            xyz_loss_function(
                predicted_xyz,
                batch_xyz
            )
        )


        gripper_loss = (
            gripper_loss_function(
                predicted_gripper_logit,
                batch_gripper
            )
        )


        total_loss = (
            xyz_loss
            + GRIPPER_LOSS_WEIGHT
            * gripper_loss
        )


        total_loss.backward()

        optimizer.step()


        batch_size = len(
            batch_x
        )


        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        xyz_loss_sum += (
            xyz_loss.item()
            * batch_size
        )

        gripper_loss_sum += (
            gripper_loss.item()
            * batch_size
        )


    average_total_loss = (
        total_loss_sum
        / len(dataset)
    )

    average_xyz_loss = (
        xyz_loss_sum
        / len(dataset)
    )

    average_gripper_loss = (
        gripper_loss_sum
        / len(dataset)
    )


    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} "
            f"| total="
            f"{average_total_loss:.6f} "
            f"| xyz="
            f"{average_xyz_loss:.6f} "
            f"| grip="
            f"{average_gripper_loss:.6f}"
        )


# ============================================================
# HELD-OUT TEST
# ============================================================

model.eval()


with torch.no_grad():

    test_tensor = torch.tensor(
        X_test_n,
        dtype=torch.float32
    ).to(
        device
    )


    (
        predicted_xyz_normalized,
        predicted_gripper_logits
    ) = model(
        test_tensor
    )


    predicted_xyz_normalized = (
        predicted_xyz_normalized
        .cpu()
        .numpy()
    )


    predicted_gripper_logits = (
        predicted_gripper_logits
        .cpu()
        .numpy()
        .reshape(-1)
    )


# ============================================================
# DENORMALIZE XYZ
# ============================================================

predicted_xyz = (
    predicted_xyz_normalized
    * xyz_std
    + xyz_mean
)


# ============================================================
# GRIPPER PROBABILITY
# ============================================================

predicted_gripper_probability = (
    1.0
    / (
        1.0
        + np.exp(
            -predicted_gripper_logits
        )
    )
)


predicted_gripper = (
    predicted_gripper_probability
    >= 0.5
).astype(
    np.int32
)


true_gripper = (
    Y_gripper_test
    .reshape(-1)
    .astype(np.int32)
)


# ============================================================
# XYZ METRICS
# ============================================================

xyz_errors = (
    predicted_xyz
    - Y_xyz_test
)


rmse_xyz = np.sqrt(
    np.mean(
        xyz_errors ** 2,
        axis=0
    )
)


overall_coordinate_rmse = np.sqrt(
    np.mean(
        xyz_errors ** 2
    )
)


position_error = np.linalg.norm(
    xyz_errors,
    axis=1
)


# ============================================================
# GRIPPER METRICS
# ============================================================

true_positive = int(
    np.sum(
        (predicted_gripper == 1)
        &
        (true_gripper == 1)
    )
)

true_negative = int(
    np.sum(
        (predicted_gripper == 0)
        &
        (true_gripper == 0)
    )
)

false_positive = int(
    np.sum(
        (predicted_gripper == 1)
        &
        (true_gripper == 0)
    )
)

false_negative = int(
    np.sum(
        (predicted_gripper == 0)
        &
        (true_gripper == 1)
    )
)


accuracy = (
    true_positive
    + true_negative
) / len(
    true_gripper
)


precision = (
    true_positive
    / (
        true_positive
        + false_positive
    )
    if (
        true_positive
        + false_positive
    ) > 0
    else 0.0
)


recall = (
    true_positive
    / (
        true_positive
        + false_negative
    )
    if (
        true_positive
        + false_negative
    ) > 0
    else 0.0
)


f1 = (
    2.0
    * precision
    * recall
    / (
        precision
        + recall
    )
    if (
        precision
        + recall
    ) > 0
    else 0.0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("====================================")
print("HELD-OUT XYZ RESULTS")
print("====================================")

print(
    f"X RMSE: "
    f"{rmse_xyz[0] * 100:.2f} cm"
)

print(
    f"Y RMSE: "
    f"{rmse_xyz[1] * 100:.2f} cm"
)

print(
    f"Z RMSE: "
    f"{rmse_xyz[2] * 100:.2f} cm"
)

print(
    f"Overall coordinate RMSE: "
    f"{overall_coordinate_rmse * 100:.2f} cm"
)

print(
    f"Mean 3D position error: "
    f"{position_error.mean() * 100:.2f} cm"
)

print(
    f"Median 3D position error: "
    f"{np.median(position_error) * 100:.2f} cm"
)


print()
print("====================================")
print("HELD-OUT GRIPPER RESULTS")
print("====================================")

print(
    f"Accuracy: "
    f"{accuracy * 100:.1f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.1f}%"
)

print(
    f"Recall: "
    f"{recall * 100:.1f}%"
)

print(
    f"F1: "
    f"{f1 * 100:.1f}%"
)

print()

print(
    "TP:",
    true_positive
)

print(
    "TN:",
    true_negative
)

print(
    "FP:",
    false_positive
)

print(
    "FN:",
    false_negative
)


# ============================================================
# PER-EPISODE RESULTS
# ============================================================

print()
print("====================================")
print("PER-EPISODE RESULTS")
print("====================================")


test_episode_array = (
    test_df["episode"]
    .to_numpy()
)


for episode in sorted(
    test_df["episode"].unique()
):

    mask = (
        test_episode_array
        == episode
    )


    episode_position_error = (
        position_error[
            mask
        ]
    )


    episode_true_gripper = (
        true_gripper[
            mask
        ]
    )

    episode_pred_gripper = (
        predicted_gripper[
            mask
        ]
    )


    episode_gripper_accuracy = (
        np.mean(
            episode_true_gripper
            == episode_pred_gripper
        )
    )


    print(
        f"Episode "
        f"{int(episode):02d} "
        f"| XYZ mean error: "
        f"{episode_position_error.mean() * 100:.2f} cm "
        f"| gripper accuracy: "
        f"{episode_gripper_accuracy * 100:.1f}%"
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

result = test_df.copy()


result[
    "pred_robot_x"
] = predicted_xyz[:, 0]

result[
    "pred_robot_y"
] = predicted_xyz[:, 1]

result[
    "pred_robot_z"
] = predicted_xyz[:, 2]


result[
    "position_error_m"
] = position_error


result[
    "pred_gripper_probability"
] = predicted_gripper_probability


result[
    "pred_gripper"
] = predicted_gripper


result.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "features":
            FEATURES,

        "x_mean":
            x_mean,

        "x_std":
            x_std,

        "xyz_mean":
            xyz_mean,

        "xyz_std":
            xyz_std,

        "gripper_threshold":
            0.5,

        "train_episodes":
            list(range(1, 17)),

        "test_episodes":
            list(range(17, 21)),

        "seed":
            SEED,
    },
    MODEL_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("====================================")
print("SAVED")
print("====================================")

print(
    "Model:"
)

print(
    MODEL_FILE
)

print()

print(
    "Predictions:"
)

print(
    RESULT_FILE
)