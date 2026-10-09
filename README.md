\# Multimodal Human-to-Robot Retargeting from RGB-D and Capacitive Interaction Sensing



A low-cost real-to-sim pipeline that converts personally collected iPhone RGB-D hand demonstrations and capacitive interaction cues into learned Franka Panda manipulation trajectories in MuJoCo.

**## 🎥 Final Demonstration**



**\*\*\[▶ Watch the final human-to-Panda demonstration](assets/demo.mov)\*\***



**The video shows the personally collected human manipulation data alongside a held-out learned-policy execution on a Franka Emika Panda in MuJoCo.**



**\*\*Held-out Episode 19:\*\* 0.79 cm mean 3D position error.**



**> The Panda trajectory and gripper state are policy predictions. Cube attachment is event-conditioned visualization rather than physics-based grasping.**



\## Demo



I collected 20 short manipulation demonstrations myself using an iPhone RGB-D recording and an MPR121 capacitive sensing setup.



The pipeline is:



\*\*Human demonstration → RGB-D + capacitance → synchronization → human features → learned policy → Panda XYZ + gripper\*\*



The learned policy predicts a 4-DoF robot action:



\- 3D end-effector position: X, Y, Z

\- Binary gripper state: open / closed



It is evaluated on complete episodes that were not used for training.



\## Personally Collected Data



Each demonstration consists of a simple manipulation sequence:



1\. Approach the object

2\. Contact / pick up

3\. Lift

4\. Return toward the surface

5\. Release

6\. Retreat



The sensing setup uses:



\- iPhone RGB video at 60 FPS

\- iPhone depth / LiDAR

\- MediaPipe hand tracking

\- MPR121 capacitive sensing with a copper electrode on the manipulation surface



RGB/hand tracking provides image-plane hand position, LiDAR provides depth, and the capacitive electrode provides a complementary surface-relative interaction cue.



The capacitive signal is \*\*not treated as a direct grasp detector or long-range height sensor\*\*.



\## Synchronization



The phone and capacitive logger use separate clocks.



Two visible/touch synchronization events were recorded at the beginning of the session. Capacitive measurements were aligned to video time and interpolated using their recorded PC timestamps rather than assuming a fixed sensor sampling frequency.



For the recorded session:



\- Video: 60 FPS

\- Capacitive sampling: approximately 46.27 Hz average

\- 3,342 RGB frames

\- 3,342 depth frames

\- 5,401 capacitive samples

\- Hand/depth detections on 2,998 / 3,342 frames (89.7%)

\- 20 extracted manipulation episodes



\## Dataset Split



The split is performed by \*\*episode\*\*, not by randomly splitting individual frames.



\- Training: Episodes 1–16

\- Held-out evaluation: Episodes 17–20

\- Training frames: 2,360

\- Test frames: 469



All episodes were recorded during one collection session, so this is not a cross-session generalization test.



\## Policy



I trained a small behaviour-cloning / learned-retargeting MLP in PyTorch.



\### Inputs



For each timestep the network receives:



\- Hand X

\- Hand Y

\- LiDAR depth

\- Hand grasp signal

\- Hand velocity X

\- Hand velocity Y

\- Hand velocity Z

\- Smoothed capacitive measurement



\### Outputs



The policy has two output heads:



\- Panda end-effector XYZ

\- Gripper open/close probability



Architecture:



`8 → 64 → 64 → 32`



followed by separate XYZ and gripper heads.



The XYZ supervision comes from a deterministic human-to-Panda retargeting mapping.



The gripper supervision comes from manually annotated object-held intervals based on first-contact and release events in the demonstrations.



Therefore, this should be interpreted as \*\*learned multimodal retargeting / behaviour cloning\*\*, rather than an autonomous vision-language-action policy.



\## Held-Out Results



Evaluation is performed on Episodes 17–20.



\### Position prediction



| Metric | Result |

|---|---:|

| X RMSE | 0.93 cm |

| Y RMSE | 4.01 cm |

| Z RMSE | 0.92 cm |

| Overall coordinate RMSE | 2.43 cm |

| Mean 3D position error | \*\*1.81 cm\*\* |

| Median 3D position error | \*\*0.53 cm\*\* |



\### Gripper prediction



| Metric | Result |

|---|---:|

| Accuracy | \*\*84.2%\*\* |

| Precision | 89.3% |

| Recall | 77.9% |

| F1 | \*\*83.2%\*\* |



\### Per-episode results



| Episode | Mean XYZ error | Gripper accuracy |

|---|---:|---:|

| 17 | 0.51 cm | 88.5% |

| 18 | 0.64 cm | 82.2% |

| 19 | 0.79 cm | 72.9% |

| 20 | 4.76 cm | 94.7% |



Episode 20 is a clear position-prediction failure case. This highlights the limited generalization of a small model trained on data from a single recording session.



\## MuJoCo Retargeting



The learned policy is replayed on a Franka Emika Panda using MuJoCo.



The predicted XYZ targets are converted to Panda joint configurations using position-based Jacobian inverse kinematics.



The gripper actuator is controlled from the \*\*learned gripper prediction\*\*, rather than the ground-truth object-held label.



A red cube is included in the demonstration visualization. Cube attachment is event-conditioned from the predicted gripper state and temporally debounced for visualization stability.



\*\*The cube visualization is not a physics-based grasp.\*\*



Collisions for the visual cube are disabled, so the simulation evaluates learned trajectory retargeting and interaction timing rather than robust contact dynamics.



\## What Worked



\- Low-cost personally collected multimodal data was sufficient to drive a simulated Panda.

\- Episode-level held-out evaluation prevents leakage from randomly splitting adjacent video frames.

\- The learned policy reproduces the retargeted trajectories accurately on Episodes 17–19.

\- A single network predicts both end-effector position and gripper state.

\- RGB-D and capacitive measurements can be synchronized despite different sampling rates.



\## What Did Not Work



\### Generalization



Episode 20 has substantially higher XYZ error than the other held-out episodes, showing that the current dataset is too small and homogeneous for strong generalization claims.



\### Full orientation IK



I experimented with constraining full end-effector orientation, but it produced unstable behaviour. The final implementation therefore uses stable position-only Jacobian IK.



\### Physics-based grasping



The focus of this prototype is multimodal retargeting rather than contact-rich manipulation. The cube demonstration uses event-conditioned visual attachment rather than simulated contact forces.



\### LIBERO



I initially explored LIBERO, but dependency compatibility between its older robosuite/MuJoCo stack and the development environment became a distraction from the core experiment. I therefore used modern MuJoCo directly as the simple simulation environment.



\## Repository Structure



```text

.

├── README.md

├── requirements.txt

├── assets/

├── data/

│   ├── episodes\_summary.csv

│   └── test\_predictions\_xyz\_gripper.csv

└── src/

&#x20;   ├── record\_capacitance.py

&#x20;   ├── process\_demo.py

&#x20;   ├── synchronize\_demo03.py

&#x20;   ├── extract\_episodes\_demo03.py

&#x20;   ├── make\_training\_dataset.py

&#x20;   ├── train\_bc.py

&#x20;   ├── play\_action\_stable.py

&#x20;   └── play\_learned\_policy.py

```



\## Installation



Tested with Python 3.11 and MuJoCo 3.1.5.



```bash

pip install -r requirements.txt

```



The simulation uses the Franka Emika Panda model from the MuJoCo Menagerie.



Some scripts currently contain local path constants for the dataset/project directory and Panda model directory. Update these paths for your machine before running.



\## Running the Pipeline



The main stages are:



```bash

python src/process\_demo.py

python src/synchronize\_demo03.py

python src/extract\_episodes\_demo03.py

python src/make\_training\_dataset.py

python src/train\_bc.py

```



To replay the learned held-out policy in MuJoCo:



```bash

python src/play\_learned\_policy.py

```



`play\_learned\_policy.py` expects the trained model/prediction files and a local MuJoCo Menagerie Panda installation at the paths configured near the top of the script.



\## Limitations



This is a small prototype based on 20 demonstrations from a single recording session.



The learned XYZ targets imitate a deterministic retargeting function rather than being trained from robot demonstrations. Gripper targets are manually annotated from interaction events.



The simulation uses position-based IK and does not evaluate physical grasp robustness.



Future work would collect multiple sessions and environments, use physical object contacts in simulation, evaluate cross-session generalization, and learn actions from richer temporal observations.



\## Summary



This project investigates whether personally collected, low-cost multimodal human demonstrations can be transformed into a compact learned robot-control representation.



With only 16 training episodes, the resulting policy predicts Panda end-effector position and gripper state and achieves \*\*1.81 cm mean held-out 3D position error\*\* and \*\*84.2% held-out gripper accuracy\*\*, while also exposing a clear generalization failure on one of the four held-out episodes.

