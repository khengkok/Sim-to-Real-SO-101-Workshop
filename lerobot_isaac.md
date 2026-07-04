# How LeRobot Interfaces with Isaac Sim

This document explains how the physical SO-101 **leader arm** (read through the
[LeRobot](https://github.com/huggingface/lerobot) library) drives the SO-101 robot inside
the **Isaac Sim / Isaac Lab** environment during teleoperation, and how the same interface
is reused for dataset recording and policy inference.

## Big picture

There is **no low-level serial code in this repo**. Communication with the physical arm is
delegated entirely to LeRobot's SO-101 driver. This repo's job is the *glue*: read joint
values from LeRobot, remap them into the simulated robot's coordinate system, and push them
into Isaac Sim as an action on every simulation step.

```
Physical SO-101 leader arm
   │  Feetech serial-bus servos over USB  (/dev/ttyACM*)
   ▼
LeRobot SO101Leader driver  ──►  robot.get_action()  →  {"shoulder_pan.pos": …, …}
   │                                                     (normalized: arm −100..100, gripper 0..100)
   ▼
LeRobotSO101Interface  (this repo)  — normalize → remap to USD joint ranges → radians
   ▼
Isaac Lab  env.step(actions)  ──►  JointPositionActionCfg  →  drives 6 sim joints
```

## The three layers

### 1. Hardware link — handled by LeRobot

`source/sim_to_real_so101/utils/lerobot_interface.py` imports `SO101LeaderConfig` from
`lerobot.teleoperators.so101_leader` (and `SO101FollowerConfig` for the follower). In
`make_cfg()` / `init_device()`:

```python
self.cfg   = SO101LeaderConfig(port=self.port, id=self.id)   # e.g. /dev/ttyACM1, "orange_teleop"
self.robot = make_robot_from_config(self.cfg)
self.robot.connect()
```

Underneath, LeRobot drives the arm's **Feetech serial-bus servos** over the USB serial
port. Port and ID come from CLI args, defaulting to the env vars `TELEOP_PORT` /
`TELEOP_ID` (set in `docker/env`). Servo calibration (raw encoder ticks 0–4095, homing
offsets) lives in the JSON files under `docker/real/scripts/sample_callibrations/` and is
consumed by LeRobot — the sim never sees it.

### 2. The teleop loop — `scripts/lerobot_agent.py`

This is the entry point (the `lerobot_agent` command). It launches Isaac Sim, builds the
gym environment, connects the leader arm, then runs this loop every frame:

```python
real_action = robot_iface.robot.get_action()                        # 1. READ leader arm
real_action, mapped_action = robot_iface.real_to_sim_obs_processor(real_action)  # 2. REMAP
actions[:] = mapped_action
obs, _, _, _, _ = env.step(actions)                                 # 3. DRIVE sim
```

`robot.get_action()` returns a dict:
`{"shoulder_pan.pos": …, "shoulder_lift.pos": …, …, "gripper.pos": …}`.

### 3. The coordinate remap — the crux of "how they talk"

The physical arm and the USD sim model **do not share the same joint ranges**, so a
conversion is required. This happens in `real_to_sim_obs_processor` →
`get_mapped_actions_vectorized` (`lerobot_interface.py`):

- **Normalize** the real values: arm joints `−100..100 → 0..1`, gripper `0..100 → 0..1`
- **Map** to the USD model's per-joint degree ranges defined in `SO101_USD_MAPPING`
  (e.g. `shoulder_pan` spans −110°..110°, `elbow_flex` −100°..90°)
- **Convert** degrees → radians

The result is an ordered tensor following `SO101_JOINT_ORDER`
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).

The per-joint mapping table (`lerobot_interface.py`):

| LeRobot joint (`SO101_JOINT_ORDER`) | USD min (deg) | USD max (deg) |
|-------------------------------------|:-------------:|:-------------:|
| `shoulder_pan`                      | −110          | 110           |
| `shoulder_lift`                     | −100          | 100           |
| `elbow_flex`                        | −100          | 90            |
| `wrist_flex`                        | −95           | 95            |
| `wrist_roll`                        | −160          | 160           |
| `gripper`                           | −10           | 100           |

## How the sim joints are actually driven

The mapped tensor is applied through Isaac Lab's action term in
`source/sim_to_real_so101/tasks/so101_env_cfg.py`:

```python
joint_positions = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    scale=1,
    use_default_offset=False,
)
```

Note the sim USD joints have **different names** (`Rotation, Pitch, …`) than the LeRobot
names. The link is **positional** — element *i* of the action tensor drives the *i*-th
joint in this list, matching `SO101_JOINT_ORDER` element-for-element:

| Index | LeRobot name    | USD joint name |
|:-----:|-----------------|----------------|
| 0     | `shoulder_pan`  | `Rotation`     |
| 1     | `shoulder_lift` | `Pitch`        |
| 2     | `elbow_flex`    | `Elbow`        |
| 3     | `wrist_flex`    | `Wrist_Pitch`  |
| 4     | `wrist_roll`    | `Wrist_Roll`   |
| 5     | `gripper`       | `Jaw`          |

When `env.step(actions)` runs, Isaac Lab commands the articulation's implicit position
actuators (stiffness/damping set in `source/sim_to_real_so101/assets/so101.py`) to those
target radians, so the sim arm mirrors the leader arm.

## Reverse direction (recording & inference)

The interface is bidirectional. `get_raw_actions_from_radians` and
`sim_to_real_dataset_processor` convert sim radians *back* to real-robot degree space:

- **Dataset recording** — teleop episodes are saved as a `LeRobotDataset`
  (`utils/lerobot_recorder.py`), triggered by the `S` key via `utils/keyboard.py`
  (`R` resets the world, also stopping recording). Sim joint positions are read back from
  `obs["policy"]["joint_pos_obs"]`; camera RGB/depth/segmentation come from the
  `obs["visual"]` group.
- **Policy inference** — `sim_obs_to_policy_processor` / `predict_action` /
  `prediction_to_sim_processor` feed sim observations to a LeRobot policy and map its
  action back into sim radians. `GR00TRemotePolicy` does the same against a remote GR00T
  VLA server (ZeroMQ, `gr00t_client/server_client.py`).

## Key files

| File | Role |
|------|------|
| `source/sim_to_real_so101/scripts/lerobot_agent.py` | Teleop entry point; the read→remap→step loop. |
| `source/sim_to_real_so101/utils/lerobot_interface.py` | `LeRobotSO101Interface`: hardware config + real↔sim joint mapping. |
| `source/sim_to_real_so101/tasks/so101_env_cfg.py` | Isaac Lab env: `JointPositionActionCfg` action term, observations. |
| `source/sim_to_real_so101/assets/so101.py` | `SO101_CFG` articulation (USD, actuator gains, USD joint names). |
| `source/sim_to_real_so101/utils/lerobot_recorder.py` | Writes teleop episodes to a `LeRobotDataset`. |
| `source/sim_to_real_so101/utils/keyboard.py` | Omniverse keyboard bindings (`R` reset, `S` record, `C` cancel). |
| `docker/env` | Port/ID config (`TELEOP_PORT`, `TELEOP_ID`, `ROBOT_PORT`, `ROBOT_ID`, cameras). |

## Reading order

`scripts/lerobot_agent.py` (the loop) → `utils/lerobot_interface.py` (the mapping) →
`tasks/so101_env_cfg.py` (the sim action term) → `assets/so101.py` (the robot model).

## Related

For running the leader arm on a **separate machine** from the sim (local arm + cloud GPU),
see the "Remote Teleoperation" section in [`README.md`](README.md), which streams
`robot.get_action()` output over ZeroMQ into this same pipeline.
