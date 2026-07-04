# Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac

![SO-101 Vial to Rack Task](images/so101_banner.png)

Welcome to this workshop on sim-to-real transfer for the SO-101 robot!

This repository contains the assets and code to accompany this [learning content](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

The rest of this README will help you setup the environment and ensure everything is installed correctly.

You can also use this repo as a basis for trying out your own tasks.

## Requirements

This content was tested on the following GPUs:

- NVIDIA RTX 6000 Pro (Blackwell)
- NVIDIA RTX 5090 (Blackwell)
- NVIDIA RTX 6000 (Ada)

OS and Software tested:
- Ubuntu Linux >22.04
- Docker
- CUDA Toolkit
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


## Installation

1. Create directory and clone this repo
```bash
cd ~/
git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git
```

### Building the Docker images

2. Navigate to the repo
```bash
cd ~/Sim-to-Real-SO-101-Workshop
```

#### Teleop & Simulation container

3. From the repo root directory, run:
```bash
docker build -t teleop-docker -f docker/sim/Dockerfile .
```

#### Real Robot & Inference Server - this may take a while to build


For **Blackwell** architecture GPUs:

4. From the repo root directory, run:
```bash
./docker/real/build.sh blackwell
```
For **Ada** architecture GPUs:

4. From the repo root directory:
```bash
./docker/real/build.sh ada
```

5. Continue with the course instructions [here](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

### Starting the images

To start the Teleop & Simulation container:

```bash
xhost + 
docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
   -v ./docker/env:/root/env \
   -v $(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source \
   -v $(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs \
   -v $(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets \
   teleop-docker:latest
```

To start the Real Robot & Inference Server:

```bash
docker run -it --rm --name real-robot --network host --privileged --gpus all \
    -e DISPLAY \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
    -v ./docker/env:/root/env \
    -v ~/models:/workspace/models \
    -v $(pwd)/docker/real/scripts:/workspace/Isaac-GR00T/gr00t/eval/real_robot/SO100 \
    real-robot \
    /bin/bash
```

## Remote Teleoperation

By default, `lerobot_agent` assumes the leader arm and Isaac Sim run on the **same
machine** — it reads the arm from a local serial port (`/dev/ttyACM*`) and drives the
simulation on the local GPU.

If your simulation runs on a **remote cloud GPU server** but the leader arm is plugged
into your **local machine**, you can split the teleop loop across the two. A small local
server reads the arm and streams only the six joint values to the cloud over a ZeroMQ
socket — the chatty serial traffic stays local, so a slow serial bus never stalls on
network latency.

```
LOCAL machine (arm)                          CLOUD server (Isaac Sim)
┌─────────────────────┐   ZMQ REQ/REP (TCP)  ┌───────────────────────────┐
│ leader_server.py    │◄─────────────────────┤ lerobot_agent.py           │
│  reads /dev/ttyACM*  │──► {joint: deg} ─────┼─► sim env.step(actions)    │
└─────────────────────┘   only 6 floats/step └───────────────────────────┘
```

### 1. Start the leader server (local machine)

On the machine the physical **leader** arm is connected to. This needs only `pyzmq`,
`msgpack`, and `lerobot` — no Isaac Sim or GPU.

```bash
python source/sim_to_real_so101/scripts/leader_server.py \
    --port /dev/ttyACM0 --robot_id leader_arm_1 \
    --host 0.0.0.0 --zmq_port 5556
```

### 2. Make the local machine reachable from the cloud (SSH reverse tunnel)

The cloud server usually cannot open a connection back to your local machine (NAT /
firewall). Rather than exposing the port publicly, forward it through an **SSH reverse
tunnel**. From your **local machine**, open a tunnel to the cloud server:

```bash
ssh -N -R 5556:localhost:5556 user@cloud-server
```

This makes `localhost:5556` **on the cloud server** forward to the leader server running
on your local machine. Keep this session open for the duration of teleoperation.
(Add `-o ServerAliveInterval=30` to keep the tunnel alive on idle connections.)

### 3. Start teleop against the remote leader (cloud server)

Inside the Teleop & Simulation container on the cloud, point `lerobot_agent` at the
tunnel endpoint. Because the reverse tunnel terminates on the cloud's `localhost`, use
`--leader_host localhost`:

```bash
lerobot_agent \
    --task Lerobot-So101-Teleop-Base \
    --remote_leader --leader_host localhost --leader_zmq_port 5556
```

Recording (`--repo_id …`, `S` key) works unchanged: camera frames are rendered inside the
sim on the cloud and never depended on the local hardware.

> **Notes**
> - **Latency caps the rate.** The loop makes one network round-trip per simulation
>   step, so a high round-trip time lowers the effective teleop frame rate. A tunnel over
>   a low-latency link keeps it usable.
> - **Security.** If you cannot use an SSH tunnel and must expose the port directly, set a
>   shared secret via `--api_token` on the server and `LEADER_API_TOKEN` on the client.

## Models and Datasets

### Downloading model weights

First, [install the HuggingFace command-line-interface (CLI)](https://huggingface.co/docs/huggingface_hub/en/guides/cli#command-line-interface-cli)

The models used in the course are listed in the course instructions [here](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/datasets-and-models.html).

You can either download them ahead of time, or as you get to them in the course.

## Tasks

### Tasks

#### Debug envs
- `Lerobot-So101-Teleop-Base` : Teleop debug
- `Lerobot-So101-Teleop-Task` : Lightbox, cameras, non-task related debug

#### Tasks
- `Lerobot-So101-Teleop-Vials-To-Rack` - Main task for the workshop - pick up the vial and place it in the yellow rack
- `Lerobot-So101-Teleop-Vials-To-Rack-DR` - Same as above but with domain randomization

#### Eval
- `Lerobot-So101-Teleop-Vials-To-Rack-Eval` - Evaluation without domain randomization (fixed orange robot, no lighting/mat DR)
- `Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval` - Evaluation with full domain randomization


## Commands

- `list_envs` - List environments in this repo
- `zero_agent` - Debug script with zero actions
- `random_agent` - Debug script with random actions
- `lerobot_agent` - LeRobot SO101 teleop script
  - `--remote_leader` - Read the leader arm from a remote `leader_server.py` instead of a local serial port (see [Remote Teleoperation](#remote-teleoperation))
  - `--leader_host` - Host of the remote leader server; required with `--remote_leader` (env: `LEADER_HOST`)
  - `--leader_zmq_port` - TCP port of the remote leader server, default `5556` (env: `LEADER_ZMQ_PORT`)
- `lerobot_eval` - Model evaluation script
- `lerobot_push_dataset` - LeRobot Dataset push to hub script

## Contributions
We are not currently accepting contributions for this project.
