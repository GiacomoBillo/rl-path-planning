# Fork: Avoid Everything (WIP)

I had some trouble using the original project, so I made this fork to fix the issues with the code and make a more stable development container for the project that anyone can use. I added my own fork of robofin package as a submodule, so that I could edit its contents to make it more general, not specific to the Franka Panda robot, for which the fishbotics project is currently adapted. I might add the atob project as a submodule as well. 

## Quick start

This section describes how you can training going with the simulated Franka Panda robot, which was used for the original Avoid Everything project. If you want to use another robot, you need to find a `.urdf` file for it, provide a configuration file with the same fields as `/workspace/assets/panda/robot_config.yaml`, and you need to generate a sphere representation of the robot, see **Spherification** below.

First, open up this workspace in the Docker devcontainer, ideally using VSCode (or a fork thereof), alternatively using docker compose manually.

Once inside the built devcontainer, download the training data for the cubby environment:
```
cd /workspace
mkdir datasets
cd datasets
wget https://zenodo.org/records/15249565/files/cubby_pretraining_data.zip?download=1 -o cubby_pretraining_data.zip
unzip cubby_pretraining_data.zip
```

Optionally, download the trained model checkpoints:
```
cd /workspace
mkdir checkpoints
cd checkpoints
wget https://zenodo.org/records/15249565/files/mpiformer_cubby.ckpt?download=1 -o mpiformer_cubby.ckpt
wget https://zenodo.org/records/15249565/files/avoid_everything_cubby.ckpt?download=1 -o avoid_everything_cubby.ckpt
```

In the model config file for testing, `/workspace/model_configs/col_test.yaml`, set the `data_dir` parameter to 
the path where the `train/` and `val/` data folders are.

Optionally, set `load_model_from_checkpoint: true` and set `load_checkpoint_path` to the file path where you have the checkpoint that you want to start from. Set `load_actor_only: true` if using the downloaded checkpoints, since they do not include critic weights. 

Most parameters are documented in `/workspace/model_configs/col_test.yaml`. Notably, `pretraining_steps` decides how many training steps to take before the replay buffer starts getting sampled from. Note that the training will appear to slow down at this point, judging from the progress bar, but this is not the case. It simply draws samples from the expert dataset at a slower rate, due to some samples being drawn from the replay buffer instead (ratio decided by parameter `expert_fraction`, default 25% -> looks like a 1/4 slowdown). `start_using_actor_loss` gives the number of global training steps to take before using the critic-guided actor loss.

Setting `logging: true` will log to wandb, log into wandb (with API key) in the devcontainer terminal before running training with `logging: true`.

Run the training with
```
python3 avoid_everything_except_exploration/run_training.py model_configs/col_test.yaml
```

### Spherification

Check out the `spherification/README.md` if you want to create a collisions sphere representation of your robot, based on its `.urdf`. Collision spheres and self-collision spheres are required if you want to use the Avoid Everything project without modification. My forked version of robofin expects the file structure:
```
robot_directory/
├── robot.urdf                           # Original URDF
├── collision_spheres/
│   ├── collision_spheres.json           # Collision spheres
│   └── self_collision_spheres.json      # Self-collision spheres
└── meshes/
    ├── visual/                          # Visual meshes
    └── collision/                       # Collision meshes
```
where `collision_spheres.json` and `self_collision_spheres.json` store the collision spheres and self-collision spheres respectively for each of the robot's links.

---
Original README below:
---

# Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning

This repository contains the official implementation of the paper **"Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning"** presented at CoRL 2024 by Fishman et al.

## Table of Contents

- [Overview](#overview)
- [Data and Checkpoints](#data-and-checkpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
  - [Pretraining](#pretraining)
  - [ROPE Fine-tuning](#rope-fine-tuning)
- [Data Generation](#data-generation)
- [License](#license)
- [Citation](#citation)

## Overview

Avoid Everything introduces a novel approach to generating collision-free motion for robotic manipulators in cluttered, partially observed environments. The system combines:

- **Motion Policy Transformer (MπFormer)**: A transformer architecture for joint space control using point clouds
- **Refining on Optimized Policy Experts (ROPE)**: A fine-tuning procedure that refines motion policies using optimization-based demonstrations

The system achieves over 91% success rate in challenging manipulation scenarios while being significantly faster than traditional planning approaches.

## Installation

Note: these installation instructions were adapted from [Motion Policy Networks](https://github.com/NVlabs/motion-policy-networks) (Fishman et al. 2022).

The easiest way to install the code here is to build our included docker container,
which contains all of the dependencies for data generation, model training,
inference. While it should be possible to run all of the training code in CUDA or with a Virtual Environment, we use Docker in this official implementation because it makes it easier to install the dependencies for data generation, notably [OMPL](https://ompl.kavrakilab.org/) requires a lot of system dependencies before building from source.

If you have a strong need to build this repo on your host machine, you can follow the same steps as are outlined in the [Dockerfile](docker/Dockerfile).

To build the docker and use this code, you can follow these steps:


First, clone this repo using:
```
git clone https://github.com/fishbotics/avoid-everything.git

```
Navigate inside the repo (e.g. `cd avoid-everything`) and build the docker with

```
docker build --tag avoid-everything --network=host --file docker/Dockerfile .

```
After this is built, you should be able to launch the docker using this command
(be sure to use the correct paths on your system for the `/PATH/TO/THE/REPO` arg)

```
docker run --interactive --tty --rm --gpus all --network host --privileged --env DISPLAY=unix$DISPLAY --volume /PATH/TO/THE/REPO:/root/avoid-everything avoid-everything /bin/bash -c 'export PYTHONPATH=/root/avoid-everything:$PYTHONPATH; git config --global --add safe.directory /root/avoid-everything; /bin/bash'
```
In order to run any GUI-based code in the docker, be sure to add the correct
user to `xhost` on the host machine. You can do this by running `xhost
+si:localuser:root` in another terminal on the host machine.

Our suggested development setup would be to have two terminals open, one
running the docker (use this one for running the code) and another editing
code on the host machine. The `docker run` command above will mount your
checkout of this repo into the docker, allowing you to edit the files from
either inside the docker or on the host machine.

## Usage

### Pretrained Models
We provide pretrained models for both the base MπFormer and ROPE-finetuned versions:
- Base MπFormer checkpoint: [Link To Be Posted Later]
- Avoid Everything checkpoint (with ROPE and DAgger): [Link To Be Posted Later]

You can find the data and checkpoints from the paper on [Zenodo](https://zenodo.org/records/15249565).

## Running the evaluations
To run evaluations with the pretrained model in either the cubby or tabletop environment, you must first download the data and checkpoints from [Zenodo](https://zenodo.org/records/15249565). After downloading the data, you can modify the `evaluation.yaml` file to point to your data and your checkpoint. Note that if you're using the Docker, these should be paths within the Docker container. Then, you can the sript `run_validation_rollouts.py` and point it to your `evaluations.yaml` config. This script will load the checkpoint and the validation dataset, run rollouts, and return the metrics.

## Training

### Pretraining
The pretraining configuration can be found in `pretraining.yaml`. Key parameters include:
- collision_loss_weight: 5
- point_match_loss_weight: 1
- min_lr: 1.0e-5
- max_lr: 5.0e-5
- warmup_steps: 5000

To start pretraining, run:
```bash
python avoid_everything/train.py pretraining.yaml
````

### ROPE Fine-tuning

ROPE fine-tuning uses the configuration in `rope.yaml`. To start fine-tuning:

```bash
python avoid_everything/train.py rope.yaml
```

## Data Generation

The data generation pipeline supports creating both pretraining and fine-tuning datasets. Based on `avoid_everything/data_generation.py`, the system can generate:

- Training trajectories in random environments
- Test scenarios for evaluation
- Expert demonstrations for ROPE fine-tuning

## License

MIT License. See LICENSE file for details.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{fishman2024avoideverything,
  title={Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning},
  author={Fishman, Adam and Walsman, Aaron and Bhardwaj, Mohak and Yuan, Wentao and Sundaralingam, Balakumar and Boots, Byron and Fox, Dieter},
  booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
  year={2024}
}

```
