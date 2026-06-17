# SERL-Plus-Plus
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

> This repository is built upon a fork of [HIL-SERL](https://github.com/rail-berkeley/hil-serl).

## Requirements

- Python 3.10
- CUDA 12.4+ (recommended for GPU acceleration)
- PyTorch 2.4.1+
- MuJoCo 2.3.7+
- See `pyproject.toml` for full dependency list

## Algorithm

- **SAC**: Deep reinforcement learning based on Soft Actor-Critic

## Installation

```bash
# clone repo
git clone <repository-url>
# cd folder
cd serl-torch
# create venv by uv
uv sync
# source venv
source .venv/bin/activate
```

## Quick Start
### 1. Peg insert sim

![peg_insert_sim](./doc/peg_insert_sim.gif)

#### 1. Train RLPD (Drq, SAC)
```bash
# cd peg_insert_sim
cd demos/experiments/peg_insert_sim
# Record demo
python ../../record_demo.py --exp_name peg_insert_sim
# Start learner node
bash run_learner.sh
# Open new terminal, start actor node
bash run_actor.sh
```

## Reference
Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning [HIL-SERL](https://github.com/rail-berkeley/hil-serl)