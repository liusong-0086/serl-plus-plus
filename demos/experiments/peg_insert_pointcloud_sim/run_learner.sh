#!/bin/bash

python ../../train_rlpd.py \
    --exp_name=peg_insert_pointcloud_sim \
    --demo_path=./demo_data/peg_insert_pointcloud_sim_20_demos.pkl \
    --checkpoint_path=ckpt \
    --agent_type=sac3 \
    --learner \