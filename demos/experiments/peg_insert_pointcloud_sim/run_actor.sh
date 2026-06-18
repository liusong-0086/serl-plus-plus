#!/bin/bash

python ../../train_rlpd.py \
    --exp_name=peg_insert_pointcloud_sim \
    --checkpoint_path=ckpt \
    --agent_type=sac3 \
    --actor \