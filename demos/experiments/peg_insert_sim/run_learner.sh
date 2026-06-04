#!/bin/bash

python ../../train_rlpd.py \
    --exp_name=peg_insert_sim \
    --demo_path=./demo_data/peg_insert_sim_20_demos_2026-01-19_22-24-15.pkl \
    --checkpoint_path=ckpt \
    --learner \