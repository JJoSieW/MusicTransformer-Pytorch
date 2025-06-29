#!/bin/bash

python generate.py \
-midi_root data/preprocessed/small-old-more \
-output_dir results/samples \
-model_weights checkpoints/v0/no_rpr/results/best_acc_weights.pickle \
-d_model 256 \
-num_heads 4 \
-n_layers 3
