#!/bin/bash

python evaluate.py \
-dataset_dir data/preprocessed/small-old-more \
-model_weights checkpoints/v0/no_rpr/results_0/best_acc_weights.pickle \
-d_model 256 \
-num_heads 4 \
-n_layers 3


# it will print
# Evaluating:
# Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.19batch/s, Loss=5.8641, Acc=0.0415]
# Avg loss: 5.862298488616943
# Avg acc: 0.05126953125