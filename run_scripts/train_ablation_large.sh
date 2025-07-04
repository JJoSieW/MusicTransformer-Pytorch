#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train.py \
-input_dir data/preprocessed/full \
-output_dir checkpoints/v2/rpr_1_ablation \
-d_model 512 \
-num_heads 8 \
-n_layers 8 \
-continue_weights checkpoints/v2/rpr_1_ablation/weights/epoch_0100.pickle \
-continue_epoch 100 \
-batch_size 16 \
-epochs 200 \
-ce_smoothing 0.1 \
-max_sequence 768 \
--rpr \
--ablation_mode

# python train.py \
# -input_dir data/preprocessed/full \
# -output_dir checkpoints/v1/rpr_1 \
# -d_model 256 \
# -num_heads 4 \
# -n_layers 3 \
# -continue_weights checkpoints/v1/rpr_1/weights/epoch_0002.pickle \
# -continue_epoch 2 \
# -epochs 48 \
# -ce_smoothing 0.1 \
# -max_sequence 512 \
# --rpr


# python train.py \
#   -input_dir INPUT_DIR  Folder of preprocessed and pickled midi files
#   -output_dir OUTPUT_DIR
#                         Folder to save model weights. Saves one every epoch
#   -weight_modulus WEIGHT_MODULUS
#                         How often to save epoch weights (ex: value of 10 means
#                         save every 10 epochs)
#   -print_modulus PRINT_MODULUS
#                         How often to print train results for a batch (batch
#                         loss, learn rate, etc.)
#   -n_workers N_WORKERS  Number of threads for the dataloader
#   --force_cpu           Forces model to run on a cpu even when gpu is
#                         available
#   --no_tensorboard      Turns off tensorboard result reporting
#   -continue_weights CONTINUE_WEIGHTS
#                         Model weights to continue training based on
#   -continue_epoch CONTINUE_EPOCH
#                         Epoch the continue_weights model was at
#   -lr LR                Constant learn rate. Leave as None for a custom
#                         scheduler.
#   -ce_smoothing CE_SMOOTHING
#                         Smoothing parameter for smoothed cross entropy loss
#                         (defaults to no smoothing)
#   -batch_size BATCH_SIZE
#                         Batch size to use
#   -epochs EPOCHS        Number of epochs to use
#   --rpr                 Use a modified Transformer for Relative Position
#                         Representations
#   -max_sequence MAX_SEQUENCE
#                         Maximum midi sequence to consider
#   -n_layers N_LAYERS    Number of decoder layers to use
#   -num_heads NUM_HEADS  Number of heads to use for multi-head attention
#   -d_model D_MODEL      Dimension of the model (output dim of embedding
#                         layers, etc.)
#   -dim_feedforward DIM_FEEDFORWARD
#                         Dimension of the feedforward layer
#   -dropout DROPOUT      Dropout rate
#   --ablation_mode       Train an ablation model

