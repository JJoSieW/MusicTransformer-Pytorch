#!/bin/bash

python evaluate.py \
-dataset_dir data/preprocessed/full \
-model_weights checkpoints/v1/rpr_1/weights/epoch_0034.pickle \
-ce_smoothing 0.1 \
-d_model 256 \
-num_heads 4 \
-n_layers 3 \
-batch_size 2 \
-max_sequence 512 \
--rpr


# parser.add_argument("-dataset_dir", type=str, required=True, help="Folder of preprocessed and pickled midi files")
# parser.add_argument("-model_weights", type=str, default="./saved_models/model.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")
# parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
# parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

# parser.add_argument("-batch_size", type=int, default=2, help="Batch size to use")

# parser.add_argument("-ce_smoothing", type=float, default=None, help="Smoothing parameter for smoothed cross entropy loss (defaults to no smoothing)")
# parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
# parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider in the model")
# parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
# parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
# parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

# parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

