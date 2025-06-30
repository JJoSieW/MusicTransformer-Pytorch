#!/bin/bash

python generate.py \
-midi_root data/preprocessed/full \
-output_dir checkpoints/v1/rpr_1/samples \
-model_weights checkpoints/v1/rpr_1/weights/epoch_0043.pickle \
-d_model 256 \
-num_heads 4 \
-n_layers 3 \
-max_sequence 512


# parser.add_argument("-midi_root", type=str, required=True, help="Midi file to prime the generator with")
# parser.add_argument("-output_dir", type=str, default="./gen", help="Folder to write generated midi to")
# parser.add_argument("-primer_file", type=str, default= None, help="File path or integer index to the evaluation dataset. Default is to select a random index.")
# parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

# parser.add_argument("-target_seq_length", type=int, default=1024, help="Target length you'd like the midi to be")
# parser.add_argument("-num_prime", type=int, default=256, help="Amount of messages to prime the generator with")
# parser.add_argument("-model_weights", type=str, default="./saved_models/model.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")
# parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")

# parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
# parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
# parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
# parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
# parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

# parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
