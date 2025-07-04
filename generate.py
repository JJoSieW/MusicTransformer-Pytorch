import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

from third_party.processor_copy import (
    RANGE_NOTE_ON,
    RANGE_NOTE_OFF,
    RANGE_TIME_SHIFT,
    RANGE_VEL,
    RANGE_CONTOUR_INTERVAL,
    RANGE_CONTOUR_DURATION,
)


START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    'contour_interval': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL,
    'contour_duration': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL + RANGE_CONTOUR_INTERVAL
}

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = encode_midi(f)
        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # INTERACTIVE GENERATION USING ROLLING WINDOW + CONTOUR CONTROL
    model.eval()
    with torch.set_grad_enabled(False):
        print("Rolling contour-controlled generation...")
        output_sequence = primer[:args.num_prime].tolist()
        current_sequence = primer[:args.num_prime].clone().unsqueeze(0)  # shape [1, T]

        while True:
            print("\nEnter new contour_interval (int between -100 and 100: negative for down, 0 for flat, positive for up), or 'q' to quit:")
            contour_input = input(">> ").strip()
            if contour_input.lower() == 'q':
                break

            try:
                contour_interval = int(contour_input)+100
            except ValueError:
                print("Invalid input. Please enter an integer.")
                continue

            print("Enter contour_duration (in time units (0.01s), int between 0 and 199):")
            try:
                contour_duration = int(input(">> ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
                continue

            # Construct input sequence with contour tokens
            CONT_INTERVAL = START_IDX["contour_interval"] + contour_interval
            CONT_DURATION = START_IDX["contour_duration"] + contour_duration
            contour_tokens = [CONT_INTERVAL, CONT_DURATION]

            input_seq = current_sequence[0].tolist() + contour_tokens
            # 「滚动窗口长度控制」
            # •	如果当前窗口长度超了，就自动截断最旧的 token；
            # •	生成前还会显示一个提醒，告诉你正在截断老的数据，避免爆长报错。
            if len(input_seq) > args.max_sequence:
                print(f"\n⚠️ Warning: Rolling window reached max length {args.max_sequence}. Oldest tokens will be discarded.")
                input_seq = input_seq[-args.max_sequence:]
            input_tensor = torch.tensor(input_seq, dtype=TORCH_LABEL_TYPE, device=get_device()).unsqueeze(0)

            # Compute num_primer dynamically based on current input length
            num_primer = input_tensor.shape[1]

            # Generate next block conditioned on this contour
            generated = model.generate(input_tensor, args.target_seq_length, beam=args.beam)

            # Append only the newly generated part
            new_tokens = generated[0].tolist()[len(input_seq):]
            output_sequence.extend(contour_tokens + new_tokens)

            # Update rolling window
            current_sequence = torch.tensor(output_sequence[-args.max_sequence:], dtype=TORCH_LABEL_TYPE, device=get_device()).unsqueeze(0)

            print(f"Generated {len(new_tokens)} new tokens. Current total: {len(output_sequence)} tokens.")

        # Save final output
        f_path = os.path.join(args.output_dir, "rolling_contour_output.mid")
        decode_midi(output_sequence, file_path=f_path)
        print(f"Final output saved to {f_path}")




if __name__ == "__main__":
    main()
