import torch
import torch.nn as nn
import os
import random
import json

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

# 添加绘图相关的导入
import pretty_midi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse


START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    'contour_interval': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL,
    'contour_duration': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL + RANGE_CONTOUR_INTERVAL
}

def generate_single(args):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # if args.force_cpu:
    #     use_cuda(False)
    #     print("WARNING: Forced CPU usage, expect model to perform slower")
    #     print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if args.primer_file is None:
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if f.isdigit():
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = encode_midi(f)
        if len(raw_mid) == 0:
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # Save primer midi
    primer_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=primer_path)

    # Prepare contour tokens
    contour_interval = args.contour_interval
    if not (-100 <= contour_interval <= 100):
        print("Error: contour_interval must be between -100 and 100.")
        return
    contour_interval_idx = START_IDX["contour_interval"] + (contour_interval + 100)

    contour_duration_sec = args.contour_duration
    if not (1.0 <= contour_duration_sec <= 15.0):
        print("Error: contour_duration must be between 1.0 and 15.0 seconds.")
        return
    contour_duration_idx = START_IDX["contour_duration"] + int(round(contour_duration_sec * 10))

    contour_tokens = [contour_interval_idx, contour_duration_idx]

    # Construct input sequence with contour tokens appended to primer
    input_seq = primer[:args.num_prime].tolist() + contour_tokens

    if len(input_seq) > args.max_sequence:
        print(f"⚠️ Warning: Input sequence length {len(input_seq)} exceeds max_sequence {args.max_sequence}. Truncating oldest tokens.")
        input_seq = input_seq[-args.max_sequence:]

    input_tensor = torch.tensor(input_seq, dtype=TORCH_LABEL_TYPE, device=get_device()).unsqueeze(0)

    model.eval()
    with torch.set_grad_enabled(False):
        generated = model.generate(input_tensor, args.target_seq_length, beam=args.beam)

    generated_tokens = generated[0].tolist()

    # Full output sequence: primer + contour tokens + generated continuation tokens
    output_sequence = generated_tokens

    # Save full music output
    full_music_path = os.path.join(args.output_dir, "music.mid")
    decode_midi(output_sequence, file_path=full_music_path)
    print(f"Full generated music saved to: {full_music_path}")

    # Load primer midi to get end time
    primer_data = pretty_midi.PrettyMIDI(primer_path)
    primer_end_time = primer_data.get_end_time() if primer_data.instruments else 0

    # Load full generated midi to extract conditioned segment
    generated_data = pretty_midi.PrettyMIDI(full_music_path)

    segment_start = primer_end_time
    segment_end = primer_end_time + contour_duration_sec

    # Filter notes in generated_data that start within [segment_start, segment_end]
    conditioned_notes = []
    for instrument in generated_data.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            if note.start >= segment_start and note.start < segment_end:
                new_instrument.notes.append(note)
        if len(new_instrument.notes) > 0:
            conditioned_notes.append(new_instrument)

    # Create new PrettyMIDI object for conditioned segment
    conditioned_pm = pretty_midi.PrettyMIDI()
    for instr in conditioned_notes:
        for note in instr.notes:
            note.start -= segment_start
            note.end -= segment_start
        conditioned_pm.instruments.append(instr)

    condition_segment_path = os.path.join(args.output_dir, "condition_segment.mid")
    conditioned_pm.write(condition_segment_path)
    print(f"Conditioned segment saved to: {condition_segment_path}")

    # Save contour info as JSON
    contour_info = {
        "interval": contour_interval,
        "duration": contour_duration_sec
    }
    contour_json_path = os.path.join(args.output_dir, "contour.json")
    with open(contour_json_path, "w") as fjson:
        json.dump(contour_info, fjson, indent=4)
    print(f"Contour info saved to: {contour_json_path}")

    # Optional plotting
    try:
        plot_primer_vs_generated(primer_path, full_music_path, args.output_dir)
    except Exception as e:
        print(f"Plotting failed: {e}")

def main():
    
    parser = argparse.ArgumentParser(description="Generate music with contour control (single generation).")
    # Add existing generate.py args by parsing first then adding new ones
    # We reuse parse_generate_args for common args - but parse_generate_args uses argparse internally without returning parser.
    # So we replicate necessary args here:
    parser.add_argument('--primer_file', default=None, help='Primer file path or dataset index')
    parser.add_argument('--midi_root', required=True, help='Root directory for midi dataset')
    parser.add_argument('--num_prime', type=int, default=512, help='Number of primer tokens')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--max_sequence', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--rpr', action='store_true', help='Use relative positional representations')
    parser.add_argument('--model_weights', required=True, help='Path to model weights')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--target_seq_length', type=int, default=512, help='Length of generation target sequence')
    parser.add_argument('--beam', type=int, default=1, help='Beam size for generation')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    # New required args:
    parser.add_argument('--contour_interval', type=int, required=True, help='Contour interval (int between -100 and 100)')
    parser.add_argument('--contour_duration', type=float, required=True, help='Contour duration in seconds (float between 1.0 and 15.0)')

    args = parser.parse_args()
    generate_single(args)

def plot_primer_vs_generated(primer_file, generated_file, output_dir):
    """
    绘制primer和generated sample的钢琴卷帘图对比
    """
    try:
        # 加载两个MIDI文件
        primer_data = pretty_midi.PrettyMIDI(primer_file)
        generated_data = pretty_midi.PrettyMIDI(generated_file)
        
        # 获取音符
        primer_notes = []
        for instrument in primer_data.instruments:
            for note in instrument.notes:
                primer_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        generated_notes = []
        for instrument in generated_data.instruments:
            for note in instrument.notes:
                generated_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # 设置颜色
        primer_color = 'blue'
        generated_color = 'red'
        
        # 绘制primer音符
        if primer_notes:
            for note in primer_notes:
                rect = Rectangle(
                    (note['start'], note['pitch'] - 0.4),
                    note['end'] - note['start'],
                    0.8,
                    facecolor=primer_color,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(rect)
        
        # 绘制generated音符，按primer_end_time分割
        primer_end_time = primer_data.get_end_time() if primer_notes else 0
        if generated_notes:
            for note in generated_notes:
                start = note['start']
                end = note['end']
                pitch = note['pitch']

                if end <= primer_end_time:
                    # Entirely in primer range
                    rect = Rectangle(
                        (start, pitch - 0.4),
                        end - start,
                        0.8,
                        facecolor=primer_color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)
                elif start >= primer_end_time:
                    # Entirely in generated range
                    rect = Rectangle(
                        (start, pitch - 0.4),
                        end - start,
                        0.8,
                        facecolor=generated_color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)
                else:
                    # Split at primer_end_time
                    rect1 = Rectangle(
                        (start, pitch - 0.4),
                        primer_end_time - start,
                        0.8,
                        facecolor=primer_color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_patch(rect1)

                    rect2 = Rectangle(
                        (primer_end_time, pitch - 0.4),
                        end - primer_end_time,
                        0.8,
                        facecolor=generated_color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_patch(rect2)
        
        # 添加分隔线
        if primer_notes and generated_notes:
            # ax.axvline(x=primer_end_time, color='black', linestyle='--', linewidth=2, alpha=0.8)
            # ax.text(primer_end_time, ax.get_ylim()[1] * 0.95, 'Primer End', 
            #        rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')
            
            plt.rcParams.update({'font.size': 16})  # 设置全局字体大小（可调）

            ax.axvline(x=primer_end_time, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(primer_end_time, ax.get_ylim()[1] * 0.85, 'Primer End',   # 从0.95改为0.85，往下移
                    rotation=90, verticalalignment='top', fontsize=18, fontweight='bold')  # 单独加大字号
        
        # 设置坐标轴
        ax.set_ylim(0, 128)  # MIDI pitch 通常是 0-127
        ax.set_xlim(0, primer_end_time + 40)  # 给足够显示范围
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Pitch')
        # ax.set_title('Primer vs Generated Sample Comparison')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=primer_color, alpha=0.7, label='Primer'),
            Patch(facecolor=generated_color, alpha=0.7, label='Generated')          
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, "primer_vs_generated_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {output_file}")
        
        plt.close()  # 关闭图形以释放内存
        
    except Exception as e:
        print(f"生成对比图时出错: {e}")

if __name__ == "__main__":
    main()
