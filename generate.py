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

# 添加绘图相关的导入
import pretty_midi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


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
    primer_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=primer_path)

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

            # print("Enter contour_duration (in time units (0.1s), int between 1 and 150):")
            # try:
            #     contour_duration = int(input(">> ").strip())
            # except ValueError:
            #     print("Invalid input. Please enter an integer.")
            #     continue
            
            print("Enter contour_duration (seconds, e.g. 3.5 for 3.5s, between 1.0 and 15.0):")
            try:
                duration_sec = float(input(">> ").strip())
                if not (1.0 <= duration_sec <= 15.0):
                    print("Please enter a value between 0.0 and 15.0 seconds.")
                    continue
                contour_duration = int(round(duration_sec * 10))  # 转成0.1s单位的整数
            except ValueError:
                print("Invalid input. Please enter a number (e.g. 3.5).")
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
        generated_path = os.path.join(args.output_dir, "rolling_contour_output.mid")
        decode_midi(output_sequence, file_path=generated_path)
        print(f"Final output saved to {generated_path}")

    # 生成对比图
    plot_primer_vs_generated(primer_path, generated_path, args.output_dir)

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
                # print(note)
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
            ax.axvline(x=primer_end_time, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(primer_end_time, ax.get_ylim()[1] * 0.95, 'Primer End', 
                   rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')
        
        # 设置坐标轴
        ax.set_ylim(0, 128)  # MIDI pitch 通常是 0-127
        ax.set_xlim(0, primer_end_time + 40)  # 给足够显示范围
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Pitch')
        ax.set_title('Primer vs Generated Sample Comparison')
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
