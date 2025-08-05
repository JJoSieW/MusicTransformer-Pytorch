import os
import json
import torch
from generate_single import generate_single  # 使用独立的 generate_single 函数
from utilities.device import get_device
from argparse import Namespace


# ------------------------------
# 参数配置
# ------------------------------
OUTPUT_DIR = "final_results/generate_music/v4/contour_conditioned"
MODEL_WEIGHTS = "checkpoints/v2/rpr_1/results/best_acc_weights.pickle"
# OUTPUT_DIR = "final_results/generate_music/v3/ablation"
# MODEL_WEIGHTS = "checkpoints/v2/rpr_1_ablation/results/best_acc_weights.pickle"
MIDI_ROOT = "data/preprocessed/full"

# Load contour groups from JSON file
with open('contour_groups.json', 'r') as f:
    contour_groups = json.load(f)

# ------------------------------
# 创建输出目录
# ------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 批量生成
# ------------------------------
for group_idx, group in enumerate(contour_groups):
    for repeat_idx in range(5):
        out_dir = os.path.join(OUTPUT_DIR, f"sample_{group['group_id']}_{repeat_idx+1}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Generating sample {group['group_id']}_{repeat_idx+1} | duration={group['duration']}s | interval={group['interval']}st ===")
        
        try:
            args = Namespace(
                midi_root=MIDI_ROOT,
                output_dir=out_dir,
                model_weights=MODEL_WEIGHTS,
                primer_file=str(group["primer_file"]),
                d_model=512,
                num_heads=8,
                n_layers=8,
                max_sequence=768,
                target_seq_length=512,
                rpr=True,
                contour_interval=group["interval"],
                contour_duration=group["duration"],
                num_prime=256,
                dim_feedforward=1024,
                beam=0
            )
            generate_single(args)
        except Exception as e:
            print(f"⚠️ Sample {group['group_id']}_{repeat_idx+1} failed: {e}")
