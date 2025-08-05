import os
import json
import muspy
import numpy as np
import pretty_midi
import pandas as pd

def compute_muspy_metrics(midi_path):
    music = muspy.read_midi(midi_path)
    
    ## compute measure_resolution
    
    if not music.time_signatures:
        # 默认4/4拍
        numerator, denominator = 4, 4
    else:
        # 取出现次数最多的时间签名
        from collections import Counter
        ts_counter = Counter((ts.numerator, ts.denominator) for ts in music.time_signatures)
        numerator, denominator = ts_counter.most_common(1)[0][0]

    quarter_note_resolution = music.resolution
    
    # 使用通用公式计算 measure_resolution
    measure_resolution = quarter_note_resolution * (4 * numerator / denominator)
    measure_resolution = int(round(measure_resolution))
    
    metrics = {
        "n_pitch_classes_used": muspy.metrics.n_pitch_classes_used(music),
        "n_pitches_used": muspy.metrics.n_pitches_used(music),
        "pitch_entropy": muspy.metrics.pitch_entropy(music),
        "pitch_range": muspy.metrics.pitch_range(music),
        "polyphony_rate": muspy.metrics.polyphony_rate(music),
        "empty_measure_rate": muspy.metrics.empty_measure_rate(music, measure_resolution),
        "groove_consistency": muspy.metrics.groove_consistency(music, measure_resolution)
    }
    return metrics

def compute_contour_metrics(segment_path, contour_json_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(segment_path)
        notes = [n for inst in midi_data.instruments for n in inst.notes]
        notes = sorted(notes, key=lambda x: x.start)
        
        
        # Extract monophonic melody using sliding window
        window_size = 1.0
        hop_size = 0.5
        
        max_time = max(note.end for note in notes)
        selected_notes = []
        selected_times = set()
        cur_time = 0.0
        while cur_time < max_time:
            window_notes = [note for note in notes if cur_time <= note.start < cur_time + window_size]
            if window_notes:
                best_note = max(window_notes, key=lambda n: n.pitch + n.velocity / 10.0)
                if best_note.start not in selected_times:
                    selected_notes.append(best_note)
                    selected_times.add(best_note.start)
            cur_time += hop_size
        notes = sorted(selected_notes, key=lambda x: x.start)
        
        
        if len(notes) < 2:
            return {"final_interval_err": np.nan, "directional_acc": np.nan, "rmse_path_tol": np.nan, "success": 0}

        with open(contour_json_path) as f:
            contour = json.load(f)
        target_interval = contour["interval"]
        target_duration = contour["duration"]

        # Use primer_end as the start of notes
        primer_end = notes[0].start
        
        # Filter notes within [primer_end, primer_end + target_duration]
        filtered_notes = [n for n in notes if primer_end <= n.start <= primer_end + target_duration]
        
        times_notes = np.array([n.start for n in filtered_notes])
        pitches_notes = np.array([n.pitch for n in filtered_notes])
        
        if len(times_notes) > 1:
            slope, intercept = np.polyfit(times_notes, pitches_notes, 1)
        else:
            slope = 0.0

        target_slope = target_interval / target_duration if target_duration != 0 else 0
        
        if target_slope != 0:
            slope_ratio = slope / target_slope
        else:
            slope_ratio = 0

        # Use first and last note for interval-based metrics
        start_pitch = notes[0].pitch
        end_pitch = notes[-1].pitch
        actual_interval = end_pitch - start_pitch
        final_interval_err = abs(actual_interval - target_interval)

        # Directional accuracy
        dir_target = np.sign(target_interval)
        dir_actual = np.sign(actual_interval)
        directional_acc = 1 if dir_target == dir_actual or (abs(target_interval) <= 2 and abs(actual_interval) <= 2) else 0

        # RMSE path with tolerance δ = 3 semitones
        delta = 3
        times = np.linspace(notes[0].start, notes[-1].end, 10)
        pitches = []
        for t in times:
            active_notes = [n.pitch for n in notes if n.start <= t <= n.end]
            if active_notes:
                pitches.append(np.mean(active_notes))
            else:
                prev_note = max([n for n in notes if n.start <= t], key=lambda x: x.start, default=notes[0])
                pitches.append(prev_note.pitch)

        ideal_line = np.linspace(start_pitch, start_pitch + target_interval, 10)
        diffs = np.abs(np.array(pitches) - ideal_line)
        diffs_tol = np.maximum(0, diffs - delta)
        rmse_path_tol = np.sqrt(np.mean(diffs_tol ** 2))

        success = 1 if (slope_ratio > 0 and final_interval_err <= 100) else 0 # and directional_acc == 1

        return {
            # "final_interval_err": final_interval_err,
            # "directional_acc": directional_acc,
            "rmse_path_tol": rmse_path_tol,
            "slope_ratio": slope_ratio,
            "success": success
        }
    except Exception as e:
        print(f"Error in contour metrics for {segment_path}: {e}")
        return {"final_interval_err": np.nan, "directional_acc": np.nan, "rmse_path_tol": np.nan, "slope_ratio": np.nan, "success": 0}

def evaluate_all(root_dir, output_csv="final_results/metrics_generated/v1/con_evaluation_results.csv"):
    results = []
    for sample in sorted(os.listdir(root_dir)):
        sample_dir = os.path.join(root_dir, sample)
        music_file = os.path.join(sample_dir, "music.mid")
        segment_file = os.path.join(sample_dir, "condition_segment.mid")
        contour_file = os.path.join(sample_dir, "contour.json")

        if not os.path.exists(music_file) or not os.path.exists(segment_file) or not os.path.exists(contour_file):
            continue

        row = {"sample": sample}
        row.update(compute_muspy_metrics(music_file))
        row.update(compute_contour_metrics(segment_file, contour_file))
        with open(contour_file) as f:
            contour = json.load(f)
        row.update({"interval_target": contour["interval"], "duration_target": contour["duration"]})
        results.append(row)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Evaluation completed. Results saved to {output_csv}")
    
    # 计算均值并输出到txt
    mean_results = df.mean(numeric_only=True)
    txt_output = output_csv.replace(".csv", "_mean.txt")
    with open(txt_output, "w") as f:
        f.write("Mean Metrics:\n")
        for metric, value in mean_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Mean metrics saved to {txt_output}")

def evaluate_muspy_only(root_dir, output_csv="final_results/metrics_generated/muspy_only_evaluation.csv"):
    results = []
    for file_name in sorted(os.listdir(root_dir)):
        if not file_name.endswith(".midi"):
            continue
        midi_path = os.path.join(root_dir, file_name)
        if not os.path.isfile(midi_path):
            continue

        row = {"file": file_name}
        row.update(compute_muspy_metrics(midi_path))
        results.append(row)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Muspy-only evaluation completed. Results saved to {output_csv}")
    
    mean_results = df.mean(numeric_only=True)
    txt_output = output_csv.replace(".csv", "_mean.txt")
    with open(txt_output, "w") as f:
        f.write("Mean Muspy Metrics:\n")
        for metric, value in mean_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Mean muspy metrics saved to {txt_output}")

if __name__ == "__main__":
    evaluate_all("final_results/generate_music/v3/contour_conditioned","final_results/metrics_generated/new_fixed_files/v4/con_evaluation_results.csv")
    evaluate_all("final_results/generate_music/v3/ablation","final_results/metrics_generated/new_fixed_files/v4/ablation_evaluation_results.csv")
    # evaluate_muspy_only("data/maestro-v2-extraction_eval")
    
