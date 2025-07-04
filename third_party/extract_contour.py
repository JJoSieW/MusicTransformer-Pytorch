import pretty_midi
import matplotlib.pyplot as plt
from collections import defaultdict
from rdp import rdp
import numpy as np


class Note:
    def __init__(self, onset, duration, abs_pitch):
        self.onset = onset
        self.duration = duration
        self.abs_pitch = abs_pitch
    def __repr__(self):
        return f"Note(onset={self.onset}, duration={self.duration}, abs_pitch={self.abs_pitch})"
    
class MelodyContourExtractor:
    
    # merge_threshold: units of 1 second
    def __init__(self, raw_melody_sequence, smooth_keep_ratio=1/3, smooth_window=3, merge_threshold=2):
        self.raw_melody_sequence = raw_melody_sequence
        self.melody_notes = self.extract_main_melody_track()
        # self.pitches = [n.abs_pitch for n in self.melody_notes]
        if self.melody_notes is None:
            raise ValueError("No melody track found.")
        self.melody_notes.sort(key=lambda n: n.onset)
        self.smooth_keep_ratio = smooth_keep_ratio
        # self.smooth_window = smooth_window
        self.merge_threshold = merge_threshold
        self.smoothed_notes = self.smooth_melody()

    def extract_main_melody_track(self):
        """使用滑动窗口从 polyphonic 音乐中提取主旋律。"""

        window_size = 1     # 窗口大小（秒）
        hop_size = 0.5        # 滑动步长（秒）

        all_notes = self.raw_melody_sequence  # 必须有属性 raw_melody_sequence
        if not all_notes:
            return []

        # 获取整个片段的最大时间
        max_time = max(note.end for note in all_notes)

        selected_notes = []
        selected_times = set()

        cur_time = 0.0
        while cur_time < max_time:
            # 找到当前窗口中的 notes
            window_notes = [note for note in all_notes if cur_time <= note.start < cur_time + window_size]

            if window_notes:
                # 选择 pitch + velocity 加权最高的音符
                best_note = max(window_notes, key=lambda n: n.pitch + n.velocity / 10.0)
                if best_note.start not in selected_times:  # 避免重复
                    selected_notes.append(Note(
                        onset=best_note.start,
                        duration=best_note.end - best_note.start,
                        abs_pitch=best_note.pitch
                    ))
                    selected_times.add(best_note.start)

            cur_time += hop_size

        # # 可视化
        # plt.figure(figsize=(12, 6))
        # raw_onsets = [n.start for n in all_notes]
        # raw_pitches = [n.pitch for n in all_notes]
        # plt.scatter(raw_onsets, raw_pitches, c='blue', alpha=0.5, label='Raw')

        # mono_onsets = [n.onset for n in selected_notes]
        # mono_pitches = [n.abs_pitch for n in selected_notes]
        # plt.scatter(mono_onsets, mono_pitches, c='red', alpha=0.7, label='Melody')

        # plt.xlabel("Time (s)")
        # plt.ylabel("MIDI Pitch")
        # plt.title("Extracted Melody")
        # plt.legend()
        # plt.grid(True)
        # plt.xlim(0, 20)
        # plt.tight_layout()
        # plt.savefig("./zzz-figs/july4/extracted_melody_no1.png", dpi=300)
        # plt.show()
        # exit()

        return selected_notes
    
    
    def smooth_melody(self):
        smooth_keep_ratio = self.smooth_keep_ratio
        # Step 1: Prepare points for RDP: (onset, pitch)
        points = np.array([[n.onset, n.abs_pitch] for n in self.melody_notes])
        original_count = len(self.melody_notes)

        # Step 2: RDP simplification, target: len(points) > num of notes * smooth_keep_ratio
        sorted_interval = np.array([], dtype=np.int32)
        end_flag = False
        while end_flag == False and len(points) > 3 and len(points) > original_count * smooth_keep_ratio:
            if sorted_interval.size == 0:
                intervals = np.abs(np.diff(points[:, 1]))
                sorted_interval = np.array(sorted(set(intervals[intervals > 0])), dtype=np.int32)
                if sorted_interval.size == 0:
                    print('size=0 break:', sorted_interval)
                    break
                delta_sorted_interval = sorted_interval // 2
            for i in sorted_interval:
                if len(points) > 3 and len(points) > original_count * smooth_keep_ratio:
                    new_points = np.array(rdp(points.tolist(), epsilon=int(i)))
                    if len(new_points) < 3:
                        end_flag = True
                        break  # Discard this result, exit for-loop and continue the next while-loop or terminate
                    points = new_points
                else:
                    end_flag = True
                    break
            # print(points)
            if len(points) > 3 and len(points) > original_count * smooth_keep_ratio:
                sorted_interval = sorted_interval + delta_sorted_interval
            # print(sorted_interval)    
        # Step 3: Recover durations by matching onset and pitch to original notes
        smoothed_notes = []
        note_lookup = {(n.onset, n.abs_pitch): n for n in self.melody_notes}
        for onset, pitch in points:
            note = note_lookup[(onset, pitch)]
            smoothed_notes.append(note)
        # print('processed notes', smoothed_notes)
        return smoothed_notes

    def simplify_contour(self):
        notes = self.smoothed_notes
        if len(notes) < 2:
            return []
        simplified = []
        start_idx = 0
        direction = None
        for i in range(1, len(notes)):
            delta = notes[i].abs_pitch - notes[i - 1].abs_pitch
            curr_dir = 'up' if delta > 0 else 'down' if delta < 0 else 'flat'
            if direction is None:
                direction = curr_dir
            is_last = (i == len(notes) - 1)
            dir_changed = (curr_dir != direction)
            if dir_changed or is_last:
                end_idx = i 
                if dir_changed and not is_last:
                    # print(dir_changed, is_last)
                    # append the interval from start_idx to end_idx - 1
                    start_pitch = notes[start_idx].abs_pitch
                    end_pitch = notes[end_idx - 1].abs_pitch
                    interval = end_pitch - start_pitch
                    start_time = notes[start_idx].onset
                    duration = notes[end_idx - 1].onset - notes[start_idx].onset
                    simplified.append((start_time, interval, duration))
                    start_idx = i - 1
                    direction = curr_dir
                elif dir_changed and is_last:
                    # print(dir_changed, is_last)
                    # append the interval from start_idx to end_idx - 1
                    start_pitch = notes[start_idx].abs_pitch
                    end_pitch = notes[end_idx - 1].abs_pitch
                    interval = end_pitch - start_pitch
                    start_time = notes[start_idx].onset
                    duration = notes[end_idx - 1].onset - notes[start_idx].onset
                    simplified.append((start_time, interval, duration))
                    # append the interval from end_idx - 1  to end_idx
                    start_pitch = notes[end_idx - 1].abs_pitch
                    end_pitch = notes[end_idx].abs_pitch
                    interval = end_pitch - start_pitch
                    start_time = notes[end_idx - 1].onset
                    duration = notes[end_idx].onset - notes[end_idx - 1].onset
                    simplified.append((start_time, interval, duration))
                elif is_last and not dir_changed:
                    # print(dir_changed, is_last)
                    # append the interval from start_idx to end_idx
                    start_pitch = notes[start_idx].abs_pitch
                    end_pitch = notes[end_idx].abs_pitch
                    interval = end_pitch - start_pitch
                    start_time = notes[start_idx].onset
                    duration = notes[end_idx].onset - notes[start_idx].onset
                    simplified.append((start_time, interval, duration))

        return simplified

    def merge_small_intervals(self, contour):
        if not contour:
            return []
        merged = []
        # for start_time, interval, dur in contour:
        #     if merged and dur <= self.merge_threshold:
        #         prev_start_time, prev_iv, prev_dur = merged[-1]
        #         merged[-1] = (prev_start_time, prev_iv + interval, prev_dur + dur)
        for start_time, interval, dur in contour:
            if merged:
                prev_start_time, prev_iv, prev_dur = merged[-1]
                if prev_dur <= self.merge_threshold:
                    
                    merged[-1] = (prev_start_time, prev_iv + interval, prev_dur + dur)
                else:
                    merged.append((start_time, interval, dur))
            else:
                merged.append((start_time, interval, dur))
                
        # return merged    
        
        merge_same_sign = []
        for start_time, interval, dur in merged:
            if merge_same_sign:
                prev_start_time, prev_iv, prev_dur = merge_same_sign[-1]
                if (prev_iv * interval > 0) or (prev_iv == 0 and interval == 0): # same sign
                    merge_same_sign[-1] = (prev_start_time, prev_iv + interval, prev_dur + dur)
                else:
                    merge_same_sign.append((start_time, interval, dur))
            else:
                merge_same_sign.append((start_time, interval, dur))
                    

        return merge_same_sign

    def get_final_contour(self):
        contour = self.simplify_contour()
        contour = self.merge_small_intervals(contour)
        return contour

    def print_final_contour(self):
        final_contour = self.get_final_contour()
        print("Final Simplified Contour (interval, duration in tokenizer units - 0.01s):")
        for i, (start_time, iv) in enumerate(final_contour):
            print(f"{i:2d}: start_time: {start_time}, Interval: {iv:+}")
        start_times = [start_time for start_time, _ in final_contour]
        contour_intervals = [iv for _, iv in final_contour]
        print("\nCopy-paste ready:")
        print("start_times =", start_times)
        print("contour_intervals =", contour_intervals)

    def plot_all(self, contour=None, time_range=None):
        if contour is None:
            contour = self.get_final_contour()
        
        # Use original and smoothed notes for plotting
        orig_notes = self.melody_notes
        smoothed_notes = self.smoothed_notes
        
        # If a time range is specified, filter notes
        if time_range:
            start_time, end_time = time_range
            orig_notes = [n for n in orig_notes if start_time <= n.onset <= end_time]
            smoothed_notes = [n for n in smoothed_notes if start_time <= n.onset <= end_time]
            contour = [(start_time, interval, duration) for start_time, interval, duration in contour 
                      if start_time <= start_time <= end_time]
        
        # Original note times and pitches
        orig_times = [n.onset for n in orig_notes]
        orig_pitches = [n.abs_pitch for n in orig_notes]
        
        # Smoothed note times and pitches
        smoothed_times = [n.onset for n in smoothed_notes]
        smoothed_pitches = [n.abs_pitch for n in smoothed_notes]
        
        # Contour start times, intervals, durations
        contour_start_times = [start_time for start_time, _, _ in contour]
        contour_intervals = [interval for _, interval, _ in contour]
        contour_durations = [duration for _, _, duration in contour]
        
        # Absolute path (cumulative intervals)
        abs_path = [0]
        
        # zqx 画到结尾用
        # for interval in contour_intervals:
        #     abs_path.append(abs_path[-1] + interval)
        # abs_path_x = contour_start_times + [smoothed_times[-1] if smoothed_times else 0]
        
        ### zqx 画一段用
        for interval in contour_intervals[:-1]:
            abs_path.append(abs_path[-1] + interval)
        abs_path_x = contour_start_times
        
        
        
        ######################################################
        #如果要画到结尾就删了
        
        # 获取最大横坐标限制 - 以abs_path_x[-1]为标准
        max_x = abs_path_x[-1] if abs_path_x else 0
        
        # 过滤数据，只保留横坐标不超过max_x的点
        filtered_orig = [(t, p) for t, p in zip(orig_times, orig_pitches) if t <= max_x]
        filtered_smoothed = [(t, p) for t, p in zip(smoothed_times, smoothed_pitches) if t <= max_x]
        filtered_contour = [(t, i) for t, i in zip(contour_start_times, contour_intervals) if t <= max_x]
        
        # 提取过滤后的数据
        filtered_orig_times = [t for t, _ in filtered_orig]
        filtered_orig_pitches = [p for _, p in filtered_orig]
        filtered_smoothed_times = [t for t, _ in filtered_smoothed]
        filtered_smoothed_pitches = [p for _, p in filtered_smoothed]
        filtered_contour_times = [t for t, _ in filtered_contour]
        filtered_contour_intervals = [i for _, i in filtered_contour]
        
        ######################################################
        
        plt.figure(figsize=(14, 6))
        #画到结尾
        # plt.plot(orig_times, orig_pitches, label="Original MIDI Pitch", marker='o', alpha=0.7)
        # plt.plot(smoothed_times, smoothed_pitches, label="Smoothed Pitch (RDP)", marker='s', alpha=0.7)
        # plt.scatter(contour_start_times, contour_intervals, label="Contour Intervals", marker='x', s=100, linewidth=2)
        
        #画一段
        plt.plot(filtered_orig_times, filtered_orig_pitches, label="Original MIDI Pitch", marker='o', alpha=0.7)
        plt.plot(filtered_smoothed_times, filtered_smoothed_pitches, label="Smoothed Pitch (RDP)", marker='s', alpha=0.7)
        plt.scatter(filtered_contour_times, filtered_contour_intervals, label="Contour Intervals", marker='x', s=100, linewidth=2)
        plt.plot(abs_path_x, abs_path, label="Absolute Interval Path", marker='^', linewidth=2)
        
        # if time_range:
            # plt.title(f"Melody Analysis (Time Range: {time_range[0]:.1f}s - {time_range[1]:.1f}s)")
        # else:
            # plt.title("Original Melody vs. Smoothed vs. Contour vs. Absolute Path")
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Pitch / Interval")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_segment(self, start_time=0, duration=10):
        """画指定时间段的分析图"""
        end_time = start_time + duration
        self.plot_all(time_range=(start_time, end_time))

# # Usage example
# if __name__ == "__main__":
#     # midi_file = "/import/c4dm-datasets/URMP/Dataset/01_Jupiter_vn_vc/Sco_01_Jupiter_vn_vc.mid"
    
#     # melody - poly
#     raw_melody_sequence1 = [[50, 50, 57, 0], [50, 50, 69, 0], [100, 50, 60, 0], [100, 50, 72, 0], [150, 50, 62, 0], [150, 50, 74, 0], [200, 100, 64, 0], [200, 100, 76, 0], [300, 100, 60, 0], [300, 100, 72, 0], [400, 100, 62, 0], [400, 100, 74, 0], [500, 50, 62, 0], [500, 50, 74, 0], [550, 50, 64, 0], [550, 50, 76, 0], [600, 200, 65, 0], [600, 200, 77, 0], [850, 50, 57, 0], [850, 50, 69, 0], [900, 50, 60, 0], [900, 50, 72, 0], [950, 50, 62, 0], [950, 50, 74, 0], [1000, 100, 64, 0], [1000, 100, 76, 0], [1100, 100, 67, 0], [1100, 100, 79, 0], [1200, 100, 67, 0], [1200, 100, 79, 0], [1300, 50, 67, 0], [1300, 50, 79, 0], [1350, 50, 69, 0], [1350, 50, 81, 0], [1400, 200, 66, 0], [1400, 200, 78, 0]]
#     # accomp - mono+poly
#     raw_melody_sequence2 = [[0, 25, 48, 0], [25, 25, 55, 0], [50, 25, 60, 0], [50, 25, 64, 0], [75, 25, 55, 0], [125, 25, 55, 0], [150, 25, 60, 0], [150, 25, 64, 0], [175, 25, 55, 0], [200, 25, 50, 0], [225, 25, 57, 0], [250, 25, 62, 0], [250, 25, 65, 0], [275, 25, 57, 0], [325, 25, 57, 0], [350, 25, 62, 0], [350, 25, 65, 0], [375, 25, 57, 0], [400, 25, 43, 0], [425, 25, 59, 0], [450, 25, 62, 0], [450, 25, 65, 0], [475, 25, 59, 0], [525, 25, 59, 0], [550, 25, 62, 0], [550, 25, 65, 0], [575, 25, 59, 0], [600, 25, 48, 0], [625, 25, 55, 0], [650, 25, 60, 0], [650, 25, 64, 0], [675, 25, 55, 0], [725, 25, 55, 0], [750, 25, 60, 0], [750, 25, 64, 0], [775, 25, 55, 0], [800, 25, 45, 0], [825, 25, 57, 0], [850, 25, 60, 0], [850, 25, 64, 0], [875, 25, 57, 0], [925, 25, 57, 0], [950, 25, 60, 0], [950, 25, 64, 0], [975, 25, 57, 0], [1000, 25, 41, 0], [1025, 25, 57, 0], [1050, 25, 60, 0], [1050, 25, 65, 0], [1075, 25, 57, 0], [1125, 25, 57, 0], [1150, 25, 60, 0], [1150, 25, 65, 0], [1175, 25, 57, 0], [1200, 25, 43, 0], [1225, 25, 59, 0], [1250, 25, 62, 0], [1250, 25, 67, 0], [1275, 25, 59, 0], [1325, 25, 59, 0], [1350, 25, 62, 0], [1350, 25, 67, 0], [1375, 25, 59, 0], [1400, 25, 48, 0], [1425, 25, 60, 0], [1450, 25, 64, 0], [1450, 25, 67, 0], [1475, 25, 60, 0], [1525, 25, 60, 0], [1550, 25, 64, 0], [1550, 25, 67, 0], [1575, 25, 60, 0]]
#     # riff - mono
#     raw_melody_sequence3 = [[0, 12, 81, 0], [12, 12, 72, 0], [25, 12, 76, 0], [38, 12, 81, 0], [50, 12, 84, 0], [62, 13, 76, 0], [75, 12, 81, 0], [88, 12, 84, 0], [100, 12, 88, 0], [112, 12, 81, 0], [125, 12, 84, 0], [137, 13, 88, 0], [150, 13, 84, 0], [162, 13, 76, 0], [175, 13, 81, 0], [188, 13, 84, 0], [200, 13, 81, 0], [213, 13, 72, 0], [225, 13, 76, 0], [238, 12, 81, 0], [250, 13, 84, 0], [263, 13, 76, 0], [275, 13, 81, 0], [288, 13, 84, 0], [300, 13, 88, 0], [313, 13, 81, 0], [325, 13, 84, 0], [338, 13, 88, 0], [350, 13, 84, 0], [363, 13, 76, 0], [375, 13, 81, 0], [388, 13, 84, 0], [400, 13, 81, 0], [413, 13, 72, 0], [425, 13, 77, 0], [438, 13, 81, 0], [450, 13, 84, 0], [463, 13, 77, 0], [475, 13, 81, 0], [488, 12, 84, 0], [500, 12, 89, 0], [513, 12, 81, 0], [525, 12, 84, 0], [538, 12, 89, 0], [550, 12, 84, 0], [562, 12, 77, 0], [575, 12, 81, 0], [587, 12, 84, 0], [600, 12, 81, 0], [612, 12, 72, 0], [625, 12, 77, 0], [637, 12, 81, 0], [650, 12, 84, 0], [662, 12, 77, 0], [675, 12, 81, 0], [687, 12, 84, 0], [700, 12, 89, 0], [712, 12, 81, 0], [725, 12, 84, 0], [737, 12, 89, 0], [750, 12, 84, 0], [762, 12, 77, 0], [775, 12, 81, 0], [787, 12, 84, 0], [800, 12, 80, 0], [812, 12, 72, 0], [825, 12, 76, 0], [837, 12, 80, 0], [850, 12, 84, 0], [862, 12, 76, 0], [875, 12, 80, 0], [887, 12, 84, 0], [900, 12, 88, 0], [912, 12, 80, 0], [925, 12, 84, 0], [937, 12, 88, 0], [950, 12, 84, 0], [962, 12, 76, 0], [975, 12, 80, 0], [987, 12, 84, 0], [1000, 12, 80, 0], [1012, 12, 72, 0], [1025, 12, 76, 0], [1037, 12, 80, 0], [1050, 12, 84, 0], [1062, 12, 76, 0], [1075, 12, 80, 0], [1087, 12, 84, 0], [1100, 12, 88, 0], [1112, 12, 80, 0], [1125, 12, 84, 0], [1137, 12, 88, 0], [1150, 12, 84, 0], [1162, 12, 76, 0], [1175, 12, 80, 0], [1187, 12, 84, 0], [1200, 12, 80, 0], [1212, 12, 71, 0], [1225, 12, 76, 0], [1237, 12, 80, 0], [1250, 12, 83, 0], [1262, 12, 76, 0], [1275, 12, 80, 0], [1287, 12, 83, 0], [1300, 12, 88, 0], [1312, 12, 80, 0], [1325, 12, 83, 0], [1337, 12, 88, 0], [1350, 12, 83, 0], [1362, 12, 76, 0], [1375, 12, 80, 0], [1387, 12, 83, 0], [1400, 12, 80, 0], [1412, 12, 71, 0], [1425, 12, 76, 0], [1437, 12, 80, 0], [1450, 12, 83, 0], [1462, 12, 76, 0], [1475, 12, 80, 0], [1487, 12, 83, 0], [1500, 12, 88, 0], [1512, 12, 80, 0], [1525, 12, 83, 0], [1537, 12, 88, 0], [1550, 12, 83, 0], [1562, 12, 76, 0], [1575, 12, 80, 0], [1587, 12, 83, 0]]
#     # pad - poly
#     raw_melody_sequence4 = [[0, 200, 64, 0], [0, 200, 67, 0], [0, 200, 72, 0], [200, 200, 67, 0], [200, 200, 70, 0], [200, 200, 74, 0], [400, 200, 64, 0], [400, 200, 67, 0], [400, 200, 72, 0], [400, 200, 76, 0], [600, 200, 67, 0], [600, 200, 70, 0], [600, 200, 74, 0], [800, 200, 65, 0], [800, 200, 69, 0], [800, 100, 74, 0], [900, 100, 72, 0], [1000, 200, 62, 0], [1000, 200, 65, 0], [1000, 200, 70, 0], [1200, 200, 65, 0], [1200, 200, 69, 0], [1200, 200, 72, 0], [1400, 200, 62, 0], [1400, 200, 67, 0], [1400, 200, 71, 0], [1400, 200, 74, 0]]
#     # riff - commu00006
#     raw_melody_sequence5 = [[0, 12, 69, 0], [12, 12, 76, 0], [25, 12, 81, 0], [38, 12, 84, 0], [50, 12, 88, 0], [62, 12, 84, 0], [75, 12, 81, 0], [88, 12, 76, 0], [100, 12, 69, 0], [112, 12, 76, 0], [125, 12, 81, 0], [138, 12, 84, 0], [150, 12, 88, 0], [162, 12, 84, 0], [175, 12, 81, 0], [188, 12, 76, 0], [200, 12, 72, 0], [212, 12, 79, 0], [225, 12, 84, 0], [238, 12, 88, 0], [250, 12, 91, 0], [262, 12, 88, 0], [275, 12, 84, 0], [288, 14, 79, 0], [300, 12, 72, 0], [312, 12, 79, 0], [325, 12, 84, 0], [338, 12, 88, 0], [350, 12, 91, 0], [362, 12, 88, 0], [375, 12, 84, 0], [388, 14, 79, 0], [400, 12, 67, 0], [412, 11, 79, 0], [425, 12, 83, 0], [438, 12, 86, 0], [450, 12, 91, 0], [462, 12, 86, 0], [475, 12, 83, 0], [488, 12, 79, 0], [500, 12, 83, 0], [512, 12, 86, 0], [525, 12, 83, 0], [538, 12, 79, 0], [550, 12, 91, 0], [562, 12, 86, 0], [575, 12, 83, 0], [588, 12, 79, 0], [600, 12, 74, 0], [612, 12, 78, 0], [625, 12, 81, 0], [638, 12, 78, 0], [650, 12, 86, 0], [662, 12, 81, 0], [675, 12, 78, 0], [688, 12, 74, 0], [700, 12, 78, 0], [712, 12, 81, 0], [725, 12, 86, 0], [738, 12, 81, 0], [750, 12, 90, 0], [762, 12, 86, 0], [775, 12, 81, 0], [788, 12, 78, 0], [800, 12, 69, 0], [812, 12, 76, 0], [825, 12, 81, 0], [838, 12, 84, 0], [850, 12, 88, 0], [862, 12, 84, 0], [875, 12, 81, 0], [888, 12, 76, 0], [900, 12, 69, 0], [912, 12, 76, 0], [925, 12, 81, 0], [938, 12, 84, 0], [950, 12, 88, 0], [962, 12, 84, 0], [975, 12, 81, 0], [988, 12, 76, 0], [1000, 12, 72, 0], [1012, 12, 79, 0], [1025, 12, 84, 0], [1038, 12, 88, 0], [1050, 12, 91, 0], [1062, 12, 88, 0], [1075, 12, 84, 0], [1088, 14, 79, 0], [1100, 12, 72, 0], [1112, 12, 79, 0], [1125, 12, 84, 0], [1138, 12, 88, 0], [1150, 12, 91, 0], [1162, 12, 88, 0], [1175, 12, 84, 0], [1188, 14, 79, 0], [1200, 12, 67, 0], [1212, 11, 79, 0], [1225, 12, 83, 0], [1238, 12, 86, 0], [1250, 12, 91, 0], [1262, 12, 86, 0], [1275, 12, 83, 0], [1288, 12, 79, 0], [1300, 12, 83, 0], [1312, 12, 86, 0], [1325, 12, 83, 0], [1338, 12, 79, 0], [1350, 12, 91, 0], [1362, 12, 86, 0], [1375, 12, 83, 0], [1388, 12, 79, 0], [1400, 12, 74, 0], [1412, 12, 78, 0], [1425, 12, 81, 0], [1438, 12, 78, 0], [1450, 12, 86, 0], [1462, 12, 81, 0], [1475, 12, 78, 0], [1488, 12, 74, 0], [1500, 12, 78, 0], [1512, 12, 81, 0], [1525, 12, 86, 0], [1538, 12, 81, 0], [1550, 12, 90, 0], [1562, 12, 86, 0], [1575, 12, 81, 0], [1588, 12, 78, 0]]
#     # riff - commu00009
#     raw_melody_sequence6 = [[0, 11, 79, 0], [12, 11, 76, 0], [25, 11, 84, 0], [38, 11, 79, 0], [50, 11, 88, 0], [62, 11, 84, 0], [75, 11, 79, 0], [88, 11, 84, 0], [100, 11, 79, 0], [112, 11, 76, 0], [125, 11, 84, 0], [138, 11, 79, 0], [150, 11, 88, 0], [162, 11, 84, 0], [175, 11, 79, 0], [188, 11, 84, 0], [200, 11, 81, 0], [212, 11, 77, 0], [225, 11, 86, 0], [238, 11, 81, 0], [250, 11, 89, 0], [262, 11, 86, 0], [275, 11, 81, 0], [288, 11, 86, 0], [300, 11, 81, 0], [312, 11, 77, 0], [325, 11, 86, 0], [338, 11, 81, 0], [350, 11, 89, 0], [362, 11, 86, 0], [375, 11, 81, 0], [388, 11, 86, 0], [400, 11, 81, 0], [412, 11, 76, 0], [425, 11, 84, 0], [438, 11, 81, 0], [450, 11, 88, 0], [462, 11, 84, 0], [475, 11, 81, 0], [488, 11, 84, 0], [500, 11, 81, 0], [512, 11, 76, 0], [525, 11, 84, 0], [538, 11, 81, 0], [550, 11, 88, 0], [562, 11, 84, 0], [575, 11, 81, 0], [588, 11, 84, 0], [600, 11, 79, 0], [612, 11, 74, 0], [625, 11, 84, 0], [638, 11, 79, 0], [650, 11, 86, 0], [662, 11, 84, 0], [675, 11, 79, 0], [688, 11, 84, 0], [700, 11, 79, 0], [712, 11, 74, 0], [725, 11, 84, 0], [738, 11, 79, 0], [750, 11, 86, 0], [762, 11, 84, 0], [775, 11, 79, 0], [788, 11, 84, 0], [800, 11, 79, 0], [812, 11, 76, 0], [825, 11, 84, 0], [838, 11, 79, 0], [850, 11, 88, 0], [862, 11, 84, 0], [875, 11, 79, 0], [888, 11, 84, 0], [900, 11, 79, 0], [912, 11, 76, 0], [925, 11, 84, 0], [938, 11, 79, 0], [950, 11, 88, 0], [962, 11, 84, 0], [975, 11, 79, 0], [988, 11, 84, 0], [1000, 11, 81, 0], [1012, 11, 77, 0], [1025, 11, 86, 0], [1038, 11, 81, 0], [1050, 11, 89, 0], [1062, 11, 86, 0], [1075, 11, 81, 0], [1088, 11, 86, 0], [1100, 11, 81, 0], [1112, 11, 77, 0], [1125, 11, 86, 0], [1138, 11, 81, 0], [1150, 11, 89, 0], [1162, 11, 86, 0], [1175, 11, 81, 0], [1188, 11, 86, 0], [1200, 11, 81, 0], [1212, 11, 76, 0], [1225, 11, 84, 0], [1238, 11, 81, 0], [1250, 11, 88, 0], [1262, 11, 84, 0], [1275, 11, 81, 0], [1288, 11, 84, 0], [1300, 11, 81, 0], [1312, 11, 76, 0], [1325, 11, 84, 0], [1338, 11, 81, 0], [1350, 11, 88, 0], [1362, 11, 84, 0], [1375, 11, 81, 0], [1388, 11, 84, 0], [1400, 11, 79, 0], [1412, 11, 74, 0], [1425, 11, 84, 0], [1438, 11, 79, 0], [1450, 11, 86, 0], [1462, 11, 84, 0], [1475, 11, 79, 0], [1488, 11, 84, 0], [1500, 11, 79, 0], [1512, 11, 74, 0], [1525, 11, 83, 0], [1538, 11, 79, 0], [1550, 11, 86, 0], [1562, 11, 83, 0], [1575, 11, 79, 0], [1588, 11, 83, 0]]
#     # riff - commu00012
#     raw_melody_sequence7 = [[0, 12, 76, 0], [12, 12, 76, 0], [25, 12, 81, 0], [38, 12, 81, 0], [50, 12, 83, 0], [62, 13, 83, 0], [75, 12, 84, 0], [88, 12, 84, 0], [100, 12, 88, 0], [112, 12, 88, 0], [125, 12, 84, 0], [137, 13, 84, 0], [150, 13, 83, 0], [162, 13, 83, 0], [175, 13, 81, 0], [188, 13, 81, 0], [200, 13, 76, 0], [213, 13, 76, 0], [225, 13, 81, 0], [238, 12, 81, 0], [250, 13, 83, 0], [263, 13, 83, 0], [275, 13, 84, 0], [288, 13, 84, 0], [300, 13, 88, 0], [313, 13, 88, 0], [325, 13, 84, 0], [338, 13, 84, 0], [350, 13, 83, 0], [363, 13, 83, 0], [375, 13, 81, 0], [388, 13, 81, 0], [400, 13, 76, 0], [413, 13, 76, 0], [425, 13, 80, 0], [438, 13, 80, 0], [450, 13, 81, 0], [463, 13, 81, 0], [475, 13, 83, 0], [488, 12, 83, 0], [500, 12, 88, 0], [513, 12, 88, 0], [525, 12, 83, 0], [538, 12, 83, 0], [550, 12, 81, 0], [562, 12, 81, 0], [575, 12, 80, 0], [587, 12, 80, 0], [600, 12, 76, 0], [612, 12, 76, 0], [625, 12, 80, 0], [637, 12, 80, 0], [650, 12, 81, 0], [662, 12, 81, 0], [675, 12, 83, 0], [687, 12, 83, 0], [700, 12, 88, 0], [712, 12, 88, 0], [725, 12, 83, 0], [737, 12, 83, 0], [750, 12, 81, 0], [762, 12, 81, 0], [775, 12, 80, 0], [787, 12, 80, 0], [800, 12, 77, 0], [812, 12, 77, 0], [825, 12, 81, 0], [837, 12, 81, 0], [850, 12, 83, 0], [862, 12, 83, 0], [875, 12, 84, 0], [887, 12, 84, 0], [900, 12, 89, 0], [912, 12, 89, 0], [925, 12, 84, 0], [937, 12, 84, 0], [950, 12, 83, 0], [962, 12, 83, 0], [975, 12, 81, 0], [987, 12, 81, 0], [1000, 12, 77, 0], [1012, 12, 77, 0], [1025, 12, 81, 0], [1037, 12, 81, 0], [1050, 12, 83, 0], [1062, 12, 83, 0], [1075, 12, 84, 0], [1087, 12, 84, 0], [1100, 12, 89, 0], [1112, 12, 89, 0], [1125, 12, 84, 0], [1137, 12, 84, 0], [1150, 12, 83, 0], [1162, 12, 83, 0], [1175, 12, 81, 0], [1187, 12, 81, 0], [1200, 12, 80, 0], [1212, 12, 80, 0], [1225, 12, 81, 0], [1237, 12, 81, 0], [1250, 12, 83, 0], [1262, 12, 83, 0], [1275, 12, 86, 0], [1287, 12, 86, 0], [1300, 12, 92, 0], [1312, 12, 92, 0], [1325, 12, 86, 0], [1337, 12, 86, 0], [1350, 12, 83, 0], [1362, 12, 83, 0], [1375, 12, 81, 0], [1387, 12, 81, 0], [1400, 12, 80, 0], [1412, 12, 80, 0], [1425, 12, 81, 0], [1437, 12, 81, 0], [1450, 12, 83, 0], [1462, 12, 83, 0], [1475, 12, 86, 0], [1487, 12, 86, 0], [1500, 12, 92, 0], [1512, 12, 92, 0], [1525, 12, 86, 0], [1537, 12, 86, 0], [1550, 12, 83, 0], [1562, 12, 83, 0], [1575, 12, 80, 0], [1587, 12, 80, 0]]
    
#     raw_melody_sequences = [raw_melody_sequence1, raw_melody_sequence2, raw_melody_sequence3, raw_melody_sequence4, raw_melody_sequence5, raw_melody_sequence6, raw_melody_sequence7]
#     # raw_melody_sequences = [raw_melody_sequence3]
#     for idx in range(1, len(raw_melody_sequences) + 1):
#         print(idx)
#         extractor = MelodyContourExtractor(raw_melody_sequences[idx - 1])
#         extractor.plot_all()
#         plt.savefig(f'./zzz-figs/rdp_v2_{idx}.png', dpi=300, bbox_inches='tight')
#         plt.close()