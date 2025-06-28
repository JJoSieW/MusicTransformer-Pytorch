import pretty_midi
import matplotlib.pyplot as plt
from third_party.extract_contour import MelodyContourExtractor
import numpy as np

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100
RANGE_CONTOUR_INTERVAL = 201  # 0-200 (对应-100到+100)
RANGE_CONTOUR_DURATION = 20   # 1-20

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    'contour_interval': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL,
    'contour_duration': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL + RANGE_CONTOUR_INTERVAL
}


class SustainAdapter:
    def __init__(self, time, type):
        self.start =  time
        self.type = type


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {} # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        ## type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return '<[SNote] time: {} type: {}, value: {}, velocity: {}>'\
            .format(self.time, self.type, self.value, self.velocity)


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
            return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)
        range_velocity = range(RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL)
        range_contour_interval = range(RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL+RANGE_CONTOUR_INTERVAL)
        range_contour_duration = range(RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL+RANGE_CONTOUR_INTERVAL, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL+RANGE_CONTOUR_INTERVAL+RANGE_CONTOUR_DURATION)

        valid_value = int_value

        if int_value in range_note_on:
            return {'type': 'note_on', 'value': valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {'type': 'note_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return {'type': 'time_shift', 'value': valid_value}
        elif int_value in range_velocity:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return {'type': 'velocity', 'value': valid_value}
        elif int_value in range_contour_interval:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL)
            return {'type': 'contour_interval', 'value': valid_value}
        else:  # contour_duration
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL + RANGE_CONTOUR_INTERVAL)
            return {'type': 'contour_duration', 'value': valid_value}


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))
    return result_array


def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type='velocity', value=modified_velocity))
    result.append(Event(event_type=snote.type, value=snote.value))
    return result


def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == 'time_shift':
            timeline += ((event.value+1) / 100)
        elif event.type == 'velocity':
            velocity = event.value * 4
        elif event.type in ['contour_interval', 'contour_duration']:
            # 跳过轮廓事件，不转换为音符
            continue
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [Event(event_type='time_shift', value=time_interval-1)]


def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(susteins, notes):
    note_stream = []

    for sustain in susteins:
        for note_idx, note in enumerate(notes):
            if note.start < sustain.start:
                note_stream.append(note)
            elif note.start > sustain.end:
                notes = notes[note_idx:]
                sustain.transposition_notes()
                break
            else:
                sustain.add_managed_note(note)

    for sustain in susteins:
        note_stream += sustain.managed_notes

    note_stream.sort(key= lambda x: x.start)
    return note_stream


def encode_midi(file_path):
    events = []
    notes = []
    contours = []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    for inst in mid.instruments: # only one instrument - piano, so only one iteration
        inst_notes = inst.notes   # inst_notes[0] Note(start=990.871094, end=990.904948, pitch=67, velocity=105)
        
        # 处理延音踏板
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64]) 
        notes = _note_preprocess(ctrls, inst_notes)  # prprcss[0] Note(start=990.871094, end=990.904948, pitch=67, velocity=105)
        
        # extract the contour
        melody_contour_extractor = MelodyContourExtractor(notes)
        contour = melody_contour_extractor.get_final_contour()  # contour[0] = (start_time, interval)
        # melody_contour_extractor.plot_segment(start_time=0, duration=25)
        # plt.savefig('./zzz-figs/contour_first25s_v2.png', dpi=300, bbox_inches='tight')
        # plt.show()

        contours.append(contour)


    # 创建轮廓时间映射
    contour_map = {}
    if contours:
        for start_time, interval, duration in contours[0]:  
            contour_map[start_time] = (interval, duration)

    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)
    
    # 在生成事件序列时插入轮廓事件
    cur_time = 0
    cur_vel = 0
    
    for snote in dnotes:
        # 检查是否需要插入轮廓事件
        if snote.type == 'note_on':
            # 检查当前时间是否对应轮廓开始时间
            matched_start_time = None
            for start_time, (interval, duration) in contour_map.items():
                if abs(snote.time - start_time) < 0.05:  # 时间容差
                    # 映射interval到0-200范围
                    mapped_interval = max(0, min(200, interval + 100))
                    
                    # 将duration转换为与time_shift相同的单位 (0.01秒)
                    duration_in_centiseconds = int(duration * 100)  # 秒转0.01秒
                    mapped_duration = max(1, min(500, duration_in_centiseconds))
                    
                    # 插入轮廓事件
                    contour_interval_event = Event('contour_interval', mapped_interval)
                    contour_duration_event = Event('contour_duration', mapped_duration)
                    events.append(contour_interval_event)
                    events.append(contour_duration_event)
                    
                    # 记录匹配的start_time，稍后删除
                    matched_start_time = start_time
                    break
            # 删除已处理的轮廓
            if matched_start_time is not None:
                del contour_map[matched_start_time]
        
        # 添加时间间隔事件
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        # 添加音符事件
        events += _snote2events(snote=snote, prev_vel=cur_vel)
        
        cur_time = snote.time
        cur_vel = snote.velocity
    # events <Event type: velocity, value: 26>, <Event type: note_on, value: 67>, <Event type: time_shift, value: 0>, <Event type: note_off, value: 67>
    # print('events: ', events)
    
    # # 可视化轮廓
    # if contours and contours[0]:
    #     # 提取轮廓数据用于可视化
    #     contour_data = contours[0]
    #     start_times = [item[0] for item in contour_data]
    #     intervals = [item[1] for item in contour_data]
    #     durations = [item[2] for item in contour_data]
        
    #     # 只取前25秒的数据
    #     filtered_data = [(start_time, interval, duration) for start_time, interval, duration in contour_data if start_time <= 25.0]
        
    #     if filtered_data:
    #         start_times = [item[0] for item in filtered_data]
    #         intervals = [item[1] for item in filtered_data]
    #         durations = [item[2] for item in filtered_data]
            
    #         # 创建图形
    #         fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
    #         # 绘制interval
    #         ax.scatter(start_times, intervals, s=100, c='blue', alpha=0.7, label='Contour Interval')
            
    #         # 在图上标注duration信息
    #         for i, (start_time, interval, duration) in enumerate(filtered_data):
    #             ax.annotate(f'd={duration:.2f}s', 
    #                        xy=(start_time, interval), 
    #                        xytext=(5, 5), 
    #                        textcoords='offset points',
    #                        fontsize=8,
    #                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
    #         ax.set_xlabel('Start Time (seconds)')
    #         ax.set_ylabel('Interval (semitones)')
    #         # ax.set_title('Melody Contour')
    #         ax.grid(True, alpha=0.3)
    #         ax.legend()
            
    #         plt.tight_layout()
    #         plt.savefig('contour_visualization.png', dpi=300, bbox_inches='tight')
    #         plt.show()
            
    #         print(f"轮廓可视化已保存为 contour_visualization_25s.png")
    #         print(f"前25秒轮廓数量: {len(filtered_data)}")
    #         print(f"时间范围: {min(start_times):.2f}s - {max(start_times):.2f}s")
    #         print(f"Interval范围: {min(intervals)} - {max(intervals)} 半音")
    #         print(f"Duration范围: {min(durations):.2f}s - {max(durations):.2f}s")
    #     else:
    #         print("前25秒内没有找到轮廓数据")
    return [e.to_int() for e in events]

def decode_midi(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]
    # print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By Yang-Kichang")
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid


if __name__ == '__main__':
    encoded = encode_midi('data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi')
    # print('encoded: ', encoded)
    # decided = decode_midi(encoded,file_path='test.mid')

    ins = pretty_midi.PrettyMIDI('data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi')
    print('ins: ', ins)
    print('ins.instruments[0]: ', ins.instruments[0])
    for i in ins.instruments:
        print('i.control_changes: ', i.control_changes)
        print('i.notes: ', i.notes)

    # ins = pretty_midi.PrettyMIDI('data/commu/commu_midi/train/raw/commu00005.mid')
    # print('文件信息:')
    # print(f'乐器数量: {len(ins.instruments)}')
    # print(f'总时长: {ins.get_end_time():.2f}秒')

    # for i, instrument in enumerate(ins.instruments):
    #     print(f'\n乐器 {i}:')
    #     print(f'  名称: {instrument.name}')
    #     print(f'  音符数量: {len(instrument.notes)}')
    #     print(f'  控制变化数量: {len(instrument.control_changes)}')
        

    #     if len(instrument.control_changes) > 0:
    #         print('  控制变化详情:')
    #         for ctrl in instrument.control_changes[:5]:  # 只显示前5个
    #             print(f'    number={ctrl.number}, value={ctrl.value}, time={ctrl.time:.2f}')
    #     else:
    #         print('  没有控制变化')