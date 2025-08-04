from generate import plot_primer_vs_generated

primer_path = "checkpoints/v2/rpr_1/samples/1.30_100/primer.mid"
generated_path = "checkpoints/v2/rpr_1/samples/1.30_100/rolling_contour_output.mid"
output_dir = "checkpoints/v2/rpr_1/samples/1.30_100/"

plot_primer_vs_generated(primer_path, generated_path, output_dir)

# import pretty_midi
# midi = pretty_midi.PrettyMIDI(primer_path)
# print('primer notes:', sum(len(inst.notes) for inst in midi.instruments))
# midi2 = pretty_midi.PrettyMIDI(generated_path)
# print('generated notes:', sum(len(inst.notes) for inst in midi2.instruments))
# for note in midi.instruments[0].notes:
#     print(note.start, note.end, note.pitch)