import numpy as np
from music21 import stream, note, meter, tempo, duration as m21duration

# simplified quantization of durations to common note values relative to tempo
CANDIDATES_QUARTER = np.array([4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125])  # quarter multiples

def seconds_to_quarter_multiplier(dur_sec, tempo_bpm):
    q = 60.0 / tempo_bpm
    dur_in_quarters = dur_sec / q
    idx = np.argmin(np.abs(CANDIDATES_QUARTER - dur_in_quarters))
    return float(CANDIDATES_QUARTER[idx])

def notes_list_to_music21(notes_list, tempo_bpm, out_xml='out_quantized.musicxml'):
    """notes_list: list of (midi_pitch, start_sec, end_sec)"""
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=tempo_bpm))
    for pitch, start, end in notes_list:
        dur = end - start
        mult = seconds_to_quarter_multiplier(dur, tempo_bpm)
        q = 60.0 / tempo_bpm
        dur_quarter = mult
        dur_obj = m21duration.Duration(dur_quarter)
        n = note.Note()
        n.pitch.midi = pitch
        n.duration = dur_obj
        s.append(n)
    s.write('musicxml', fp=out_xml)
    return out_xml
