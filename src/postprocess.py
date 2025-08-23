import pretty_midi
from music21 import converter

def preds_to_midi(note_preds, durations, hop_length=512, sr=16000):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    frame_time = hop_length / sr
    
    n_frames, n_notes = note_preds.shape
    for note_idx in range(n_notes):
        is_note_on = False
        start_frame = 0
        for t in range(n_frames):
            if note_preds[t, note_idx] == 1 and not is_note_on:
                is_note_on = True
                start_frame = t
            elif (note_preds[t, note_idx] == 0 or t == n_frames-1) and is_note_on:
                end_frame = t
                is_note_on = False
                start_time = start_frame * frame_time
                end_time = end_frame * frame_time
                dur = durations[start_frame, note_idx]
                note = pretty_midi.Note(
                    velocity=100, pitch=note_idx + 21, start=start_time, end=end_time)
                piano.notes.append(note)
    pm.instruments.append(piano)
    return pm

def midi_to_musicxml(midi_path, xml_path):
    score = converter.parse(midi_path)
    score.write('musicxml', fp=xml_path)
