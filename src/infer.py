import torch
import librosa
import numpy as np
from model import CnnTransformerTranscriber
import pretty_midi
from postprocess import preds_to_midi, midi_to_musicxml
import argparse
import os

def infer(audio_path, checkpoint, device='cpu', threshold=0.5, out_dir='outputs'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = CnnTransformerTranscriber()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).to(device)
    
    with torch.no_grad():
        note_logits, duration_preds = model(mel_tensor)
        note_probs = torch.sigmoid(note_logits).cpu().numpy()[0]
        durations = duration_preds.cpu().numpy()[0]
        
    # Threshold notes
    note_preds = (note_probs > threshold).astype(np.int32)
    
    # Convert preds to MIDI
    pm = preds_to_midi(note_preds, durations, hop_length=512, sr=16000)
    
    # Save MIDI and convert to MusicXML
    os.makedirs(out_dir, exist_ok=True)
    midi_path = os.path.join(out_dir, 'output.mid')
    pm.write(midi_path)
    
    xml_path = midi_path.replace('.mid', '.musicxml')
    midi_to_musicxml(midi_path, xml_path)
    print(f"Salida guardada en MIDI: {midi_path} y MusicXML: {xml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--out_dir', default='outputs')
    args = parser.parse_args()
    infer(args.audio, args.checkpoint, args.device, args.threshold, args.out_dir)
