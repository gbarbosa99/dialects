import os
import sys
import shutil
import torch
import torchaudio
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf

# Load Silero VAD
def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    get_speech_timestamps, _, _, read_audio, _ = utils
    return model, get_speech_timestamps

model, get_speech_timestamps = load_silero_vad()

def audiosegment_to_waveform(segment, target_sr=16000):
    buffer = BytesIO()
    segment.export(buffer, format='wav')
    buffer.seek(0)

    waveform, sr = sf.read(buffer, dtype='float32')  # returns numpy array
    waveform = torch.from_numpy(waveform)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)  # downmix to mono if stereo

    return waveform, sr



def detect_speech_start(audio_segment, initial_skip_ms=13000, max_skip_ms=20000, step_ms=2000):
    """
    Skip the first `initial_skip_ms` milliseconds (narration),
    then incrementally scan for real speech using Silero VAD.
    """
    skip_ms = initial_skip_ms

    while skip_ms <= max_skip_ms:
        post_narration = audio_segment[skip_ms:]
        waveform, sr = audiosegment_to_waveform(post_narration)

        segments = get_speech_timestamps(waveform, model, sampling_rate=sr)
        if segments:
            # Convert first segment start time to ms
            relative_start_ms = segments[0]['start'] / sr * 1000
            absolute_start_ms = skip_ms + relative_start_ms

            # If the VAD detects something at least 1 second after skip, accept it
            if relative_start_ms > 1000:
                print(f"Accepted speech start at {absolute_start_ms:.0f}ms (skip: {skip_ms}ms)")
                return absolute_start_ms
            else:
                print(f"Ignored early detection at {relative_start_ms:.0f}ms after skip of {skip_ms}ms")

        skip_ms += step_ms

    raise ValueError("Could not find valid speech start after skipping narration")



def preprocess(input_dir, output_dir, corrupted_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(corrupted_dir, exist_ok=True)

    for _, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".mp3"):
                continue

            file_path = os.path.join(input_dir, file)
            output_filename = file.replace(".mp3", ".wav")
            output_path = os.path.join(output_dir, output_filename)

            try:
                audio = AudioSegment.from_file(file_path)
                start_ms = detect_speech_start(audio)
                trimmed_audio = audio[int(start_ms):]
                trimmed_audio.export(output_path, format="wav")
                print(f"Trimmed and saved: {output_filename}")

            except Exception as e:
                print(f"Failed to process {file}: {e}")
                shutil.move(file_path, os.path.join(corrupted_dir, file))

                txt_file = file.replace(".mp3", ".txt")
                txt_path = os.path.join(input_dir, txt_file)
                if os.path.exists(txt_path):
                    shutil.move(txt_path, os.path.join(corrupted_dir, txt_file))

def main():
    if len(sys.argv) < 3:
        print("Usage: python 02_preprocess_audio.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    corrupted_dir = os.path.join(input_dir, "corrupted")

    preprocess(input_dir, output_dir, corrupted_dir)

if __name__ == "__main__":
    main()
