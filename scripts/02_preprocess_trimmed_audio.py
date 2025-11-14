import os
import sys
import shutil
import torch
import torchaudio
from pydub import AudioSegment
from io import BytesIO
from pathlib import Path
import soundfile as sf
import assemblyai as aai 
import csv
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

# -------CONFIG---------
BASE_DIR = Path(__file__).resolve().parent.parent # go up from /scripts

AUDIO_DIR = BASE_DIR / "data/raw"
TXT_DIR = BASE_DIR / "data/raw"

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_TRANSCRIPTS = OUTPUT_DIR / "full_transcripts.csv"
# ----------------------

# Not currently using torch to convert audio to waveform. This can be done if needed for the model we are using.

# Step 1: Separate audio into speakers and add timestamps to separate narrator from speech
def transcribe_and_timestamp_audio(file_path, file_name):
    #Convert to wav first
    audio = AudioSegment.from_file(file_path)

    with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio.export(tmp.name, format='wav')
        tmp_path = tmp.name

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(tmp_path, config)

    #first_utterance = True
    for utterance in transcript.utterances:
        if utterance.speaker == "B":
            return file_name, utterance.start, utterance.text
    
    # No speaker B found
    print(f"[WARN] No speaker B found in {file_name}")
    return file_name, None, None


# Step 2: Trim narrator speech from audio using Pydub. Save audio as Wav in output folder.
def trim_audio(start_time, file_name, file_path, output_dir):
    output_filename = file_name.replace(".mp3", ".wav")
    output_path = os.path.join(output_dir, output_filename)

    audio = AudioSegment.from_file(file_path)
    trimmed_audio = audio[int(start_time):]
    trimmed_audio.export(output_path, format='wav')

    print(f'Trimmed and saved: {output_path}')

# Step 3: Should we save the transcript from the python library or from the website? 
# We could just save the transcription from first function to a separate csv for each recording?
# We should check if the transcript from the library matches the transript from the website since these speakers have heavy accents.

def main():
    load_dotenv()
    api_key = os.getenv("MY_API_KEY")

    if api_key:
        print("API Key retrieved successfully.")
    else:
        print("API Key not found in environment variables.")

    aai.settings.api_key = api_key
    transcripts = []

    max_files = 1
    processed = 0

    for file_name in os.listdir(AUDIO_DIR):
        if not file_name.endswith(".mp3"):
            continue
        
        if processed >= max_files:
            break

        try:
            file_path = os.path.join(AUDIO_DIR, file_name)

            result = transcribe_and_timestamp_audio(file_path, file_name)
            file_name, start_time, transcription = result

            if start_time is None:
                print(f"[INFO] Skipping trimming for {file_name} because no speaker B was found.")
                continue

            trim_audio(start_time, file_name, file_path, OUTPUT_DIR) 
            transcripts.append([file_name, start_time, transcription])

            processed += 1

        except Exception as e:
            print(f'Failed to transcribe {file_path}: {e}')
                    

    with open(OUTPUT_TRANSCRIPTS, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Start Time", "Words"])
        writer.writerows(transcripts)

if __name__ == "__main__":
    main()