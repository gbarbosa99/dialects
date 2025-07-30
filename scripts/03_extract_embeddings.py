import os
import csv
import whisper
import difflib
import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # go up from /scripts

AUDIO_DIR = BASE_DIR / "data/processed"
TXT_DIR = BASE_DIR / "data/raw"
PASSAGE_DIR = BASE_DIR / "passages"
OUTPUT_MATCHES = BASE_DIR / "matched_passages.csv"
OUTPUT_TRANSCRIPTS = BASE_DIR / "full_transcripts.csv"
OUTPUT_METADATA = BASE_DIR / "synthetic_metadata.csv"
# ----------------------------


def load_passages(passage_dir):
    passages = {}
    for file in os.listdir(passage_dir):
        if file.endswith(".txt"):
            path = os.path.join(passage_dir, file)
            passages[file] = open(path).read().strip().lower()
    return passages

def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path)
    return result["text"].strip().lower()

def classify_passage(transcribed, passages):
    best_name = None
    best_score = 0
    for name, text in passages.items():
        score = difflib.SequenceMatcher(None, transcribed, text).ratio()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

def main():
    model = whisper.load_model("base")
    passages = load_passages(PASSAGE_DIR)

    matches = []
    transcripts = []

    for file in os.listdir(AUDIO_DIR):
        if not file.endswith(".wav"):
            continue

        wav_path = os.path.join(AUDIO_DIR, file)
        txt_path = os.path.join(TXT_DIR, file.replace(".wav", ".txt"))

        if not os.path.exists(txt_path):
            print(f"Missing text for {file}")
            continue

        try:
            # Step 1: Transcribe with Whisper
            transcribed = transcribe_audio(wav_path, model)

            # Step 2: Classify which passage
            match_name, score = classify_passage(transcribed, passages)
            matches.append([file, match_name, score])

            # Step 3: Build full transcript
            tail = open(txt_path).read().strip()
            passage_text = passages[match_name]
            full_text = f"{passage_text} {tail}"
            transcripts.append([file, full_text])

        except Exception as e:
            print(f"Failed on {file}: {e}")

    # Save matched passages
    with open(OUTPUT_MATCHES, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "matched_passage", "similarity"])
        writer.writerows(matches)

    # Save full transcripts
    with open(OUTPUT_TRANSCRIPTS, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text"])
        writer.writerows(transcripts)

    # Save metadata for synthetic generation
    with open(OUTPUT_METADATA, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "text"])
        for fname, text in transcripts:
            full_audio_path = os.path.join(AUDIO_DIR, fname)
            writer.writerow([full_audio_path, text])

    print(f"Done! Saved: {OUTPUT_MATCHES}, {OUTPUT_TRANSCRIPTS}, {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()
