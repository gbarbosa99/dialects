import os
import csv
from pathlib import Path
from pydub import AudioSegment

import sys
from pathlib import Path

# Add OpenVoice to the path (assuming script is in /scripts)
openvoice_dir = Path(__file__).resolve().parent.parent / "OpenVoice"
sys.path.insert(0, str(openvoice_dir))

from inference import infer


# ------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_AUDIO_DIR = BASE_DIR / "data/processed"
METADATA_PATH = BASE_DIR / "data/embeddings/synthetic_metadata.csv"
SYNTHETIC_OUTPUT_DIR = BASE_DIR / "data/synthetic/output_wav"
SYNTHETIC_METADATA_CSV = BASE_DIR / "data/synthetic/metadata.csv"
REFERENCE_CLIP_DURATION_MS = 3000  # 3 seconds
# -------------------------------------

def extract_reference_clip(source_path, ref_path, duration_ms=REFERENCE_CLIP_DURATION_MS):
    """Extract the first N ms of audio as a reference clip"""
    audio = AudioSegment.from_wav(source_path)
    ref_clip = audio[:duration_ms]
    ref_clip.export(ref_path, format="wav")

def main():
    os.makedirs(SYNTHETIC_OUTPUT_DIR, exist_ok=True)
    metadata_out = open(SYNTHETIC_METADATA_CSV, "w")
    writer = csv.writer(metadata_out)
    writer.writerow(["audio_path", "text", "speaker_reference"])

    with open(METADATA_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = Path(row["audio_path"])
            text = row["text"]
            uid = audio_path.stem

            # Paths for reference and output
            ref_clip_path = SYNTHETIC_OUTPUT_DIR / f"{uid}_ref.wav"
            synth_out_path = SYNTHETIC_OUTPUT_DIR / f"{uid}_synth.wav"

            try:
                extract_reference_clip(audio_path, ref_clip_path)

                # Run OpenVoice inference
                infer(
                    input_text=text,
                    reference_audio=str(ref_clip_path),
                    output_path=str(synth_out_path)
                )

                writer.writerow([str(synth_out_path), text, str(ref_clip_path)])
                print(f"Generated: {synth_out_path.name}")

            except Exception as e:
                print(f"Failed to generate for {uid}: {e}")

    metadata_out.close()
    print(f"\n✅ Synthetic generation complete. Metadata saved to {SYNTHETIC_METADATA_CSV}")

if __name__ == "__main__":
    main()