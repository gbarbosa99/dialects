import os
import sys
import csv
import torch
from pydub import AudioSegment
from pathlib import Path

# Add parent directory (one level up from scripts/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Now you can import modules from the local openvoice repo
#import se_extractor
#from api import BaseSpeakerTTS, ToneColorConverter

# --------- CONFIG ---------
AUDIO_INPUT = BASE_DIR / "data/processed"
METADATA_IN = BASE_DIR / "data/embeddings/synthetic_metadata.csv"
SYNTHETIC_DIR = BASE_DIR / "data/synthetic"
OUTPUT_WAV_DIR = SYNTHETIC_DIR / "output_wav"
OUTPUT_CSV = SYNTHETIC_DIR / "metadata.csv"
CHECKPOINT_BASE = BASE_DIR / "checkpoints/base_speakers/EN"
CHECKPOINT_CONVERTER = BASE_DIR / "checkpoints/converter"
REFERENCE_CLIP_MS = 3000
ENCODE_MESSAGE = "@MyShell"
# --------------------------

def init_models(device):
    base_tts = BaseSpeakerTTS(f"{CHECKPOINT_BASE}/config.json", device=device)
    base_tts.load_ckpt(f"{CHECKPOINT_BASE}/checkpoint.pth")

    tone_converter = ToneColorConverter(f"{CHECKPOINT_CONVERTER}/config.json", device=device)
    tone_converter.load_ckpt(f"{CHECKPOINT_CONVERTER}/checkpoint.pth")

    source_se = torch.load(f"{CHECKPOINT_BASE}/en_default_se.pth").to(device)
    return base_tts, tone_converter, source_se

def extract_reference_clip(audio_path, output_path, duration_ms=REFERENCE_CLIP_MS):
    audio = AudioSegment.from_wav(audio_path)
    ref_clip = audio[:duration_ms]
    ref_clip.export(output_path, format="wav")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)

    base_tts, tone_converter, source_se = init_models(device)

    with open(METADATA_IN, newline='') as f_in, open(OUTPUT_CSV, "w", newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["audio_path", "text", "reference_clip"])

        for row in reader:
            try:
                full_path = Path(row["audio_path"])
                text = row["text"]
                uid = full_path.stem

                # File paths
                ref_clip_path = OUTPUT_WAV_DIR / f"{uid}_ref.wav"
                tts_output_path = OUTPUT_WAV_DIR / f"{uid}_tmp.wav"
                final_output_path = OUTPUT_WAV_DIR / f"{uid}_synth.wav"

                # Extract speaker reference clip
                extract_reference_clip(full_path, ref_clip_path)

                # Get target tone color embedding
                target_se, _ = se_extractor.get_se(
                    reference_audio_path=str(ref_clip_path),
                    tone_color_converter=tone_converter,
                    target_dir=str(OUTPUT_WAV_DIR),
                    vad=True
                )

                # Step 1: Base speaker TTS
                base_tts.tts(
                    text=text,
                    speaker='default',
                    language='English',
                    output_path=str(tts_output_path),
                    speed=1.0
                )

                # Step 2: Convert tone to match reference speaker
                tone_converter.convert(
                    audio_src_path=str(tts_output_path),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(final_output_path),
                    message=ENCODE_MESSAGE
                )

                writer.writerow([str(final_output_path), text, str(ref_clip_path)])
                print(f"Generated: {final_output_path.name}")

            except Exception as e:
                print(f"Failed to process {row['audio_path']}: {e}")

    print(f"\nDone! Synthetic files saved to: {OUTPUT_WAV_DIR}")
    print(f"Metadata saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()