#!/usr/bin/env python3
import os
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchaudio

# SpeechBrain (v1+)
from speechbrain.inference import EncoderClassifier


# ----------------------
# CONFIG
# ----------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_AUDIO_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
FAILED_DIR = BASE_DIR / "data" / "failed"
CORRUPTED_DIR = BASE_DIR / "data" / "corrupted"

EMBEDDINGS_INDEX_CSV = EMBEDDINGS_DIR / "embeddings_index.csv"
EMBEDDING_FAILURES_CSV = FAILED_DIR / "embedding_failures.csv"

# Optional: enrich the index with metadata from your scraper output (if present)
DIALECTS_METADATA_JSON = BASE_DIR / "data" / "raw" / "dialects_metadata.json"

# Model info
HF_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_CACHE_DIR = BASE_DIR / "pretrained_models" / "ecapa_voxceleb"

# Audio normalization
TARGET_SAMPLE_RATE = 16000
RESAMPLE_IF_NEEDED = False  # You said you already resampled during scraping; set True if you want safety.
FORCE_MONO = True

# Device
DEFAULT_DEVICE = "cpu"  # On Mac, keep CPU. If on GPU box later, set "cuda".


# ----------------------
# Utilities
# ----------------------
def ensure_dirs():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)


def load_optional_metadata(metadata_path: Path) -> dict:
    """
    Returns a mapping keyed by local audio filename stem if possible.
    Example: "asia_russia_22" -> {...metadata...}
    """
    if not metadata_path.exists():
        return {}

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    by_stem = {}
    for item in data if isinstance(data, list) else []:
        # Try to align using local_audio_path when available
        local_path = item.get("local_audio_path", "")
        stem = Path(local_path).stem if local_path else None
        if not stem:
            # fallback: try to derive from continent/country/speaker format
            # but your filenames vary; keep it conservative
            continue
        by_stem[stem] = item
    return by_stem


def load_model(device: str = DEFAULT_DEVICE) -> EncoderClassifier:
    """
    Loads ECAPA-TDNN speaker embedding model once.
    """
    classifier = EncoderClassifier.from_hparams(
        source=HF_MODEL_ID,
        savedir=str(MODEL_CACHE_DIR),
        run_opts={"device": device},
    )
    classifier.eval()
    return classifier


def load_audio(wav_path: Path, device: str = DEFAULT_DEVICE):
    """
    Loads audio with torchaudio and standardizes:
      - float32 tensor
      - mono if FORCE_MONO
      - optional resampling to TARGET_SAMPLE_RATE
    Returns: (waveform, sample_rate)
      waveform shape typically: [channels, time]  (SpeechBrain accepts this)
    """
    waveform, sr = torchaudio.load(str(wav_path))  # waveform: [C, T]

    # Convert to float32
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    # Force mono
    if FORCE_MONO and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Optional resample safeguard
    if RESAMPLE_IF_NEEDED and sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = TARGET_SAMPLE_RATE

    # Move to device (CPU by default)
    waveform = waveform.to(device)

    return waveform, sr


def extract_embedding(model: EncoderClassifier, waveform: torch.Tensor) -> np.ndarray:
    """
    Runs ECAPA inference and returns a 1D NumPy embedding vector.
    Typical model output: [B, N, D] -> squeeze -> [D]
    """
    with torch.no_grad():
        emb = model.encode_batch(waveform)  # often [1, 1, 192] for single waveform

    emb = emb.squeeze().detach().cpu().numpy()  # -> (192,)
    # Ensure 1D
    emb = np.asarray(emb).reshape(-1)
    return emb


def append_csv_row(csv_path: Path, header: list, row: list):
    """
    Appends a single row. Creates file + header if missing.
    """
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def safe_move_to_corrupted(src_path: Path, reason: str):
    dst_path = CORRUPTED_DIR / src_path.name
    try:
        shutil.move(str(src_path), str(dst_path))
        return str(dst_path), reason
    except Exception as e:
        return "", f"{reason} | move_failed: {e}"


# ----------------------
# Main
# ----------------------
def main():
    ensure_dirs()

    # Optional metadata mapping (for index enrichment)
    meta_by_stem = load_optional_metadata(DIALECTS_METADATA_JSON)

    device = DEFAULT_DEVICE
    model = load_model(device=device)

    # Find WAV files (recursive)
    wav_files = sorted(PROCESSED_AUDIO_DIR.rglob("*.wav"))
    if not wav_files:
        print(f"[WARN] No .wav files found under: {PROCESSED_AUDIO_DIR}")
        return

    index_header = [
        "audio_path",
        "audio_filename",
        "audio_stem",
        "embedding_path",
        "embedding_dim",
        "sample_rate",
        "duration_sec",
        "device",
        "created_utc",
        # Optional metadata fields (may be blank)
        "continent",
        "country",
        "speaker",
        "audio_url",
    ]

    failures_header = [
        "audio_path",
        "audio_filename",
        "reason",
        "created_utc",
    ]

    created_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for wav_path in wav_files:
        stem = wav_path.stem
        embedding_path = EMBEDDINGS_DIR / f"{stem}.npy"

        # Skip if already embedded
        if embedding_path.exists():
            skipped_count += 1
            continue

        try:
            waveform, sr = load_audio(wav_path, device=device)

            # Duration estimate
            # waveform: [C, T]
            num_samples = waveform.shape[-1]
            duration_sec = float(num_samples) / float(sr) if sr else 0.0

            emb = extract_embedding(model, waveform)
            np.save(str(embedding_path), emb)

            # Optional metadata enrichment
            meta = meta_by_stem.get(stem, {})
            continent = meta.get("continent", "")
            country = meta.get("country", "")
            speaker = meta.get("speaker", "")
            audio_url = meta.get("audio_url", "")

            append_csv_row(
                EMBEDDINGS_INDEX_CSV,
                index_header,
                [
                    str(wav_path),
                    wav_path.name,
                    stem,
                    str(embedding_path),
                    int(emb.shape[0]),
                    int(sr),
                    f"{duration_sec:.3f}",
                    device,
                    created_utc,
                    continent,
                    country,
                    speaker,
                    audio_url,
                ],
            )

            processed_count += 1
            if processed_count % 25 == 0:
                print(f"[INFO] Embedded {processed_count} files...")

        except Exception as e:
            failed_count += 1

            # If it looks like an audio decode/load issue, move to corrupted
            reason = str(e)
            moved_path = ""
            if "Error opening input" in reason or "Decoding failed" in reason or "ffmpeg" in reason:
                moved_path, reason = safe_move_to_corrupted(wav_path, reason)

            append_csv_row(
                EMBEDDING_FAILURES_CSV,
                failures_header,
                [str(wav_path), wav_path.name, reason, created_utc],
            )

            if moved_path:
                print(f"[ERROR] Corrupted audio moved: {wav_path.name} -> {moved_path}")
            else:
                print(f"[ERROR] Failed: {wav_path.name} | {reason}")

    print("\nDone.")
    print(f"  Embedded: {processed_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nIndex: {EMBEDDINGS_INDEX_CSV}")
    print(f"Failures: {EMBEDDING_FAILURES_CSV}")


if __name__ == "__main__":
    main()
