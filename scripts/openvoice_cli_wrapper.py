#!/usr/bin/env python3
"""
openvoice_cli_wrapper.py

Goal:
- Run inside the OpenVoice virtual environment (NOT your Dialects env).
- Provide a stable CLI interface for Dialects to call via subprocess.

Two modes:
1) --list-api : lists useful callables in openvoice.api (so we can verify what's available)
2) synthesis attempt: tries common synthesis entrypoints safely.
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path

def add_repo_to_syspath(openvoice_repo: str) -> None:
    repo = Path(openvoice_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"--openvoice-repo not found: {repo}")
    sys.path.insert(0, str(repo))

def list_api_symbols() -> None:
    api = importlib.import_module("openvoice.api")
    print("=== openvoice.api symbols (callables) ===")
    for name, obj in sorted(vars(api).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            print(name)

def try_synthesize(openvoice_repo: str, text: str, ref_wav: str, out_wav: str) -> None:
    """
    This is a 'best-effort' implementation because OpenVoice forks differ.

    Strategy:
    - Import openvoice.api
    - Look for a likely function name among common patterns.
    - If found, attempt calling it with standard argument names.
    - If not found, raise a helpful error explaining what to do next.
    """
    add_repo_to_syspath(openvoice_repo)

    api = importlib.import_module("openvoice.api")

    candidates = [
        "synthesize",
        "tts",
        "infer",
        "inference",
        "voice_clone",
        "clone_voice",
        "generate",
    ]

    fn = None
    for name in candidates:
        if hasattr(api, name) and callable(getattr(api, name)):
            fn = getattr(api, name)
            picked = name
            break

    if fn is None:
        # Print symbols to help you choose the real entrypoint
        available = [k for k, v in vars(api).items() if callable(v) and not k.startswith("_")]
        raise RuntimeError(
            "Couldn't find a standard synthesis function in openvoice.api.\n"
            f"Available callables in openvoice.api:\n- " + "\n- ".join(sorted(map(str, available))) + "\n\n"
            "Next step: pick the correct callable and we’ll wire it explicitly.\n"
            "Run: python scripts/openvoice_cli_wrapper.py --openvoice-repo <path> --list-api"
        )

    # We now attempt to call it.
    # Because signatures differ across forks, we inspect parameters and fill what we can.
    sig = inspect.signature(fn)
    kwargs = {}
    for param in sig.parameters.values():
        pname = param.name.lower()
        if pname in ("text", "prompt", "sentence", "input_text"):
            kwargs[param.name] = text
        elif pname in ("ref_wav", "ref_audio", "speaker_wav", "speaker_audio", "reference_wav", "reference_audio"):
            kwargs[param.name] = str(Path(ref_wav).expanduser().resolve())
        elif pname in ("out_wav", "output_wav", "out_path", "output_path", "save_path"):
            kwargs[param.name] = str(Path(out_wav).expanduser().resolve())

    # Basic safety check: ensure we at least passed the core inputs
    if not kwargs:
        raise RuntimeError(
            f"Found function openvoice.api.{picked} but couldn't map its parameters.\n"
            f"Signature is: {sig}\n"
            "Paste this signature and we’ll wire it cleanly."
        )

    result = fn(**kwargs)

    # Some implementations return bytes/array/path; we just ensure output exists if we were given an out path
    outp = Path(out_wav).expanduser().resolve()
    if outp.suffix.lower() != ".wav":
        raise ValueError(f"--out-wav should end with .wav (got: {outp})")

    if outp.exists():
        print(f"OK: wrote {outp}")
        return

    # If output wasn't written, give useful debugging info
    raise RuntimeError(
        f"Called openvoice.api.{picked} successfully but did not find output at {outp}.\n"
        f"Return value was: {type(result)} -> {result}\n"
        "Next step: we’ll adapt the wrapper to match this fork’s API exactly."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvoice-repo", required=True, help="Path to OpenVoice repo root")
    parser.add_argument("--list-api", action="store_true", help="List available callables in openvoice.api and exit")

    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--ref-wav", help="Reference speaker wav for cloning")
    parser.add_argument("--out-wav", help="Where to write the generated wav")

    args = parser.parse_args()

    add_repo_to_syspath(args.openvoice_repo)

    if args.list_api:
        list_api_symbols()
        return

    if not (args.text and args.ref_wav and args.out_wav):
        raise SystemExit("For synthesis you must provide --text, --ref-wav, and --out-wav")

    try_synthesize(args.openvoice_repo, args.text, args.ref_wav, args.out_wav)

if __name__ == "__main__":
    main()

