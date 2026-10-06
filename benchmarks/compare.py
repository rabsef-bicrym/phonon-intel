"""Alternate fresh CLI processes and save timings/hashes, never transcript text."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Compare complete process latency using caller-supplied recordings and models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--whisper-cli", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("audio", type=Path, nargs="+")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("Repeats and threads must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve prior evidence")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), VECLIB_MAXIMUM_THREADS=str(args.threads))
    rows = []
    for repeat in range(args.repeats):
        engines = ("phonon", "whisper") if repeat % 2 == 0 else ("whisper", "phonon")
        for index, audio in enumerate(args.audio):
            for engine in engines:
                if engine == "phonon":
                    command = [sys.executable, str(ROOT / "intel_macos/runtime.py"), "--model-dir",
                               str(args.model_dir), "--threads", str(args.threads), str(audio)]
                else:
                    command = [str(args.whisper_cli.resolve()), "-m", str(args.whisper_model),
                               "-f", str(audio), "-t", str(args.threads), "-l", "en", "-nt"]
                start = time.perf_counter()
                run = subprocess.run(["/usr/bin/time", "-l", *command], env=env, capture_output=True, text=True)
                elapsed = time.perf_counter() - start
                if run.returncode:
                    raise RuntimeError(f"{engine} failed for audio-{index + 1} (exit {run.returncode}); run its CLI directly for diagnostics")
                memory = re.search(r"(\d+)\s+maximum resident set size", run.stderr)
                if memory is None:
                    raise RuntimeError("Expected macOS /usr/bin/time RSS output")
                row = {"engine": engine, "input": f"audio-{index + 1}", "repeat": repeat,
                       "process_s": elapsed, "peak_rss_bytes": int(memory[1])}
                if engine == "phonon":
                    result = json.loads(run.stdout)
                    text = result["text"]
                    row["audio_seconds"] = result["audio_seconds"]
                    row["words_sha256"] = hashlib.sha256(json.dumps(result["words"], sort_keys=True).encode()).hexdigest()
                else:
                    text = run.stdout
                row["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    for index in range(len(args.audio)):
        group = [row for row in rows if row["engine"] == "phonon" and row["input"] == f"audio-{index + 1}"]
        for key in ("text_sha256", "words_sha256"):
            if len({row[key] for row in group}) != 1:
                raise RuntimeError(f"Phonon {key} changed between repetitions for audio-{index + 1}")


if __name__ == "__main__":
    main()
