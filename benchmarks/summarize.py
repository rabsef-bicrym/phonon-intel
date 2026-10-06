"""Verify the published timing evidence without models or private recordings."""

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent / "results"


def main():
    """Print medians and require parity with the recorded Python control."""
    reference = {row["length"]: row for row in json.loads((ROOT / "results.json").read_text())
                 if row["engine"] == "python"}
    count = 0
    for path in sorted(ROOT.glob("*.json")):
        rows = json.loads(path.read_text())
        if not rows:
            raise ValueError(f"Empty evidence: {path.name}")
        for row in rows:
            if row["engine"] == "whisper":
                continue
            for key in ("text_sha256", "words_sha256"):
                if row[key] != reference[row["length"]][key]:
                    raise ValueError(f"Output mismatch: {path.name}, {key}")
            if row["engine"] != "python" and row["python_torch_imported"]:
                raise ValueError("Native path imported Python torch")
            count += 1
        for length, engine in sorted({(row["length"], row["engine"]) for row in rows}):
            group = [row["process_s"] for row in rows if (row["length"], row["engine"]) == (length, engine)]
            print(f"{path.name}: {length} {engine}: n={len(group)}, median={median(group):.3f}s, range={min(group):.3f}-{max(group):.3f}s")
    print(f"Verified text and word-timestamp hashes for {count} Phonon runs.")


if __name__ == "__main__":
    main()
