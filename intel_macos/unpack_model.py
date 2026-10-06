"""Verify and unpack the published Phonon-2 archive without an external zstd CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_release_bps import join_file

ARCHIVE_SHA256 = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"


def unpack(archive: Path, destination: Path):
    """Verify archive and member hashes; refuse to overwrite an existing install."""
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Not the verified Phonon-2 release archive")
    destination.mkdir(parents=True, exist_ok=False)
    with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            members = iter(tar)
            manifest = next(members, None)
            if manifest is None or manifest.name != "bps_manifest.json":
                raise ValueError("Missing byte-plane manifest")
            index = {entry["path"]: entry for entry in json.load(tar.extractfile(manifest))["files"]}
            for member in members:
                if not member.isfile():
                    raise ValueError("Expected regular file")
                name = member.name.removesuffix(".bps")
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive path")
                entry = index.pop(name)
                data = tar.extractfile(member).read()
                if "transform" in entry:
                    data = join_file(data, entry["transform"])
                if len(data) != entry["original_bytes"] or hashlib.sha256(data).hexdigest() != entry["original_sha256"]:
                    raise ValueError(f"Checksum mismatch: {name}")
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if index:
                raise ValueError(f"Missing archive members: {sorted(index)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    unpack(args.archive, args.destination)
