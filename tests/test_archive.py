"""Verify release extraction and preservation of an existing installation."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import zstandard
import unpack_model


class ArchiveTests(unittest.TestCase):
    """Exercise the archive verification boundary using a tiny synthetic release."""

    def test_verified_archive_and_existing_install(self):
        payload = b"{}"
        manifest = {"files": [{"path": "config.json", "original_bytes": 2,
                               "original_sha256": hashlib.sha256(payload).hexdigest()}]}
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as tar:
            for name, data in (("bps_manifest.json", json.dumps(manifest).encode()), ("config.json", payload)):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        archive_bytes = zstandard.ZstdCompressor().compress(raw.getvalue())
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive.zst"
            archive.write_bytes(archive_bytes)
            destination = Path(temporary) / "model"
            with patch.object(unpack_model, "ARCHIVE_SHA256", hashlib.sha256(archive_bytes).hexdigest()):
                unpack_model.unpack(archive, destination)
                self.assertEqual((destination / "config.json").read_bytes(), payload)
                with self.assertRaises(FileExistsError):
                    unpack_model.unpack(archive, destination)
            with self.assertRaisesRegex(ValueError, "verified"):
                unpack_model.unpack(archive, Path(temporary) / "bad")
            self.assertFalse((Path(temporary) / "bad").exists())


if __name__ == "__main__":
    unittest.main()
