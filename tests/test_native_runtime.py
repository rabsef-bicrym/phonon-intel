"""Full-note framing and native loader failures without private model fixtures."""

import gc
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import soundfile as sf

import runtime
from test_packed import five_record


def checkpoint(directory, entries, truncate=False):
    """Write a small published-format container for a loader-boundary test."""
    header = {"format": runtime.FORMAT, "index": [{**entry, "b": len(blob)} for entry, blob in entries]}
    encoded = json.dumps(header).encode()
    body = b"".join(blob for _, blob in entries)
    if truncate:
        body = body[:-1]
    (directory / "model.fermion").write_bytes(len(encoded).to_bytes(8, "little") + encoded + body)


class NativeRuntimeTests(unittest.TestCase):
    """Cover framing and worker cleanup independently of the benchmark notes."""

    def test_entire_recording_and_final_tail_are_preserved(self):
        transcriber = runtime.Transcriber.__new__(runtime.Transcriber)
        segments = []

        def decode(audio):
            segments.append(audio.copy())
            return "word", [(" word", 0.0, 0.08)]

        transcriber.segment_decode = decode
        audio = np.full(65 * 16000 + 123, .1, np.float32)
        result = transcriber.transcribe(audio)
        np.testing.assert_array_equal(np.concatenate(segments), audio)
        self.assertEqual(result["text"], "word word word")
        self.assertEqual([word["start"] for word in result["words"]], [0, 30, 60])

    def test_empty_silent_and_invalid_input(self):
        transcriber = runtime.Transcriber.__new__(runtime.Transcriber)
        for samples in (0, 1, 16000, 65 * 16000):
            self.assertEqual(transcriber.transcribe(np.zeros(samples)), {"text": "", "words": []})
        for audio in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 5))):
            with self.assertRaises(ValueError):
                transcriber.transcribe(audio)
        with self.assertRaises(ValueError):
            transcriber.segment_decode(np.ones(480001))

    def test_stereo_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            sf.write(path, np.full((48000, 2), .25), 48000, subtype="FLOAT")
            audio = runtime.read_audio(path)
        self.assertEqual(audio.shape, (16000,))
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_allclose(audio[100:-100], .25, atol=1e-6)

    def test_failed_construction_joins_workers(self):
        entry, blob = five_record(np.ones((3, 33), np.int8), np.ones((3, 33), bool),
                                  np.full(3, .125), np.full(3, .25))
        cases = [([(entry, blob), ({**entry, "n": "other"}, blob)], True, "Truncated tensor"),
                 ([(entry, b""), ({**entry, "n": "other"}, blob)], False, "Invalid compact matrix")]
        for entries, truncate, message in cases:
            before = set(threading.enumerate())
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                checkpoint(directory, entries, truncate)
                with self.assertRaisesRegex(ValueError, message):
                    runtime.load_model(directory, 2)
            gc.collect()
            self.assertEqual(set(threading.enumerate()) - before, set())

    def test_projection_contract_rejects_missing_and_extra_records(self):
        weight = ({"n": "encoder_projector.weight", "k": "fp16", "shape": [1, 1]}, b"\0\0")
        bias = ({"n": "encoder_projector.bias", "k": "fp16", "shape": [1]}, b"\0\0")
        extra = ({"n": "unexpected", "k": "fp16", "shape": [1]}, b"\0\0")
        for entries, message in (([weight], "projection checkpoint"),
                                 ([weight, bias, extra], "projection checkpoint"),
                                 ([weight, bias], "projection shape")):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                checkpoint(directory, entries)
                with self.assertRaisesRegex(ValueError, message):
                    runtime.load_model(directory, 2)


if __name__ == "__main__":
    unittest.main()
