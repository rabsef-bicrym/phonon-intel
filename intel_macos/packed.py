"""Native compact-weight matrices; ctypes owns handles for their full lifetime."""

import ctypes as C
import os
from pathlib import Path

import numpy as np


def library(threads=2):
    """Load the explicitly built Intel library; never select a dense fallback."""
    # Set Accelerate's documented process limit before loading its framework.
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    lib = C.CDLL(str(Path(__file__).parent / "build/libphonon_intel.dylib"))
    lib.p2_five_create.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_size_t, C.c_int, C.c_int]
    lib.p2_five_create.restype = C.c_void_p
    lib.p2_int_create.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_size_t, C.c_int, C.c_int]
    lib.p2_int_create.restype = C.c_void_p
    lib.p2_destroy.argtypes = [C.c_void_p]
    lib.p2_destroy.restype = None
    lib.p2_gemm.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
    lib.p2_gemm.restype = C.c_int
    lib.p2_row.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
    lib.p2_row.restype = C.c_int
    lib.p2_tdt.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                          C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_int,
                          C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.p2_tdt.restype = C.c_int
    lib.p2_encoder_create.argtypes = [C.c_int] * 7
    lib.p2_encoder_create.restype = C.c_void_p
    lib.p2_encoder_destroy.argtypes = [C.c_void_p]
    lib.p2_encoder_destroy.restype = None
    lib.p2_encoder_sub.argtypes = [C.c_void_p, C.c_void_p]
    lib.p2_encoder_sub.restype = C.c_int
    lib.p2_encoder_layer.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    lib.p2_encoder_layer.restype = C.c_int
    lib.p2_encoder_forward.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    lib.p2_encoder_forward.restype = C.c_int
    return lib


class Matrix:
    """Own a native matrix decoded directly from one compact container record."""

    def __init__(self, lib, entry, blob, threads=2, exact=False):
        self.lib, self.handle = lib, None
        self.rows, self.cols = entry["shape"]
        raw = C.create_string_buffer(blob)
        kind = entry["k"]
        if kind == "five_value":
            self.handle = lib.p2_five_create(self.rows, self.cols, raw, len(blob), threads, int(exact))
        elif kind in ("int6", "int8"):
            self.handle = lib.p2_int_create(self.rows, self.cols, raw, len(blob), int(kind[3:]), threads)
        else:
            raise ValueError(f"Unsupported native matrix type: {kind}")
        if not self.handle:
            raise ValueError(f"Invalid compact matrix: {entry['n']}")

    def __del__(self):
        if self.handle:
            self.lib.p2_destroy(self.handle)
            self.handle = None

    def gemm(self, x):
        """Multiply finite float32 activations by compact weights using AVX2."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.cols:
            raise ValueError("Incorrect activation dimensions")
        y = np.empty((x.shape[0], self.rows), dtype=np.float32)
        if self.lib.p2_gemm(self.handle, x.ctypes.data, x.shape[0], y.ctypes.data) != 0:
            raise RuntimeError("Native GEMM failed")
        return y

    def row(self, index):
        """Decode one row, for embeddings and exactness checks, never the full matrix."""
        out = np.empty(self.cols, dtype=np.float32)
        if self.lib.p2_row(self.handle, index, out.ctypes.data) != 0:
            raise ValueError("Invalid weight row")
        return out


class TDT:
    """Own compact decoder tables and run the entire recurrent loop in C++."""

    def __init__(self, lib, matrices, biases, config):
        self.lib, self.config = lib, config
        names = ["decoder.embedding.weight", "decoder.lstm.weight_ih_l0", "decoder.lstm.weight_hh_l0",
                 "decoder.lstm.weight_ih_l1", "decoder.lstm.weight_hh_l1", "decoder.decoder_projector.weight", "joint.head.weight"]
        bias_names = ["decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0", "decoder.lstm.bias_ih_l1",
                      "decoder.lstm.bias_hh_l1", "decoder.decoder_projector.bias", "joint.head.bias"]
        if set(matrices) != set(names) or set(biases) != set(bias_names):
            raise ValueError("Decoder checkpoint does not match the native TDT contract")
        self.matrices = [matrices[name] for name in names]
        self.biases = [np.ascontiguousarray(biases[name], dtype=np.float32) for name in bias_names]
        width = config["decoder_hidden_size"]
        expected = [4 * width] * 4 + [width, config["vocab_size"] + len(config["durations"])]
        if any(b.shape != (size,) for b, size in zip(self.biases, expected)):
            raise ValueError("Incorrect native decoder bias dimensions")
        self.handles = (C.c_void_p * 7)(*[m.handle for m in self.matrices])
        self.bias_ptrs = (C.c_void_p * 6)(*[b.ctypes.data for b in self.biases])
        self.durations = np.array(config["durations"], dtype=np.int32)

    def decode(self, encoded):
        """Return token IDs, emission frames and durations, or an explicit error."""
        encoded = np.ascontiguousarray(encoded, dtype=np.float32)
        cfg = self.config
        if encoded.ndim != 2 or encoded.shape[1] != cfg["decoder_hidden_size"] or not np.isfinite(encoded).all():
            raise ValueError("Invalid projected encoder output")
        cap = cfg["max_symbols_per_step"] * len(encoded) + 16
        tokens, times, lengths = (np.empty(cap, dtype=np.int32) for _ in range(3))
        count = self.lib.p2_tdt(self.handles, self.bias_ptrs, cfg["decoder_hidden_size"], cfg["vocab_size"],
                              self.durations.ctypes.data, len(self.durations), cfg["blank_token_id"], cfg["max_symbols_per_step"],
                              encoded.ctypes.data, len(encoded), tokens.ctypes.data, times.ctypes.data, lengths.ctypes.data, cap)
        if count < 0:
            raise RuntimeError(f"Native TDT failed: {count}")
        return tokens[:count].tolist(), times[:count].tolist(), lengths[:count].tolist()
