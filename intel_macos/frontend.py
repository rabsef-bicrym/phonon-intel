"""Own native ATen preprocessing and projection with NumPy-shaped boundaries."""

import ctypes as C
from pathlib import Path

import numpy as np


class Frontend:
    """Keep the model's small float operators without Python's torch import."""

    def __init__(self, filters, weight, bias, threads):
        self.handle = None
        filters, weight, bias = [np.ascontiguousarray(a, dtype=np.float32) for a in (filters, weight, bias)]
        if (weight.ndim != 2 or bias.shape != (weight.shape[0],) or filters.ndim != 2 or filters.shape[1] != 257):
            raise ValueError("Invalid frontend weights")
        self.outputs, self.inputs = weight.shape
        self.mels = filters.shape[0]
        self.lib = lib = C.CDLL(str(Path(__file__).resolve().parents[1] / "frontend.dylib"))
        lib.p2_frontend_create.argtypes = [C.c_void_p]*3 + [C.c_int]*4
        lib.p2_frontend_create.restype = C.c_void_p
        lib.p2_frontend_destroy.argtypes = [C.c_void_p]
        lib.p2_frontend_destroy.restype = None
        lib.p2_frontend_error.restype = C.c_char_p
        lib.p2_frontend_features.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int]
        lib.p2_frontend_project.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        lib.p2_frontend_positions.argtypes = [C.c_void_p,C.c_void_p]
        self.handle = lib.p2_frontend_create(filters.ctypes.data,weight.ctypes.data,bias.ctypes.data,
                                           self.inputs,self.outputs,self.mels,threads)
        if not self.handle:
            raise RuntimeError(lib.p2_frontend_error().decode())

    def __del__(self):
        if self.handle:
            self.lib.p2_frontend_destroy(self.handle)
            self.handle = None

    def _check(self, status):
        """Surface native failures rather than returning an uninitialized output."""
        if status:
            raise RuntimeError(self.lib.p2_frontend_error().decode())

    def features(self, audio):
        """Return contiguous time-by-mel features for one upstream-sized segment."""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not 160 <= len(audio) <= 480000:
            raise ValueError("Invalid feature audio shape")
        frames = len(audio)//160+1
        out = np.empty((frames,self.mels), dtype=np.float32)
        self._check(self.lib.p2_frontend_features(self.handle,audio.ctypes.data,len(audio),out.ctypes.data,frames))
        return out

    def project(self, encoded):
        """Apply the checkpoint's dense output projection with the original ATen op."""
        encoded = np.ascontiguousarray(encoded, dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[1] != self.inputs:
            raise ValueError("Invalid projection shape")
        out = np.empty((len(encoded),self.outputs), dtype=np.float32)
        self._check(self.lib.p2_frontend_project(self.handle,encoded.ctypes.data,len(encoded),out.ctypes.data))
        return out

    def positions(self):
        """Produce the exact float32 inverse frequencies for the native encoder."""
        out = np.empty(self.inputs//2,dtype=np.float32)
        self._check(self.lib.p2_frontend_positions(self.handle,out.ctypes.data))
        return out
