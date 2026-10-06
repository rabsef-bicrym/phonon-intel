"""Require bitwise agreement with the installed Python PyTorch operator path."""

import unittest

import numpy as np
import torch

from frontend import Frontend
from reference import Features, RelativePositions, mel_filters


class FrontendTests(unittest.TestCase):
    """Check original operators, input bounds and model-owned tensor copies."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.filters = mel_filters()
        cls.native = Frontend(cls.filters,np.zeros((9,16),np.float32),np.zeros(9,np.float32),2)
        cls.reference = Features()

    def test_features_are_bitwise_identical(self):
        rng = np.random.default_rng(79)
        for samples in (160,161,255,400,1024,16000,171096,480000):
            for kind in ("random","silence","impulse"):
                x = (rng.normal(size=samples)*.05).astype(np.float32) if kind == "random" else np.zeros(samples,np.float32)
                if kind == "impulse":
                    x[0],x[-1] = .9,-.75
                with torch.inference_mode():
                    expected = self.reference.extract(x)
                actual = self.native.features(x)
                self.assertEqual(actual.tobytes(),expected[0].numpy().tobytes(),(samples,kind))

    def test_positions_are_bitwise_identical(self):
        for width in (16,64,1024):
            front = Frontend(self.filters,np.zeros((9,width),np.float32),np.zeros(9,np.float32),2)
            config = type("Config",(),{"hidden_size":width,"max_position_embeddings":5000})()
            expected = RelativePositions(config).inv_freq.numpy()
            self.assertEqual(front.positions().tobytes(),expected.tobytes())

    def test_projection_is_identical_and_owns_its_weights(self):
        rng = np.random.default_rng(27)
        for inputs,outputs in ((16,9),(1024,640)):
            w = rng.normal(size=(outputs,inputs)).astype(np.float32)
            b = rng.normal(size=outputs).astype(np.float32)
            weight,bias = torch.from_numpy(w.copy()),torch.from_numpy(b.copy())
            front = Frontend(self.filters,w,b,2)
            w[:],b[:] = np.nan,np.nan
            for frames in (1,17,134,376):
                x = rng.normal(size=(frames,inputs)).astype(np.float32)
                with torch.inference_mode():
                    expected = torch.nn.functional.linear(torch.from_numpy(x)[None],weight,bias)[0].numpy()
                self.assertEqual(front.project(x).tobytes(),expected.tobytes(),(inputs,outputs,frames))

    def test_input_guards_reject_invalid_arrays(self):
        for x in (np.zeros(159),np.zeros(480001),np.zeros((160,2))):
            with self.assertRaises(ValueError):
                self.native.features(x)
        for value in (np.nan,np.inf):
            with self.assertRaises(RuntimeError):
                self.native.features(np.full(160,value))
            with self.assertRaises(RuntimeError):
                self.native.project(np.full((2,16),value))
        for frames in (0,377):
            with self.assertRaises(RuntimeError):
                self.native.project(np.zeros((frames,16)))
        with self.assertRaises(ValueError):
            self.native.project(np.zeros((2,17)))

    def test_native_capacity_and_null_guards(self):
        audio = np.zeros(160,np.float32)
        out = np.full((2,128),123.0,np.float32)
        lib = self.native.lib
        self.assertNotEqual(lib.p2_frontend_features(self.native.handle,audio.ctypes.data,160,out.ctypes.data,1),0)
        self.assertTrue((out == 123.0).all())
        self.assertNotEqual(lib.p2_frontend_features(None,audio.ctypes.data,160,out.ctypes.data,2),0)
        self.assertNotEqual(lib.p2_frontend_project(None,audio.ctypes.data,1,out.ctypes.data),0)
        self.assertNotEqual(lib.p2_frontend_positions(None,out.ctypes.data),0)
        # A rejected call must not poison a subsequent valid call on this owner.
        self.assertTrue(np.isfinite(self.native.features(audio)).all())
        self.assertEqual(lib.p2_frontend_error(),b"")


if __name__ == "__main__":
    unittest.main()
