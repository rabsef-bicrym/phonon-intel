"""Native graph parity with independent Transformers operators on small tensors."""

import unittest

import numpy as np
import torch
from transformers import ParakeetEncoder, ParakeetEncoderConfig

from encoder import Encoder
from packed import Matrix, library
from reference import Linear, RelativePositions
from test_packed import five_record


class EncoderTests(unittest.TestCase):
    """Exercise full operator order, relative positions, padding and ownership."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        rng = np.random.default_rng(71)
        config = ParakeetEncoderConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, conv_kernel_size=3,
            subsampling_conv_channels=4, num_mel_bins=128, scale_input=False,
            attention_bias=False, convolution_bias=False, hidden_act="silu",
        )
        config._attn_implementation = "eager"
        cls.reference = ParakeetEncoder(config).eval()
        cls.reference.encode_positions = RelativePositions(config)
        lib = library()
        # Only the separately verified quantized GEMM is shared. All surrounding
        # graph operators run independently in Transformers and native C++.
        for name, module in list(cls.reference.named_modules()):
            if not name.startswith("layers."):
                continue
            if isinstance(module, torch.nn.Linear) or (
                    isinstance(module, torch.nn.Conv1d) and module.kernel_size == (1,)):
                rows, cols = module.weight.shape[:2]
                sign = rng.integers(-1, 2, (rows, cols), dtype=np.int8)
                hi = rng.integers(0, 2, (rows, cols)).astype(bool)
                entry, blob = five_record(sign, hi, np.full(rows, 0.015625), np.full(rows, 0.046875))
                matrix = Matrix(lib, entry, blob)
                parent, leaf = name.rsplit(".", 1)
                setattr(cls.reference.get_submodule(parent), leaf,
                        Linear(matrix, None, isinstance(module, torch.nn.Conv1d)))
            elif isinstance(module, torch.nn.BatchNorm1d):
                with torch.no_grad():
                    module.running_mean.copy_(torch.linspace(-0.3, 0.3, 16))
                    module.running_var.copy_(torch.linspace(0.7, 1.4, 16))
                    module.weight.copy_(torch.linspace(0.8, 1.2, 16))
                    module.bias.copy_(torch.linspace(-0.1, 0.1, 16))
        cls.lib = lib
        cls.matrices = {name: module.matrix for name, module in cls.reference.named_modules()
                        if isinstance(module, Linear)}
        cls.state = dict(cls.reference.state_dict())
        cls.state["encode_positions.inv_freq"] = cls.reference.encode_positions.inv_freq
        cls.native = Encoder(lib, config, cls.matrices, cls.state)

    @torch.inference_mode()
    def test_subsampling_and_layer_prefixes(self):
        rng = np.random.default_rng(19)
        for frames in (1, 8, 17, 32, 65):
            features = rng.normal(size=(frames, 128)).astype(np.float32)
            h = self.reference.subsampling(torch.from_numpy(features)[None])
            pos = self.reference.encode_positions(h)
            np.testing.assert_allclose(self.native.forward(features, layers=0), h[0], atol=1e-7, rtol=1e-5)
            for i, layer in enumerate(self.reference.layers):
                h = layer(h, position_embeddings=pos)
                np.testing.assert_allclose(self.native.forward(features, layers=i + 1), h[0], atol=2e-5, rtol=2e-5)

    def test_invalid_features_and_prefix_rejected(self):
        for features in (np.zeros((0, 128)), np.zeros((3002, 128)), np.zeros((8, 127)),
                         np.full((8, 128), np.nan)):
            with self.assertRaises(ValueError):
                self.native.forward(features)
        with self.assertRaises(RuntimeError):
            self.native.forward(np.zeros((8, 128)), layers=3)

    def test_checkpoint_records_are_required_and_consumed(self):
        # Native loading no longer relies on a Python reference graph to reject
        # missing or extra checkpoint records, including unsupported matrix bias.
        for name in ("subsampling.linear.weight", "layers.0.conv.norm.num_batches_tracked"):
            state = dict(self.state)
            del state[name]
            with self.assertRaises(KeyError):
                Encoder(self.lib, self.reference.config, self.matrices, state)
        for name in ("extra", "layers.0.feed_forward1.linear1.bias"):
            with self.assertRaises(ValueError):
                Encoder(self.lib, self.reference.config, self.matrices, {**self.state, name: torch.zeros(16)})
        matrices = dict(self.matrices)
        del matrices["layers.0.self_attn.q_proj"]
        with self.assertRaises(KeyError):
            Encoder(self.lib, self.reference.config, matrices, self.state)
        with self.assertRaises(ValueError):
            Encoder(self.lib, self.reference.config, {**self.matrices, "extra": next(iter(self.matrices.values()))}, self.state)


if __name__ == "__main__":
    unittest.main()
