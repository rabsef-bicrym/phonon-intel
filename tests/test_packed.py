"""AVX2 output parity against the published quantized arithmetic contract."""

import unittest

import numpy as np

from packed import Matrix, TDT, library


def five_record(sign, high_mask, low, high):
    """Build the documented trit/bitmap/binary16 encoding for independent probes."""
    rows, cols = sign.shape
    codes = np.ones((rows, (cols + 4) // 5 * 5), dtype=np.uint8)
    codes[:, :cols] = sign + 1
    codes = codes.reshape(rows, -1, 5)
    trits = (codes * (3 ** np.arange(5))).sum(-1).astype(np.uint8).tobytes()
    bits = np.packbits(high_mask[sign != 0], bitorder="little").tobytes()
    blob = trits + bits + low.astype(np.float16).tobytes() + high.astype(np.float16).tobytes()
    return {"n": "probe", "k": "five_value", "shape": [rows, cols]}, blob


class PackedTests(unittest.TestCase):
    """Require an Intel build and test signed dots, padding, scales and tail rows."""

    @classmethod
    def setUpClass(cls):
        cls.lib = library()

    def test_five_value_arithmetic(self):
        rng = np.random.default_rng(43)
        for cols in (1, 31, 32, 33, 640):
            sign = rng.integers(-1, 2, (7, cols), dtype=np.int8)
            high_mask = rng.integers(0, 2, sign.shape).astype(bool)
            low = rng.uniform(0.01, 0.04, 7).astype(np.float16)
            high = (low.astype(np.float32) * 2.7).astype(np.float16)
            entry, blob = five_record(sign, high_mask, low, high)
            weights = sign * np.where(high_mask, high[:, None], low[:, None]).astype(np.float32)
            for exact in (False, True):
                matrix = Matrix(self.lib, entry, blob, threads=2, exact=exact)
                native_weights = np.stack([matrix.row(i) for i in range(7)])
                if exact:
                    np.testing.assert_array_equal(native_weights, weights)
                else:
                    bound = np.broadcast_to(high.astype(np.float32)[:, None] / 254 + 1e-7, weights.shape)
                    np.testing.assert_array_less(np.abs(native_weights - weights), bound)
                for count in (1, 3, 4, 5, 7, 8, 9, 15, 16, 17):
                    x = rng.normal(size=(count, cols)).astype(np.float32)
                    x[0] = 0
                    scale = np.maximum(np.abs(x).max(1, keepdims=True) / 127, 1e-20)
                    quantized = np.clip(np.rint(x / scale), -127, 127)
                    expected = (quantized.astype(np.float64) @ native_weights.astype(np.float64).T) * scale
                    np.testing.assert_allclose(matrix.gemm(x), expected, atol=3e-6, rtol=2e-5)

    def test_five_value_quantization_boundaries(self):
        # Exhaust all trit bytes and row tails; bitmap offsets cross byte boundaries.
        digits = ((np.arange(243)[:, None] // (3 ** np.arange(5))) % 3 - 1).astype(np.int8)
        for cols in range(1, 6):
            for low_value, high_value in ((0, 0), (0, 1), (1, 1), (0.125, 0.25),
                                          (2 ** -24, 2 ** -23), (2 ** -24, 65504)):
                sign = digits[:, :cols]
                high_mask = (np.arange(sign.size).reshape(sign.shape) % 2).astype(bool)
                low = np.full(243, low_value, dtype=np.float16)
                high = np.full(243, high_value, dtype=np.float16)
                entry, blob = five_record(sign, high_mask, low, high)
                magnitude = np.where(high_mask, high[:, None], low[:, None]).astype(np.float32)
                scale = high.astype(np.float32)[:, None] / np.float32(127)
                q = np.rint(np.divide(magnitude, scale, out=np.zeros_like(magnitude), where=scale != 0))
                expected = (sign * q).astype(np.int8).astype(np.float32) * scale
                for exact in (False, True):
                    matrix = Matrix(self.lib, entry, blob, exact=exact)
                    np.testing.assert_array_equal(np.stack([matrix.row(i) for i in range(243)]),
                                                  sign * magnitude if exact else expected)

    def test_saturation_and_int6(self):
        # Large same-sign products exercise the maddubs signed saturation boundary.
        for bits, q in ((8, np.full((3, 640), 127, dtype=np.int8)),
                        (6, np.full((3, 640), -32, dtype=np.int8))):
            scales = np.array([0.125, 0.5, 1], dtype=np.float16)
            if bits == 8:
                body = q.tobytes()
            else:
                codes = (q.astype(np.int32).ravel() + 32).reshape(-1, 4)
                values = (codes.astype(np.uint32) << (6 * np.arange(4, dtype=np.uint32))).sum(1)
                body = np.stack([values & 255, (values >> 8) & 255, (values >> 16) & 255], 1).astype(np.uint8).tobytes()
            entry = {"n": "int probe", "k": f"int{bits}", "shape": [3, 640]}
            matrix = Matrix(self.lib, entry, body + scales.tobytes())
            x = np.array([[127.0] * 640, [-127.0] * 640], dtype=np.float32)
            np.testing.assert_allclose(matrix.gemm(x), x @ (q * scales[:, None]).T, rtol=1e-6)

    def test_integer_gemm_bitwise_contract(self):
        # Integer dots have no reduction-order ambiguity. Check exact float bits
        # after scaling, including partial tiles and uneven worker partitions.
        rng = np.random.default_rng(73)
        for rows, cols in ((9, 1), (9, 31), (9, 32), (9, 33), (9, 1024),
                           (9, 4096), (9, 16384), (49, 33), (49, 1024)):
            q = rng.integers(-127, 128, (rows, cols), dtype=np.int8)
            q[0], q[1] = 127, -127
            weight_scale = rng.uniform(.001, 1, rows).astype(np.float16)
            blob = q.tobytes() + weight_scale.tobytes()
            entry = {"n": "integer parity", "k": "int8", "shape": [rows, cols]}
            for threads in (1, 2, 3):
                matrix = Matrix(self.lib, entry, blob, threads=threads)
                for count in (0, 1, 3, 4, 5, 8, 17):
                    x = rng.normal(size=(count, cols)).astype(np.float32)
                    if count:
                        x[0] = 127
                    if count > 1:
                        x[1] = -127
                    scale = np.abs(x).max(1, keepdims=True) / np.float32(127)
                    scale[scale == 0] = 1
                    activation = np.clip(np.rint(x / scale), -127, 127).astype(np.int32)
                    expected = (activation @ q.astype(np.int32).T).astype(np.float32)
                    expected *= weight_scale.astype(np.float32)[None]
                    expected *= scale
                    self.assertEqual(matrix.gemm(x).tobytes(), expected.tobytes())

    def test_invalid_records_and_activations_fail(self):
        entry, blob = five_record(np.ones((2, 32), dtype=np.int8), np.ones((2, 32), dtype=bool), np.ones(2), np.ones(2))
        with self.assertRaises(ValueError):
            Matrix(self.lib, entry, blob[:-1])
        with self.assertRaises(ValueError):
            Matrix(self.lib, entry, bytes([243]) + blob[1:])
        matrix = Matrix(self.lib, entry, blob)
        for value in (np.inf, np.nan):
            with self.assertRaises(RuntimeError):
                matrix.gemm(np.full((1, 32), value, dtype=np.float32))
        with self.assertRaises(ValueError):
            matrix.row(-1)

    def test_expansion_lookup_matches_integer_arithmetic(self):
        # Expansion shapes select the LUT path. Include batch tails and padded
        # five-trit groups so this does not merely exercise the fallback kernel.
        rng = np.random.default_rng(94)
        for cols in (33, 64, 128):
            rows = 2 * cols + 1
            sign = rng.integers(-1, 2, (rows, cols), dtype=np.int8)
            mask = rng.integers(0, 2, sign.shape).astype(bool)
            low = rng.uniform(.01, .04, rows).astype(np.float16)
            high = (low.astype(np.float32) * 2.7).astype(np.float16)
            entry, blob = five_record(sign, mask, low, high)
            scale = high.astype(np.float32) / np.float32(127)
            magnitude = np.where(mask, high[:, None], low[:, None]).astype(np.float32)
            weights = (sign * np.rint(magnitude / scale[:, None])).astype(np.int32)
            matrix = Matrix(self.lib, entry, blob, threads=2)
            for count in (32, 33, 64, 65):
                x = rng.normal(size=(count, cols)).astype(np.float32)
                activation_scale = np.abs(x).max(1, keepdims=True) / np.float32(127)
                q = np.clip(np.rint(x / activation_scale), -127, 127).astype(np.int32)
                expected = (q @ weights.T).astype(np.float32)
                expected *= scale[None]
                expected *= activation_scale
                self.assertEqual(matrix.gemm(x).tobytes(), expected.tobytes())

    def test_native_tdt_matches_quantized_reference(self):
        rng = np.random.default_rng(50)
        width, vocab = 8, 9
        config = {"decoder_hidden_size": width, "vocab_size": vocab, "blank_token_id": 8,
                  "durations": [0, 1, 2], "max_symbols_per_step": 3}
        shapes = {"decoder.embedding.weight": (vocab, width),
                  "decoder.lstm.weight_ih_l0": (4 * width, width),
                  "decoder.lstm.weight_hh_l0": (4 * width, width),
                  "decoder.lstm.weight_ih_l1": (4 * width, width),
                  "decoder.lstm.weight_hh_l1": (4 * width, width),
                  "decoder.decoder_projector.weight": (width, width), "joint.head.weight": (vocab + 3, width)}
        matrices = {}
        for name, shape in shapes.items():
            q = rng.integers(-31, 32, shape, dtype=np.int8)
            scales = np.full(shape[0], 0.015625, dtype=np.float16)
            entry = {"n": name, "k": "int8", "shape": shape}
            matrices[name] = Matrix(self.lib, entry, q.tobytes() + scales.tobytes(), threads=1)
        bias_names = ["decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0", "decoder.lstm.bias_ih_l1",
                      "decoder.lstm.bias_hh_l1", "decoder.decoder_projector.bias", "joint.head.bias"]
        biases = {name: rng.normal(0, 0.05, length).astype(np.float32)
                  for name, length in zip(bias_names, [4 * width] * 4 + [width, vocab + 3])}
        native = TDT(self.lib, matrices, biases, config)
        encoded = rng.normal(size=(16, width)).astype(np.float32)
        # Each stage is evaluated in Python/NumPy rather than the native recurrent loop.
        def project(name, x):
            return matrices[name].gemm(x[None])[0]

        state_h = np.zeros((2, width), dtype=np.float32)
        state_c = np.zeros_like(state_h)
        last, frame, symbols = 8, 0, 0
        tokens, times, lengths = [], [], []
        while frame < len(encoded):
            x = matrices["decoder.embedding.weight"].row(last)
            next_h, next_c = np.empty_like(state_h), np.empty_like(state_c)
            for layer in range(2):
                gates = project(f"decoder.lstm.weight_ih_l{layer}", x)
                gates += project(f"decoder.lstm.weight_hh_l{layer}", state_h[layer])
                gates += biases[f"decoder.lstm.bias_ih_l{layer}"] + biases[f"decoder.lstm.bias_hh_l{layer}"]
                i, f, g, o = np.split(gates, 4)
                sigmoid = lambda value: 1 / (1 + np.exp(-value))
                next_c[layer] = sigmoid(f) * state_c[layer] + sigmoid(i) * np.tanh(g)
                x = next_h[layer] = sigmoid(o) * np.tanh(next_c[layer])
            prediction = project("decoder.decoder_projector.weight", x) + biases["decoder.decoder_projector.bias"]
            scores = project("joint.head.weight", np.maximum(encoded[frame] + prediction, 0)) + biases["joint.head.bias"]
            token = int(scores[:vocab].argmax())
            duration = config["durations"][int(scores[vocab:].argmax())]
            if token == 8 and duration == 0:
                duration = 1
            if token != 8:
                tokens.append(token); times.append(frame); lengths.append(duration)
                last, state_h, state_c = token, next_h, next_c
            symbols = symbols + 1 if duration == 0 else 0
            if symbols >= 3:
                duration, symbols = 1, 0
            frame += duration
        self.assertTrue(tokens)
        self.assertEqual(native.decode(encoded), (tokens, times, lengths))
        self.assertEqual(native.decode(encoded), (tokens, times, lengths))

        # Force blank + zero-duration: it must advance, without emitting or hanging.
        head_bias = native.biases[-1]
        head_bias[:] = -1e6
        head_bias[8] = head_bias[vocab] = 1e6
        self.assertEqual(native.decode(encoded), ([], [], []))
        # Force token + zero-duration: the per-frame emission limit must advance.
        head_bias[8] = -1e6
        head_bias[0] = 1e6
        self.assertEqual(native.decode(encoded),
                         ([0] * 48, np.repeat(np.arange(16), 3).tolist(), [0] * 48))


if __name__ == "__main__":
    unittest.main()
