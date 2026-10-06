"""Shape-checked ownership boundary for the native FastConformer encoder."""

import ctypes as C

import numpy as np


class Encoder:
    """Copy small float tensors; retain owners of every borrowed compact matrix."""

    def __init__(self, lib, config, matrices, state):
        self.lib, self.handle, self.matrices = lib, None, []
        matrices, state = dict(matrices), dict(state)
        d, inner, channels = config.hidden_size, config.intermediate_size, config.subsampling_conv_channels
        self.width, self.layers, self.mels = d, config.num_hidden_layers, config.num_mel_bins
        if (config.subsampling_factor != 8 or config.subsampling_conv_stride != 2 or
                config.subsampling_conv_kernel_size != 3 or config.scale_input or config.convolution_bias or
                config.attention_bias or config.hidden_act != "silu" or
                config.num_key_value_heads != config.num_attention_heads):
            raise ValueError("Unsupported native encoder configuration")
        self.handle = lib.p2_encoder_create(self.layers, d, inner, config.num_attention_heads,
                                             config.conv_kernel_size, config.num_mel_bins, channels)
        if not self.handle:
            raise ValueError("Invalid native encoder dimensions")

        names = [f"subsampling.layers.{i}.{field}" for i in (0, 2, 3, 5, 6) for field in ("weight", "bias")]
        names += ["subsampling.linear.weight", "subsampling.linear.bias", "encode_positions.inv_freq"]
        vectors = [state.pop(name) for name in names]
        sizes = [channels * 9, channels, channels * 9, channels, channels * channels, channels,
                 channels * 9, channels, channels * channels, channels, d * channels * (config.num_mel_bins // 8), d, d // 2]
        arrays, pointers = self._vectors(vectors, sizes)
        if lib.p2_encoder_sub(self.handle, pointers) != 0:
            raise RuntimeError("Native subsampling setup failed")
        for index in range(self.layers):
            prefix = f"layers.{index}."
            names = ["feed_forward1.linear1", "feed_forward1.linear2", "self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj", "self_attn.relative_k_proj", "conv.pointwise_conv1",
                     "conv.pointwise_conv2", "feed_forward2.linear1", "feed_forward2.linear2"]
            layer_matrices = [matrices.pop(prefix + name) for name in names]
            dimensions = [(inner, d), (d, inner)] + [(d, d)] * 5 + [(2 * d, d), (d, d), (inner, d), (d, inner)]
            if any((matrix.rows, matrix.cols) != shape for matrix, shape in zip(layer_matrices, dimensions)):
                raise ValueError("Incorrect native encoder matrix shape")
            self.matrices.extend(layer_matrices)
            handles = (C.c_void_p * 11)(*[matrix.handle for matrix in layer_matrices])
            names = [f"{norm}.{field}" for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv",
                     "norm_feed_forward2", "norm_out") for field in ("weight", "bias")]
            names += ["self_attn.bias_u", "self_attn.bias_v", "conv.depthwise_conv.weight",
                      "conv.norm.weight", "conv.norm.bias", "conv.norm.running_mean", "conv.norm.running_var"]
            vectors = [state.pop(prefix + name) for name in names]
            sizes = [d] * 12 + [d * config.conv_kernel_size] + [d] * 4
            arrays, pointers = self._vectors(vectors, sizes)
            if lib.p2_encoder_layer(self.handle, index, handles, pointers) != 0:
                raise RuntimeError(f"Native layer {index} setup failed")
            # Batch counters are checkpoint state, but inference uses frozen means
            # and variances. Still require their declared scalar shape.
            if state.pop(prefix + "conv.norm.num_batches_tracked").shape != ():
                raise ValueError("Invalid batch counter shape")
        if matrices or state:
            raise ValueError(f"Unexpected encoder records: {sorted((*matrices, *state))}")

    @staticmethod
    def _vectors(tensors, sizes):
        """Validate each input length before C++ copies the pointed-to storage."""
        arrays = [np.ascontiguousarray(t.reshape(-1), dtype=np.float32) for t in tensors]
        if any(a.size != n for a, n in zip(arrays, sizes)):
            raise ValueError("Native encoder tensor length mismatch")
        return arrays, (C.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])

    def __del__(self):
        if self.handle:
            self.lib.p2_encoder_destroy(self.handle)
            self.handle = None

    def forward(self, features, layers=None):
        """Run a full encoder, or a prefix for numerical boundary verification."""
        features = np.ascontiguousarray(features, dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != self.mels or
                not 1 <= len(features) <= 3001 or not np.isfinite(features).all()):
            raise ValueError(f"Expected finite [1..3001 frames,{self.mels}] mel features")
        count = (len(features) + 7) // 8
        output = np.empty((count, self.width), dtype=np.float32)
        result = self.lib.p2_encoder_forward(self.handle, features.ctypes.data, len(features),
                                             self.layers if layers is None else layers, output.ctypes.data)
        if result != count:
            raise RuntimeError(f"Native encoder failed: {result}")
        return output
