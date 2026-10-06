"""Independent Python operator references for native-boundary tests."""

import numpy as np
import torch
from torch import nn

from runtime import mel_filters


class RelativePositions(nn.Module):
    """Evaluate Parakeet's float32 position expression on PyTorch 2.2."""

    def __init__(self, config):
        super().__init__()
        frequency = 1.0 / (10000.0 ** (torch.arange(0, config.hidden_size, 2).float() / config.hidden_size))
        self.register_buffer("inv_freq", frequency, persistent=False)

    def forward(self, hidden_states):
        """Return interleaved relative sine/cosine positions for the input length."""
        length = hidden_states.shape[1]
        positions = torch.arange(length - 1, -length, -1).float()
        angles = positions[:, None] * self.inv_freq[None, :]
        embeddings = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return embeddings[None].expand(hidden_states.shape[0], -1, -1)


class Features:
    """Preserve the Python feature operator sequence used before the C bridge."""

    def __init__(self):
        self.window = torch.hann_window(400, periodic=False)
        self.filters = torch.from_numpy(mel_filters())

    def extract(self, audio):
        """Return the original preemphasis, STFT, mel and normalization result."""
        waveform = torch.from_numpy(audio)[None]
        waveform = torch.cat([waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)
        spectrum = torch.stft(waveform, 512, hop_length=160, win_length=400,
                              window=self.window, return_complex=True, pad_mode="constant")
        power = torch.view_as_real(spectrum).pow(2).sum(-1).sqrt().pow(2)
        features = torch.log(self.filters @ power + 2 ** -24).permute(0, 2, 1)
        mean = features.mean(1, keepdim=True)
        variance = ((features - mean) ** 2).sum(1) / (features.shape[1] - 1)
        return (features - mean) / (variance.sqrt().unsqueeze(1) + 1e-5)


class Linear(nn.Module):
    """Share only quantized matrix arithmetic with an independent HF graph."""

    def __init__(self, matrix, bias=None, convolution=False):
        super().__init__()
        self.matrix, self.bias, self.convolution = matrix, bias, convolution

    def forward(self, x):
        """Apply a separately tested matrix to either linear or pointwise input."""
        if self.convolution:
            x = x.transpose(1, 2)
        shape = x.shape
        y = torch.from_numpy(self.matrix.gemm(np.ascontiguousarray(x.detach().numpy().reshape(-1, shape[-1]))))
        y = y.reshape(*shape[:-1], self.matrix.rows)
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(1, 2) if self.convolution else y
