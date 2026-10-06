# Phonon-2 on Intel Macs

An independent, experimental CPU runtime for the published
[Phonon-2](https://github.com/fermionresearch/phonon) weights. It combines AVX2
compact matrix kernels, a native FastConformer encoder and TDT decoder,
Accelerate, and PyTorch's native ATen operators.

This is **not Fermion's official runtime** or a reconstruction of its unpublished
native source. Fermion's official Intel Mac runtime also works; see the
[validation notes](docs/results.md). This repository shares our implementation,
optimization method and measured results for collaboration.

## Results

Physical Core i5-8500B Mac mini, 8 GiB RAM, macOS 15.8.1, two threads.
Complete fresh processes, including startup and model loading:

| Recording | This runtime | whisper.cpp base.en |
| --- | ---: | ---: |
| 10.69 seconds | **2.677 s** | 2.791 s |
| 109.05 seconds | **13.490 s** | 18.995 s |

The short result is the median of seven alternating pairs; the long result is
the median of three pairs using the packaged build. The filesystem cache was
not flushed. One first launch after compilation took 3.334 seconds on the short
recording, so **not every launch beats Whisper**. Peak RAM is higher than Whisper:
about 0.98 GB / 1.20 GB here versus 0.30 GB / 0.52 GB on these inputs.

No resident process, persistent converted-weight cache or caller-side chunking
is used. These two private recordings are not a representative accuracy corpus.
Output hashes establish parity with our previous quantized implementation, not
equal accuracy or bitwise equivalence to Fermion's runtime.

- [Method and rejected approaches](docs/method.md)
- [Results, raw timings and limitations](docs/results.md)
- [Source provenance and attribution](docs/provenance.md)

## Build

Requires **Intel macOS with AVX2 and F16C**, Apple's Command Line Tools and
Python 3.12. Use an isolated environment. The tested PyTorch 2.2.2 Intel wheel
provides native headers/libraries; inference does not import Python's `torch`.
Rebuild if that wheel changes. No native ABI compatibility across wheel versions
is promised. This build does not target Linux, Windows or Apple Silicon.

```sh
git clone https://github.com/rabsef-bicrym/phonon-intel.git
cd phonon-intel
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py
```

Obtain and verify the published model separately:

```sh
curl -fL https://huggingface.co/FermionResearch/Phonon-2/resolve/main/phonon-2.bps.tar.zst -o /tmp/phonon-2.bps.tar.zst
.venv/bin/python intel_macos/unpack_model.py /tmp/phonon-2.bps.tar.zst models/phonon-2
```

The unpacker pins the release archive checksum, verifies extracted member hashes
and refuses to overwrite an existing directory. A failed extraction can leave
a partial new directory; inspect it before removing or retrying. An upstream
archive change fails verification rather than silently loading different weights.

## Transcribe

```sh
.venv/bin/python intel_macos/runtime.py --model-dir models/phonon-2 --threads 2 recording.wav
```

Stdout contains one JSON object per input file: `text`, timed `words`, `file`,
`audio_seconds`, and inference `seconds`. Model-loading time goes to stderr.
Pass multiple files to share a model within that invocation. No service is
installed. The Python API is `runtime.Transcriber` plus `runtime.read_audio`
with `intel_macos` on the import path; use each instance sequentially.

Complete recordings are accepted. Phonon's upstream energy/pause segmentation
handles long inputs internally, with a 30-second segment limit and file-relative
word timestamps. It can discard quiet passages. Supported audio formats depend
on libsndfile; unsupported input can be converted to mono 16 kHz WAV.

## Test and benchmark

The synthetic tests need no model download or private recordings:

```sh
.venv/bin/python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python tests/run.py
.venv/bin/python benchmarks/summarize.py
```

Use your own WAV recordings for a fresh-process comparison:

```sh
.venv/bin/python benchmarks/compare.py \
  --model-dir models/phonon-2 \
  --whisper-cli /path/to/whisper-cli \
  --whisper-model /path/to/ggml-base.en.bin \
  --threads 2 --repeats 3 --output benchmark-results.json \
  short.wav long.wav
```

This alternates complete CLI processes, records peak RSS, and saves hashes rather
than transcripts or input paths. It refuses to overwrite an existing result.
The recorded historical comparison used an API wrapper per fresh process;
this portable harness measures the public CLI, so small startup differences
between harnesses are possible.

## Precision and scope

The native path already uses signed-byte weights and activations. A five-value
weight row is mapped using its high magnitude as the scale, with weight error
bounded by `high / 254`; activations are quantized per input row. The optimization
preserves that arithmetic. **It is not lossless float32 model inference.**
The lower-level two-plane matrix mode preserves stored weight magnitudes but
still quantizes activations; it is not the transcription CLI's default.

This is a research implementation with a tested Intel build, not a general ASR
quality claim or a replacement for upstream support. Broader word-error-rate
evaluation, additional CPUs and deployment lifecycle work remain useful next
steps. No private audio, transcripts, credentials or model weights are published.

Apache-2.0 source. Upstream license and notice are retained. Model weights and
runtime dependencies retain their respective licenses; see
[provenance](docs/provenance.md).
