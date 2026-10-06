# Results and limits

Measured October 4-5, 2026 on a physical Intel Core i5-8500B Mac mini: six cores,
8 GiB RAM, macOS 15.8.1. Two threads, one benchmark process at a time, on an active
host. Python 3.12.13, PyTorch 2.2.2, NumPy 1.26.4, SciPy 1.17.1, SoundFile 0.14.0.
Whisper comparison: whisper.cpp 1.9.2, `base.en`.

Inputs are the same 10.6935-second and 109.0535-second mono 16 kHz recordings.
They are private and not distributed. Raw published results contain timings,
peak RSS and output hashes only. Others can reproduce the procedure on their
own audio, but cannot independently reproduce these exact recordings from this
repository. No human-corrected reference corpus or WER claim is provided.

## Complete-process comparison

Times include imports, model loading, audio reading, transcription and exit.
Each run uses a fresh process. Filesystem caches were not flushed; these are not
cold-boot measurements. Inference-only timings must not be substituted for these
numbers. The historical Phonon harness used the Python API in a fresh process;
Whisper used its CLI. The portable harness in this repository uses both CLIs.

| Experiment | Runs per input/engine | Short Phonon / Whisper | Long Phonon / Whisper |
| --- | ---: | ---: | ---: |
| Python wrapper control | 3 | 4.124 / 2.813 s | 14.988 / 18.950 s |
| ATen bridge, sequential loading | 3 | 3.275 / 2.813 s | 14.063 / 18.950 s |
| ATen bridge, two-worker loading | 3 | 2.676 / 2.784 s | 13.526 / 19.004 s |
| Additional alternating short comparison | 7 | 2.677 / 2.791 s | Not run |
| Rebuilt standalone candidate | 3 | 2.685 / 2.852 s | 13.490 / 18.995 s |

The seven-pair short confirmation ranges are 2.643-2.700 s for Phonon and
2.780-2.861 s for Whisper. Phonon wins all seven pairs, with a 4.1% lower median.
The rebuilt long-recording median is 29.0% lower than Whisper's.

**The rebuilt candidate's first short launch took 3.334 s**, including 1.784 s
of model construction. The next two took 2.685 and 2.668 s, with construction
around 1.19 s. The slower first sample is included, not discarded. Its extra
startup cost has not been attributed. These results do not establish that every
launch or every recording is faster than Whisper.

Memory remains a tradeoff. Rebuilt median peak RSS:

| Recording | Phonon | Whisper |
| --- | ---: | ---: |
| Short | 982,274,048 bytes | 298,504,192 bytes |
| Long | 1,202,163,712 bytes | 515,837,952 bytes |

## Raw evidence

- [`results.json`](../benchmarks/results/results.json): Python control, native
  ATen bridge and Whisper, with rotating order.
- [`parallel-results.json`](../benchmarks/results/parallel-results.json): serial
  and parallel native loaders against Whisper.
- [`short-confirmation.json`](../benchmarks/results/short-confirmation.json):
  seven alternating short-recording pairs.
- [`candidate-results.json`](../benchmarks/results/candidate-results.json):
  source-rebuilt standalone candidate against Whisper.

`python benchmarks/summarize.py` checks all 37 Phonon runs against the Python
control's text and word-timestamp hashes. Every native benchmark process also
checked that `torch` and `torch.*` were absent from Python's module registry.
ATen's native CPU library remains in use.

Hashes show preservation across our optimization steps. They do not establish
that the original custom implementation is equivalent to a different recognizer
or to Fermion's official engine.

## Official Intel runtime validation

Fermion Research **0.2.9 works on this physical Intel Mac**, without Rosetta.
It reports the native C encoder and C TDT decoder, AVX2 tier, 264 packed linears,
and successful binary self-tests, without the fp32 fallback.

An earlier October 4 test measured loaded-model medians of 1.578 s / 16.919 s,
and fresh-process medians of 11.585 s / 27.050 s on the same two recordings.
The official generated plane cache was present for the latter comparison.
These are historical measurements of 0.2.9, not a claim about later releases or
a simultaneous comparison against the optimized candidate above.

The short transcript matched our earlier custom runtime. The long transcripts
had small wording differences; using matched audio windows removed one of those
differences but not all. There is no ground-truth transcript establishing which
was more accurate. The official-runtime raw logs include private transcript text
and are not distributed.

## Validation

The publication build passes 21 synthetic tests on the physical Intel machine:
independent quantized matrix arithmetic, expansion lookup batches and tails,
native encoder prefixes against Transformers, recurrent decoder state,
bitwise ATen features/projection/positions, native bounds, worker cleanup,
checkpoint contract failures, archive verification, audio resampling and full-recording framing.
They require no model download or private test data.

Before packaging, optimized and ASan/UBSan frontend tests passed, and both full
recordings passed with the matrix/encoder library and ATen bridge instrumented.
Text and timestamp hashes matched the non-instrumented path. Python-hosted leak
checking was disabled, and UBSan failures were fatal. Precompiled dependency
internals were not instrumented. Sanitizer timings are not benchmark results.

The standalone CLI was invoked separately and its JSON output checked against
the same short-recording hashes. The published runtime sources preserve that
candidate; test fixtures and benchmark paths have been made portable.

Useful remaining validation includes public-corpus WER, more CPUs, cold-boot
behavior, input segmentation boundaries and deployment lifecycle tests. This
repository does not install or replace any live transcription service.
