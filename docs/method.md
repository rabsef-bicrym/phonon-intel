# Method

## Profile the complete process

The target is a process-per-recording workflow on an AVX2 Intel Mac, not a server
with a model already loaded. Measure imports, model construction, audio handling,
encoder, decoder and process exit separately. Keep the complete process as the
acceptance metric. A faster matrix microbenchmark can make an entire note slower.

The retained implementation uses two threads. Every benchmark process finishes
before the next starts. It does not change a live assistant or its transcription
configuration. The published timing samples include outliers and ordinary OS
filesystem caching; they do not measure a cold filesystem or cold boot.

## Keep the graph native

`intel_macos/encoder.cpp` executes the FastConformer graph. Small convolutions and
attention products use Accelerate; large linears use compact AVX2 kernels.
`p2_tdt` in `intel_macos/packed.cpp` executes greedy recurrent decoding, including
blank-token state retention and bounded zero-duration emissions. Python does not
dispatch every transformer layer or decoder token.

`intel_macos/encoder.py` validates checkpoint shapes and retains matrix owners for
every pointer borrowed by the native graph. The loader rejects missing or extra
records instead of filling gaps with initialized model parameters.

## Preserve quantized arithmetic while changing its layout

The starting native implementation uses signed-byte weights/activations, row
scales, integer dot products and two separate float multiplications. These
optimizations retain that contract, including its existing approximation to the
published five-value weights. `-ffp-contract=off` preserves the separate rounding.

The direct-dot kernel reuses activations across four output rows and two input
rows, within 24-output-row blocks. Four horizontal reductions are combined in
SIMD. The signed-dot construction uses absolute activations and moves their signs
onto weights before `maddubs`; the supported magnitude bound prevents saturation.

For expansion matrices (`rows >= 2 * cols`), a lookup path exploits the weight
alphabet. Each group of five signed ternary coefficients has only 243 possible
codes. Separate sign and high-magnitude planes reproduce the existing quantized
weight as a low magnitude plus a high-minus-low contribution. Activation sums for
all codes are shared across output rows.

The retained layout uses 32 input frames, blocks of eight five-weight groups,
and 16-bit table entries, accumulating into 32-bit output sums. Small batches
and tails use the direct-dot kernel. Restricting expansion to eligible matrices
avoids the memory cost of building lookup layouts for every matrix.

## Decode the checkpoint once per process

`p2_five_create` uses the 243-byte wire alphabet to precompute trit signs and
prefix counts. Prefix counts locate the packed nonzero-magnitude bits, including
partial final groups. A compile-time table maps those bits into SIMD lane masks;
five signed weight bytes are constructed together. This table describes the
fixed format, not a model cache.

Low/high magnitudes are quantized once per row. The compact container is streamed
record by record. Two independent constructors run through `ctypes.CDLL`, which
releases the GIL while native code executes. Pending jobs are bounded to the
thread count, plus the record being read. Each constructor owns its allocations;
shared decode tables are compile-time constants. Graph assembly waits for all
constructors, and both producer and worker failures join the pool.

## Keep ATen, remove Python torch startup

Importing Python's torch package was a material fixed cost after kernel tuning.
The Intel wheel includes ATen headers and its CPU library. `frontend.cpp` calls
those same operators through a small C ABI; `intel_macos/frontend.py` manages
NumPy buffers and native handles without importing `torch`.

The bridge preserves the original CPU expression, including Python STFT's input
padding, square-root-then-square power calculation, reduction order, projection
shape and reciprocal operation order. Algebraically simplifying those operations
could change float rounding and is not part of this optimization. The model owns
copies of the small projection/filter tensors; exceptions do not cross the C ABI.

This removes Python wrapper startup, not the PyTorch dependency. The build links
against the installed wheel, and must be rebuilt if that dependency changes.

## Experiments not retained

- Larger register tiles, alternate loop orders and manual prefetching did not
  establish a consistent complete-recording gain.
- Lookup layouts on every matrix added memory without enough latency benefit.
- Skipping initial weight zero-fill did not provide a useful measured gain.
- Vectorized validation/counting and a scalar packed-word decoder did not beat
  the retained SIMD checkpoint loader end to end.
- Resident processes and persistent converted-weight caches were excluded from
  the target workflow, rather than credited as arithmetic improvements.

## Verification boundaries

Synthetic tests independently evaluate integer-dot scaling, expansion lookups,
native encoder prefixes against Transformers, recurrent state transitions, input
bounds, loader errors and frontend operations against Python PyTorch. Full-note
output hashes protect text and word timestamps across the retained optimization
steps. ASan/UBSan checks cover native boundaries, but not code inside precompiled
third-party libraries; Python-hosted leak checking was disabled.

These checks are not a corpus-wide accuracy evaluation or a proof of equivalence
to Fermion's native binary. The original custom graph had small transcript
differences from that binary. The published results say exactly what was measured.
