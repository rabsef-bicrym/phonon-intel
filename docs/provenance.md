# Source provenance

The runtime builds on Fermion Research's published Phonon-2 model format,
configuration and Python reference behavior. It independently implements native
Intel operators; no unpublished native Fermion source was used or redistributed.

The following upstream files are copied unchanged from
[`fermionresearch/phonon` at `ba0339cb01d6103a4c632cfc8c7744c23c1587cb`](https://github.com/fermionresearch/phonon/tree/ba0339cb01d6103a4c632cfc8c7744c23c1587cb):

- `docker-phonon2/fermion_container.py`: compact tensor decoding helpers.
- `docker-phonon2/phonon2_cuda_engine.py`: configuration, mel filters and timed
  word assembly. Only these CPU-independent helpers are used by this runtime;
  the upstream CUDA entry point is not a supported interface in this repository.
- `docker-phonon2/_live.py`: whole-recording segmentation.
- `package_release_bps.py`: archive byte-plane reconstruction helper.
- `LICENSE` and `NOTICE`: upstream license and attribution.

Our additions and modifications are `intel_macos/`, `frontend.cpp`, the build,
tests, benchmarks and documentation. The native frontend mirrors CPU feature
arithmetic from the published Fermion Python implementation and PyTorch wrappers.
Encoder tests use Hugging Face Transformers' Parakeet operators as an independent
graph reference. `tests/reference.py` preserves the corresponding Python operator
expressions for compatibility with the tested PyTorch 2.2.2 environment.

The expansion-matrix lookup layout was inspired by
[Vec-LUT (Li et al., MobiSys 2026)](https://arxiv.org/html/2512.06443v2).
Our five-valued-weight adaptation is independently implemented; no code from
the paper's LLM runtime is included. See [the method](method.md) for the boundary.

The upstream NOTICE also describes upstream assets not included in this smaller
repository. No model weights, third-party wheels, precompiled libraries, private
recordings or private transcript text are distributed here.

Runtime dependencies: PyTorch (BSD-style), NumPy (BSD-3-Clause), SciPy
(BSD-3-Clause), SoundFile (BSD-3-Clause, dynamically linked libsndfile), Zstandard
(BSD license). Test-only Transformers uses Apache-2.0. Apple Accelerate is supplied
by macOS. Consult the distributions for their complete terms. Model weights
retain the license declared by their publisher.

Original Intel implementation and optimization work: Eric Helal, with AI-assisted
development and testing. The added source is distributed under the included
Apache License, Version 2.0.
