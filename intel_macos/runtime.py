"""Experimental native Phonon path using ATen without importing Python torch."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker-phonon2"))
from fermion_container import FORMAT, _intn
from phonon2_cuda_engine import HF_CONFIG, mel_filters, words_from_tokens
from _live import LiveSession
from encoder import Encoder
from frontend import Frontend
from packed import library, Matrix, TDT

SAMPLE_RATE = 16000


def records(path):
    """Read and validate one compact record at a time from the published container."""
    with path.open("rb") as source:
        size = int.from_bytes(source.read(8), "little")
        header = json.loads(source.read(size))
        if header["format"] != FORMAT:
            raise ValueError("Unsupported model format")
        for entry in header["index"]:
            blob = source.read(entry["b"])
            if len(blob) != entry["b"]:
                raise ValueError(f"Truncated tensor: {entry['n']}")
            yield entry, blob
        if source.read(1):
            raise ValueError("Trailing model bytes")


def load_model(directory, threads):
    """Consume every checkpoint record into its native architectural owner."""
    lib = library(threads)
    config = SimpleNamespace(**HF_CONFIG["encoder_config"])
    encoder_matrices, decoder_matrices, decoder_biases, state = {}, {}, {}, {}
    pending = deque()

    def finish_matrix():
        """Transfer the oldest completed native owner into its model section."""
        destination, name, future = pending.popleft()
        destination[name] = future.result()

    # ctypes releases the GIL during native construction. Bound queued blobs to
    # the worker count and join every constructor before graph assembly or error.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for entry, blob in records(directory / "model.fermion"):
            name, kind, shape = entry["n"], entry["k"], entry["shape"]
            if kind == "five_value" or (name.startswith(("decoder.", "joint.")) and kind in ("int6", "int8")):
                if len(pending) == threads:
                    finish_matrix()
                destination = encoder_matrices if kind == "five_value" else decoder_matrices
                key = name.removeprefix("encoder.") if kind == "five_value" else name
                pending.append((destination,key,pool.submit(Matrix,lib,entry,blob,threads)))
                continue
            if name.startswith(("decoder.", "joint.")):
                if kind == "fp16":
                    decoder_biases[name] = np.frombuffer(blob,dtype=np.float16).reshape(shape).astype(np.float32)
                else:
                    raise ValueError(f"Unsupported decoder record: {name} ({kind})")
                continue
            if kind in ("int6", "int8"):
                value = _intn(blob,shape,int(kind[3:]))
            elif kind == "fp16":
                value = np.frombuffer(blob,dtype=np.float16).reshape(shape)
            else:
                raise ValueError(f"Unsupported tensor encoding: {kind}")
            if name.endswith("num_batches_tracked"):
                number = float(value.reshape(-1)[0])
                state[name] = np.asarray(int(number) if np.isfinite(number) else 0,dtype=np.int64)
            else:
                state[name] = np.ascontiguousarray(value,dtype=np.float32)
        while pending:
            finish_matrix()
    encoder_state = {name.removeprefix("encoder."): state.pop(name)
                     for name in list(state) if name.startswith("encoder.")}
    if set(state) != {"encoder_projector.weight", "encoder_projector.bias"}:
        raise ValueError("Native projection checkpoint has missing or extra records")
    weight, bias = state["encoder_projector.weight"], state["encoder_projector.bias"]
    if weight.shape != (HF_CONFIG["decoder_hidden_size"],config.hidden_size) or bias.shape != (HF_CONFIG["decoder_hidden_size"],):
        raise ValueError("Incorrect native encoder projection shape")
    frontend = Frontend(mel_filters(),weight,bias,threads)
    encoder_state["encode_positions.inv_freq"] = frontend.positions()
    encoder = Encoder(lib,config,encoder_matrices,encoder_state)
    decoder = TDT(lib,decoder_matrices,decoder_biases,HF_CONFIG)
    return SimpleNamespace(native_encoder=encoder,native_decoder=decoder,frontend=frontend)


class Transcriber:
    """Transcribe entire recordings with the existing pause-based segmentation."""

    def __init__(self, directory, threads=2, engine="native"):
        if threads < 1 or engine != "native":
            raise ValueError("This experiment requires the native engine and positive threads")
        self.model = load_model(directory,threads)
        self.vocabulary = json.loads((directory / "config.json").read_text())["joint"]["vocabulary"]

    def segment_decode(self, audio):
        """Preserve input guards and timed token assembly for one model-sized segment."""
        audio = np.asarray(audio,dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio")
        if len(audio) == 0 or float(np.max(np.abs(audio))) < 1e-4:
            return "", []
        if len(audio) > 30*SAMPLE_RATE:
            raise ValueError("Use transcribe for audio longer than 30 seconds")
        if len(audio) < 160:
            audio = np.pad(audio,(0,160-len(audio)))
        features = self.model.frontend.features(audio)
        encoded = self.model.native_encoder.forward(features)
        projected = self.model.frontend.project(encoded)
        ids, frames, durations = self.model.native_decoder.decode(projected)
        tokens = []
        for token, frame, duration in zip(ids,frames,durations):
            piece = self.vocabulary[token]
            if piece in ("<unk>","<pad>") or (piece.startswith("<|") and piece.endswith("|>")):
                continue
            tokens.append((piece.replace("\u2581"," "),frame*8/SAMPLE_RATE*160,duration*8/SAMPLE_RATE*160))
        return "".join(piece for piece,_,_ in tokens).strip(), tokens

    def transcribe(self, audio):
        """Accept a full voice note, retaining upstream segment and word offsets."""
        audio = np.asarray(audio,dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio")
        if len(audio) <= 30*SAMPLE_RATE:
            text, tokens = self.segment_decode(audio)
            return {"text":text,"words":words_from_tokens(tokens,limit=len(audio)/SAMPLE_RATE)}
        words, pending = [], []

        def decode(segment):
            text, tokens = self.segment_decode(segment)
            pending[:] = tokens
            return text

        def on_segment(text, start_sample, n_samples):
            words.extend(words_from_tokens(pending,start_sample/SAMPLE_RATE,n_samples/SAMPLE_RATE))

        session = LiveSession(decode,partials=False,on_segment=on_segment)
        session.feed_pcm(audio)
        return {"text":session.finish(),"words":words}


def read_audio(path):
    """Read, mix and resample audio exactly as the current runtime does."""
    audio, rate = sf.read(path,dtype="float32",always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(rate,SAMPLE_RATE)
        audio = resample_poly(audio,SAMPLE_RATE//divisor,rate//divisor)
    return np.ascontiguousarray(audio,dtype=np.float32)


def main():
    """Run a complete file through the isolated native experimental path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("audio",type=Path,nargs="+")
    args = parser.parse_args()
    start = time.perf_counter()
    transcriber = Transcriber(args.model_dir,args.threads)
    print(f"Model loaded in {time.perf_counter()-start:.2f}s",file=sys.stderr,flush=True)
    for path in args.audio:
        audio = read_audio(path)
        start = time.perf_counter()
        result = transcriber.transcribe(audio)
        result.update(file=str(path),audio_seconds=len(audio)/SAMPLE_RATE,seconds=time.perf_counter()-start)
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
