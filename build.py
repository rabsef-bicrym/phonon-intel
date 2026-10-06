"""Build the retained Intel candidate using the installed PyTorch CPU library."""

from pathlib import Path
import platform
import subprocess
import sysconfig

HERE = Path(__file__).resolve().parent


def main():
    """Build native operators and compact kernels without importing Python torch."""
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        raise SystemExit("This candidate targets Intel macOS with AVX2 and F16C.")
    torch = Path(sysconfig.get_paths()["purelib"]) / "torch"
    if not (torch / "include/ATen/ATen.h").is_file() or not (torch / "lib/libtorch_cpu.dylib").is_file():
        raise SystemExit("The existing Intel PyTorch installation must include ATen headers and CPU libraries.")
    common = ["xcrun", "clang++", "-std=c++17", "-O3", "-mavx2", "-ffp-contract=off",
              "-Wall", "-Wextra", "-dynamiclib"]
    subprocess.run(common + ["-isystem", str(torch / "include"),
                             "-isystem", str(torch / "include/torch/csrc/api/include"),
                             str(HERE / "frontend.cpp"), "-L" + str(torch / "lib"), "-ltorch_cpu", "-lc10",
                             "-Wl,-rpath," + str(torch / "lib"), "-o", str(HERE / "frontend.dylib")], check=True)
    # Keep the measured layout and compiler arithmetic contract fixed. The
    # frontend is separate so this kernel does not acquire an ATen dependency.
    root = HERE / "intel_macos"
    output = root / "build"
    output.mkdir(exist_ok=True)
    subprocess.run(common + ["-mfma", "-mf16c", "-Werror", "-DLUT_TRITS=5", "-DLUT_GROUPS=8",
                             "-DLUT_BATCH=32", "-DLUT_ROW_MAJOR=2", "-DLUT_EXPANSIONS_ONLY=1",
                             str(root / "packed.cpp"), str(root / "encoder.cpp"),
                             "-framework", "Accelerate", "-o", str(output / "libphonon_intel.dylib")], check=True)


if __name__ == "__main__":
    main()
