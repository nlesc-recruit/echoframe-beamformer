# Echoframe tensor core beamformer

## Requirements

| Software | Minimum version |
| -------- | --------------- |
| CUDA | 10.0|
| CMake | 3.20 |

| Hardware | Type |
| -------- | ---- |
| GPU | Ampere-generation NVIDIA GPU or later |

## Installation
This project uses CMake. To build the library:
```shell
git clone https://github.com/nlesc-recruit/echoframe-beamformer
cd echoframe-beamformer
cmake -S . -B build
make -C build
```

This will create executables and a `libtcbf` library in the build directory.

## Usage

### Standalone beamformer
The standalone beamformer executable is `echoframe-beamformer`. It can read a prepared A matrix and raw RF from disk, process it on the GPU, and store the BF back to disk. Example commandline:

`echoframe-beamformer --a_matrix A_matrix.bin --rf RF.bin --bf BF.bin --pixels 4096 --frames 8192 --samples 16384`