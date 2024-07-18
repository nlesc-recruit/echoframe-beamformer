[![github url](https://img.shields.io/badge/github-url-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/nlesc-recruit/echoframe-beamformer)
[![github license badge](https://img.shields.io/github/license/nlesc-recruit/echoframe-beamformer)](https://github.com/nlesc-recruit/echoframe-beamformer)
[![CI status](https://github.com/nlesc-recruit/echoframe-beamformer/actions/workflows/build.yml/badge.svg)](https://github.com/nlesc-recruit/echoframe-beamformer/actions/workflows/build.yml)

# Echoframe tensor core beamformer

## Requirements

| Software | Minimum version |
| -------- | --------------- |
| CUDA | 11.0|
| CMake | 3.20 |

| Hardware | Type |
| -------- | ---- |
| GPU | NVIDIA GPU with 1-bit tensor cores (Turing-generation or later)|

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
