#include <fstream>
#include <iostream>

#include "tcbf.h"

int main() {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t pixels = 38880;
  const size_t frames = 8041;
  const size_t samples = 524288;

  tcbf::Beamformer beamformer(pixels, frames, samples, device, stream);
  cu::HostMemory RF(2 * frames * samples);
  cu::HostMemory BF(2 * pixels * frames * sizeof(unsigned));

  beamformer.read_A_matrix("/var/scratch/oostrum/cube_data/gemm/sign_demo/"
                           "A_packed_transposed_conjugated_64_256.bin");
  beamformer.read_RF(
      RF,
      "/var/scratch/oostrum/cube_data/gemm/sign_demo/RF_full_524288_8041.bin");

  beamformer.process(RF, BF);

  std::ofstream out("/var/scratch/oostrum/cube_data/gemm/sign_demo/BF.bin",
                    std::ios::binary | std::ios::out);
  if (!out) {
    throw std::runtime_error("Failed to open output file");
  }

  // real part
  for (size_t frame = 0; frame < frames_data; frame++) {
    out.write(static_cast<char *>(BF) + frame * pixels * sizeof(int),
              pixels_data * sizeof(int));
  }
  // imag part
  for (size_t frame = frames; frame < frames + frames_data; frame++) {
    out.write(static_cast<char *>(BF) + frame * pixels * sizeof(int),
              pixels_data * sizeof(int));
  }
}