#include <fstream>
#include <iostream>

#include <ccglib/helper.h>
#include <tcbf.h>

inline size_t align(size_t a, size_t b) {
  return b * ccglib::helper::ceildiv(a, b);
}

int main() {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t pixels_data = 38880;
  const size_t frames_data = 8041;
  const size_t samples_data = 524288;

  dim3 tile_sizes =
      ccglib::mma::GEMM::GetDimensions(ccglib::mma::int1, ccglib::mma::opt);

  const size_t pixels = align(pixels_data, tile_sizes.x);
  const size_t frames = align(frames_data, tile_sizes.y);
  const size_t samples = align(samples_data, tile_sizes.z);

  tcbf::Beamformer beamformer(pixels, frames, samples, device, stream);
  cu::HostMemory RF(beamformer.bytesRF_);
  cu::HostMemory BF(beamformer.bytesBF_);

  beamformer.read_A_matrix("/var/scratch/oostrum/cube_data/gemm/sign_demo/"
                           "A_packed_transposed_conjugated_64_256.bin");
  beamformer.read_RF(
      RF,
      "/var/scratch/oostrum/cube_data/gemm/sign_demo/RF_full_524288_8041.bin",
      frames_data, samples_data);
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