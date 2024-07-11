#include "tcbf.h"
#include <fstream>

#include <iostream>

namespace tcbf {
Beamformer::Beamformer(const size_t pixels, const size_t frames,
                       const size_t samples, cu::Device &device,
                       cu::Stream &stream)
    : pixels_(pixels), frames_(frames), samples_(samples), device_(device),
      stream_(stream) {
  // array sizes, factor 2 is for complex axis
  bytesAPacked_ = 2 * pixels_ * samples_ / CHAR_BIT;
  bytesRF_ = 2 * frames_ * samples_;
  bytesRFPacked_ = bytesRF_ / CHAR_BIT;
  bytesBF_ = 2 * pixels_ * frames_ * sizeof(int);
  // allocate device memory
  d_A = std::make_unique<cu::DeviceMemory>(bytesAPacked_);
  d_RF = std::make_unique<cu::DeviceMemory>(bytesRF_);
  d_RF_packed = std::make_unique<cu::DeviceMemory>(bytesRFPacked_);
  d_RF_transposed = std::make_unique<cu::DeviceMemory>(bytesRFPacked_);
  d_BF = std::make_unique<cu::DeviceMemory>(bytesBF_);
  // create objects to run kernels
  pack_rf_ = std::make_unique<ccglib::packing::Packing>(2 * frames_ * samples_,
                                                        device_, stream_);
  transpose_rf_ = std::make_unique<ccglib::transpose::Transpose>(
      kBatchSize, frames_, samples_, kGEMMTileSize.y, kGEMMTileSize.z,
      kBitsPerSample, device_, stream_);
  gemm_ = std::make_unique<ccglib::mma::GEMM>(
      kBatchSize, pixels_, frames_, samples_, kBitsPerSample, device_, stream_,
      kGEMMPrecision, kGEMMVariant);
}

void Beamformer::read_A_matrix(const std::string path) {
  // Read an already padded, packed, transposed, conjugated A matrix
  cu::HostMemory A(bytesAPacked_);

  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  in.read(static_cast<char *>(A), bytesAPacked_);

  stream_.memcpyHtoDAsync(*d_A, A, bytesAPacked_);
  stream_.synchronize();
}

void Beamformer::read_RF(cu::HostMemory &RF, const std::string path,
                         const size_t frames_data, const size_t samples_data) {
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  // read row-by-row to handle padding
  for (size_t frame = 0; frame < frames_data; frame++) {
    in.read(static_cast<char *>(RF) + frame * samples_, samples_data);
  }
  for (size_t frame = frames_; frame < frames_ + frames_data; frame++) {
    in.read(static_cast<char *>(RF) + frame * samples_, samples_data);
  }
}

void Beamformer::process(cu::HostMemory &RF, cu::HostMemory &BF) {
  // transfer RF to GPU
  stream_.memcpyHtoDAsync(*d_RF, RF, bytesRF_);
  // pack bits
  pack_rf_->Run(*d_RF, *d_RF_packed, ccglib::packing::pack);
  // transpose to format required by GEMM
  transpose_rf_->Run(*d_RF_packed, *d_RF_transposed);
  // do GEMM
  gemm_->Run(*d_A, *d_RF_transposed, *d_BF);
  // transfer BF to host and sync
  stream_.memcpyDtoHAsync(BF, *d_BF, bytesBF_);
  stream_.synchronize();
}

} // namespace tcbf