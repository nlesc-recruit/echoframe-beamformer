#include <ccglib/helper.h>

#include <fstream>
#include <iostream>

#include "tcbf.h"

#ifndef COMPLEX
#define COMPLEX 2
#endif

static inline size_t align(size_t a, size_t b) { return b * ccglib::helper::ceildiv(a, b); }

namespace tcbf {
Beamformer::Beamformer(const size_t pixels, const size_t frames, const size_t samples, cu::Device &device,
                       cu::Stream &stream)
    : pixels_(pixels), frames_(frames), samples_(samples), device_(device), stream_(stream) {
  // padded sizes
  dim3 tile_sizes = ccglib::mma::GEMM::GetDimensions(kGEMMPrecision, kGEMMVariant);
  pixels_padded_ = align(pixels, tile_sizes.x);
  frames_padded_ = align(frames, tile_sizes.y);
  samples_padded_ = align(samples, tile_sizes.z);
  // array sizes on the device (i.e. padded)
  bytesAPacked_ = COMPLEX * pixels_padded_ * samples_padded_ / CHAR_BIT;
  bytesRF_ = COMPLEX * frames_padded_ * samples_padded_;
  bytesRFPacked_ = bytesRF_ / CHAR_BIT;
  bytesBF_ = COMPLEX * pixels_padded_ * frames_padded_ * sizeof(int);
  // allocate device memory
  d_A = std::make_unique<cu::DeviceMemory>(bytesAPacked_);
  d_RF = std::make_unique<cu::DeviceMemory>(bytesRF_);
  d_RF_packed = std::make_unique<cu::DeviceMemory>(bytesRFPacked_);
  d_RF_transposed = std::make_unique<cu::DeviceMemory>(bytesRFPacked_);
  d_BF = std::make_unique<cu::DeviceMemory>(bytesBF_);
  // create objects to run kernels
  pack_rf_ = std::make_unique<ccglib::packing::Packing>(COMPLEX * frames_padded_ * samples_padded_, device_, stream_);
  transpose_rf_ = std::make_unique<ccglib::transpose::Transpose>(
      kBatchSize, frames_padded_, samples_padded_, kGEMMTileSize.y, kGEMMTileSize.z, kBitsPerSample, device_, stream_);
  gemm_ = std::make_unique<ccglib::mma::GEMM>(kBatchSize, pixels_padded_, frames_padded_, samples_padded_,
                                              kBitsPerSample, device_, stream_, kGEMMPrecision, kGEMMVariant,
                                              ccglib::mma::complex_middle, ccglib::mma::col_major);
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
}

void Beamformer::read_RF(cu::HostMemory &RF, const std::string path) {
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  in.read(static_cast<char *>(RF), COMPLEX * frames_ * samples_);
}

void Beamformer::write_BF(cu::HostMemory &BF, const std::string path) {
  std::ofstream out(path, std::ios::binary | std::ios::out);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + path);
  }
  out.write(static_cast<char *>(BF), COMPLEX * frames_ * pixels_ * sizeof(unsigned));
}

void Beamformer::RF_to_device(cu::HostMemory &RF) {
  // transfer in chunks to handle padding
  // RF shape is frames(padded) * samples(padded) * complex
  for (size_t f = 0; f < frames_; f++) {
    // get objects pointing to start of chunk to transfer
    // factors of 2 are for complex
    const size_t d_offset = f * samples_padded_ * 2;
    const size_t offset = f * samples_ * 2;
    const size_t bytes_to_transfer = samples_ * 2;
    cu::DeviceMemory d_RF_chunk(*d_RF, d_offset, bytes_to_transfer);
    stream_.memcpyHtoDAsync(d_RF_chunk, static_cast<char *>(RF) + offset, bytes_to_transfer);
  }
}

void Beamformer::BF_to_host(cu::HostMemory &BF) {
  // transfer in chunks to handle padding
  // BF shape is complex * frames(padded) * pixels(padded) * sizeof(unsigned)
  for (size_t c = 0; c < COMPLEX; c++) {
    for (size_t f = 0; f < frames_; f++) {
      // get objects pointing to start of chunk to transfer
      const size_t d_offset = (c * frames_padded_ * pixels_padded_ + f * pixels_padded_) * sizeof(unsigned);
      const size_t offset = (c * frames_ * pixels_ + f * pixels_) * sizeof(unsigned);
      const size_t bytes_to_transfer = pixels_ * sizeof(unsigned);
      cu::DeviceMemory d_BF_chunk(*d_BF, d_offset, bytes_to_transfer);
      stream_.memcpyDtoHAsync(static_cast<char *>(BF) + offset, d_BF_chunk, bytes_to_transfer);
    }
  }
}

void Beamformer::process(cu::HostMemory &RF, cu::HostMemory &BF) {
  // transfer RF to GPU
  RF_to_device(RF);
  // pack bits
  pack_rf_->Run(*d_RF, *d_RF_packed, ccglib::packing::pack, ccglib::packing::complex_last);
  // transpose to format required by GEMM
  transpose_rf_->Run(*d_RF_packed, *d_RF_transposed);
  // do GEMM
  gemm_->Run(*d_A, *d_RF_transposed, *d_BF);
  // transfer BF to host and sync
  BF_to_host(BF);
  stream_.synchronize();
}

}  // namespace tcbf
