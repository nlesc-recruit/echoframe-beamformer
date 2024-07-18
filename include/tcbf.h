#ifndef TCBF_H_
#define TCBF_H_

#include <ccglib/ccglib.hpp>
#include <cudawrappers/cu.hpp>
#include <limits.h>

namespace tcbf {

class Beamformer {
public:
  Beamformer(const size_t pixels, const size_t frames, const size_t samples,
             cu::Device &device, cu::Stream &stream);

  // static void prepare_A_matrix(const std::string input_path, const
  // std::string output_path, const size_t pixels, const size_t samples,
  //                              const );
  void read_A_matrix(const std::string path);
  void read_RF(cu::HostMemory &RF, const std::string path);
  void process(cu::HostMemory &RF, cu::HostMemory &BF);

private:
  void RF_to_device(cu::HostMemory &RF);
  void BF_to_host(cu::HostMemory &BF);

  static const size_t kBitsPerSample{1};
  static const size_t kBatchSize{1};
  static const ccglib::mma::Precision kGEMMPrecision{ccglib::mma::int1};
  static const ccglib::mma::Variant kGEMMVariant{ccglib::mma::opt};
  const dim3 kGEMMTileSize{
      ccglib::mma::GEMM::GetDimensions(kGEMMPrecision, kGEMMVariant)};

  std::unique_ptr<cu::DeviceMemory> d_A;
  std::unique_ptr<cu::DeviceMemory> d_RF;
  std::unique_ptr<cu::DeviceMemory> d_RF_complex_first;
  std::unique_ptr<cu::DeviceMemory> d_RF_packed;
  std::unique_ptr<cu::DeviceMemory> d_RF_transposed;
  std::unique_ptr<cu::DeviceMemory> d_BF;

  std::unique_ptr<ccglib::packing::Packing> pack_rf_;
  std::unique_ptr<ccglib::transpose::Transpose> transpose_rf_;
  std::unique_ptr<ccglib::mma::GEMM> gemm_;

  size_t pixels_;
  size_t pixels_padded_;
  size_t frames_;
  size_t frames_padded_;
  size_t samples_;
  size_t samples_padded_;

  size_t bytesRF_;
  size_t bytesBF_;
  size_t bytesAPacked_;
  size_t bytesRFPacked_;

  cu::Device &device_;
  cu::Stream &stream_;
};

} // namespace tcbf

#endif // TCBF_H_