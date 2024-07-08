#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits.h>

#include <ccglib/ccglib.hpp>
#include <cudawrappers/cu.hpp>

template <typename T> inline T align(const T a, const T b) {
  return b * (a / b + ((a % b) != 0));
}

template <typename T>
void read_file(const std::string path, char *data, const size_t M,
               const size_t N, const size_t M_padded, const size_t N_padded) {
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }

  for (size_t m = 0; m < M; m++) {
    const size_t byte_offset = m * N_padded * sizeof(T);

    in.read(data + byte_offset, N * sizeof(T));
  }
}

void read_file(const std::string path, cu::HostMemory &data,
               const size_t bytes) {
  std::ifstream in(path, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }

  in.read(static_cast<char *>(data), bytes);
}

template <typename T>
void write_file(const std::string path, char *data, const size_t M,
                const size_t N, const size_t M_padded, const size_t N_padded) {
  std::ofstream out(path, std::ios::binary | std::ios::out);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + path);
  }

  // real part
  for (size_t m = 0; m < M; m++) {
    const size_t byte_offset = m * N_padded * sizeof(T);
    out.write(data + byte_offset, N * sizeof(T));
  }
  // imag part
  for (size_t m = M_padded; m < M_padded + M; m++) {
    const size_t byte_offset = m * N_padded * sizeof(T);
    out.write(data + byte_offset, N * sizeof(T));
  }
}

int main() {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  const std::string path = "/var/scratch/oostrum/cube_data/gemm/sign_demo/";

  const unsigned complex = 2;
  const size_t beams_data = 38880;    // M axis
  const size_t frames_data = 8041;    // N axis
  const size_t samples_data = 524288; // K axis
  const size_t nr_bits = 1;

  // initialize all ccglib objects -- this triggers compilation as well
  const auto gemm_precision = ccglib::mma::int1;
  const auto gemm_variant = ccglib::mma::opt;
  // obtain matrix tile sizes from GEMM, order is M, N, K
  const dim3 dimensions =
      ccglib::mma::GEMM::GetDimensions(gemm_precision, gemm_variant);

  // for ccglib only supports matrices that are multiples of the GEMM
  // dimensions, take care of padding in this code for now const size_t beams =
  // align(beams_data, static_cast<size_t>(dimensions.x)); const size_t frames =
  // align(frames_data,  static_cast<size_t>(dimensions.y)); const size_t
  // samples = align(samples_data,  static_cast<size_t>(dimensions.z));
  constexpr size_t beams = 38912;
  constexpr size_t frames = 8064;
  constexpr size_t samples = 524288;
  std::cout << beams_data << " " << frames_data << " " << samples_data
            << std::endl;
  std::cout << beams << " " << frames << " " << samples << std::endl;

  const size_t bytes_a_matrix = beams * samples * complex;
  const size_t bytes_rf = frames * samples * complex;
  const size_t bytes_a_matrix_packed = bytes_a_matrix / CHAR_BIT;
  const size_t bytes_rf_packed = bytes_rf / CHAR_BIT;

  ccglib::packing::Packing pack_rf(frames * samples, device, stream);
  ccglib::transpose::Transpose transpose_rf(
      1, frames, samples, dimensions.y, dimensions.z, nr_bits, device, stream);
  ccglib::mma::GEMM gemm(1, beams, frames, samples, nr_bits, device, stream,
                         gemm_precision, gemm_variant);

  // the inputs are sign data stored as one byte per sample
  // read prepared A matrix
  cu::HostMemory A(bytes_a_matrix_packed);
  read_file(path + "A_packed_transposed_64_256.bin", A, bytes_a_matrix_packed);
  cu::DeviceMemory d_A_matrix_trans(bytes_a_matrix_packed);
  stream.memcpyHtoDAsync(d_A_matrix_trans, A, bytes_a_matrix_packed);

  auto RF_real = new char[frames][samples];
  auto RF_imag = new char[frames][samples];

  std::cout << "Loading RF - real" << std::endl;
  read_file<char>(path + "RF_Real_524288_8041.bin", &RF_real[0][0], frames_data,
                  samples_data, frames, samples);
  std::cout << "Loading RF - imag" << std::endl;
  read_file<char>(path + "RF_Imag_524288_8041.bin", &RF_imag[0][0], frames_data,
                  samples_data, frames, samples);
  std::cout << "Data loaded" << std::endl;

  // step 1. packing
  std::cout << "Packing RF" << std::endl;
  cu::DeviceMemory d_RF(bytes_rf_packed);
  // get a pointer to start of RF imag
  CUdeviceptr RF_offset = reinterpret_cast<CUdeviceptr>(
      reinterpret_cast<char *>(static_cast<CUdeviceptr>(d_RF)) +
      bytes_rf_packed / complex);
  cu::DeviceMemory d_RF_imag_only(RF_offset, bytes_rf_packed / complex);

  cu::HostMemory RF_imag_host(RF_imag, bytes_rf / complex);
  cu::HostMemory RF_real_host(RF_real, bytes_rf / complex);
  pack_rf.Run(RF_real_host, d_RF, ccglib::packing::pack);
  pack_rf.Run(RF_imag_host, d_RF_imag_only, ccglib::packing::pack);

  // step 2. transpose
  std::cout << "Transpose RF" << std::endl;
  cu::DeviceMemory d_RF_trans(d_RF.size());
  transpose_rf.Run(d_RF, d_RF_trans);

  // step 3. beamform
  std::cout << "GEMM" << std::endl;
  cu::DeviceMemory d_BF(beams * frames * complex * sizeof(int));
  gemm.Run(d_A_matrix_trans, d_RF_trans, d_BF);

  cu::HostMemory BF(beams * frames * complex * sizeof(int));
  stream.memcpyDtoHAsync(BF, d_BF, beams * frames * complex * sizeof(int));
  stream.synchronize();

  const std::string output_file = path + "BF.bin";
  std::cout << "Writing output to " << output_file << std::endl;
  // row major
  // write_file<int>(output_file, BF, beams_data, frames_data, beams, frames);
  // col major
  write_file<int>(output_file, static_cast<char *>(BF), frames_data, beams_data,
                  frames, beams);
}