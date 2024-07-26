#include <ccglib/helper.h>
#include <limits.h>
#include <tcbf.h>

#include <ccglib/ccglib.hpp>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>

inline size_t align(size_t a, size_t b) { return b * ccglib::helper::ceildiv(a, b); }

cxxopts::Options create_commandline_parser(const char *argv[]) {
  cxxopts::Options options(argv[0], "Echoframe beamformer A matrix preparation");

  options.add_options()("a_matrix_in", "Full path to input A matrix", cxxopts::value<std::string>())(
      "a_matrix_out", "Full path to output A matrix", cxxopts::value<std::string>())(
      "pixels", "Number of pixels", cxxopts::value<size_t>())("samples", "Number of samples", cxxopts::value<size_t>())(
      "device", "GPU device ID", cxxopts::value<unsigned>()->default_value(std::to_string(0)))("h,help", "Print help");

  return options;
}

cxxopts::ParseResult parse_commandline(int argc, const char *argv[]) {
  cxxopts::Options options = create_commandline_parser(argv);

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(EXIT_SUCCESS);
    }

    std::vector<std::string> required_options{"a_matrix_in", "a_matrix_out", "pixels", "samples"};
    for (auto &opt : required_options) {
      if (!result.count(opt)) {
        std::cerr << "Required argument missing: " << opt << std::endl;
        std::cerr << "Run " << argv[0] << " -h for help" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    return result;
  } catch (const cxxopts::exceptions::exception &err) {
    std::cerr << "Error parsing commandline: " << err.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, const char *argv[]) {
  cxxopts::ParseResult cmdline = parse_commandline(argc, argv);
  const std::string path_a_matrix_in = cmdline["a_matrix_in"].as<std::string>();
  const std::string path_a_matrix_out = cmdline["a_matrix_out"].as<std::string>();
  const size_t pixels = cmdline["pixels"].as<size_t>();
  const size_t samples = cmdline["samples"].as<size_t>();
  const unsigned device_id = cmdline["device"].as<unsigned>();
  const size_t complex = 2;

  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  // tile size in beams, frames, samples axes
  dim3 tile_sizes = ccglib::mma::GEMM::GetDimensions(ccglib::mma::int1, ccglib::mma::opt);

  const size_t pixels_padded = align(pixels, tile_sizes.x);
  const size_t samples_padded = align(samples, tile_sizes.z);

  // factor 2 for complex
  const size_t bytes_a_matrix = complex * pixels_padded * samples_padded;
  const size_t bytes_a_matrix_packed = bytes_a_matrix / CHAR_BIT;

  // Read data from disk
  // row-by-row to handle padding
  cu::HostMemory a_matrix_host(bytes_a_matrix);
  std::ifstream in(path_a_matrix_in, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path_a_matrix_in);
  }
  for (size_t c = 0; c < complex; c++) {
    for (size_t pixel = 0; pixel < pixels; pixel++) {
      in.read(static_cast<char *>(a_matrix_host) + c * pixels_padded * samples_padded + pixel * samples_padded,
              samples);
    }
  }
  in.close();

  // conjugate
  std::cout << "Conjugate" << std::endl;
#pragma omp parallel for collapse(2)
  for (size_t pixel = 0; pixel < pixels; pixel++) {
    for (size_t sample = 0; sample < samples; sample++) {
      const size_t idx = pixels_padded * samples_padded + pixel * samples_padded + sample;
      static_cast<char *>(a_matrix_host)[idx] = 1 - static_cast<char *>(a_matrix_host)[idx];
    }
  }

  // Device memory for output packed data
  cu::DeviceMemory d_a_matrix_packed(bytes_a_matrix_packed);
  d_a_matrix_packed.zero(bytes_a_matrix_packed);
  // Device memory for transposed data
  cu::DeviceMemory d_a_transposed(bytes_a_matrix_packed);

  // chunk of input data on device in case it doesn't fit in GPU memory
  // get available GPU memory (after allocating other device memory)
  // use at most 80% of available memory
  size_t bytes_per_chunk = .8 * context.getFreeMemory();
  // packing kernel uses at most 1024 threads per block (and should be a power of 2), each thread processes one byte
  // round to multiple of a kilobyte such that it correspond to a whole number of blocks
  bytes_per_chunk = 1024 * (bytes_per_chunk / 1024);
  if (bytes_per_chunk > bytes_a_matrix) {
    bytes_per_chunk = bytes_a_matrix;
  }
  cu::DeviceMemory d_a_chunk(bytes_per_chunk);
  d_a_chunk.zero(bytes_per_chunk);

  // process, complex-first for now
  std::cout << "Packing" << std::endl;
  for (size_t byte_start = 0; byte_start < bytes_a_matrix; byte_start += bytes_per_chunk) {
    size_t local_nbytes = bytes_per_chunk;
    // correct nbytes in last chunk
    if (byte_start + local_nbytes > bytes_a_matrix) {
      local_nbytes = bytes_a_matrix - byte_start;
      // ensure any padded region is set to zero
      d_a_chunk.zero(bytes_per_chunk);
    }
    // copy chunk to device
    stream.memcpyHtoDAsync(d_a_chunk, static_cast<char *>(a_matrix_host) + byte_start, local_nbytes);
    // get device memory slice for this chunk in a_packed
    cu::DeviceMemory d_a_packed_chunk(d_a_matrix_packed, byte_start / CHAR_BIT, local_nbytes / CHAR_BIT);
    // run packing kernel
    ccglib::packing::Packing packing(local_nbytes, device, stream);
    packing.Run(d_a_chunk, d_a_packed_chunk, ccglib::packing::pack, ccglib::packing::complex_first);
  }

  // transpose
  std::cout << "Transpose" << std::endl;
  ccglib::transpose::Transpose transpose(1, pixels_padded, samples_padded, tile_sizes.x, tile_sizes.z, 1, device,
                                         stream);
  transpose.Run(d_a_matrix_packed, d_a_transposed);

  // copy output to host
  std::cout << "Copy to host" << std::endl;
  cu::HostMemory a_matrix_output(bytes_a_matrix_packed);
  stream.memcpyDtoHAsync(a_matrix_output, d_a_transposed, bytes_a_matrix_packed);
  stream.synchronize();

  // write to disk
  std::cout << "Write to disk" << std::endl;
  std::ofstream out(path_a_matrix_out, std::ios::binary | std::ios::out);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + path_a_matrix_out);
  }
  out.write(static_cast<char *>(a_matrix_output), bytes_a_matrix_packed);
}
