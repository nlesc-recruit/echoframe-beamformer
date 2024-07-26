#include <ccglib/helper.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <tcbf.h>

#include <ccglib/ccglib.hpp>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>

__global__ void conjugate_1bit(unsigned *data, const size_t n_bytes) {
  size_t tid = threadIdx.x + blockDim.x * static_cast<size_t>(blockIdx.x);
  const size_t n_elements = n_bytes / sizeof(unsigned);
  if (tid >= n_elements) {
    return;
  }
  data[tid] = ~data[tid];
}

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

  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  // tile size in beams, frames, samples axes
  dim3 tile_sizes = ccglib::mma::GEMM::GetDimensions(ccglib::mma::int1, ccglib::mma::opt);

  const size_t pixels_padded = align(pixels, tile_sizes.x);
  const size_t samples_padded = align(samples, tile_sizes.z);

  // factor 2 for complex
  // host is unpadded, device is always padded
  const size_t bytes_a_matrix = 2UL * pixels * samples;
  const size_t bytes_a_matrix_packed = 2UL * pixels_padded * samples_padded / CHAR_BIT;

  // Read data from disk
  cu::HostMemory a_matrix_host(bytes_a_matrix);
  std::ifstream in(path_a_matrix_in, std::ios::binary | std::ios::in);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path_a_matrix_in);
  }
  in.read(static_cast<char *>(a_matrix_host), bytes_a_matrix);
  in.close();

  // Device memory for output packed data
  cu::DeviceMemory d_a_matrix_packed(bytes_a_matrix_packed);
  d_a_matrix_packed.zero(bytes_a_matrix_packed);
  // Device memory for transposed data
  cu::DeviceMemory d_a_transposed(bytes_a_matrix_packed);

  // chunk of input data on device in case it doesn't fit in GPU memory
  // get available GPU memory (after allocating other device memory)
  // use at most 80% of available memory
  size_t chunk_size = .8 * context.getFreeMemory();
  size_t pixels_per_chunk = chunk_size / (samples_padded);
  if (pixels_per_chunk > pixels) {
    pixels_per_chunk = pixels;
  }
  chunk_size = pixels_per_chunk * samples_padded;
  cu::DeviceMemory d_a_chunk(chunk_size);
  d_a_chunk.zero(chunk_size);

  // process, complex-first for now
  // first real, then imag part
  std::cout << "Start of processing" << std::endl;
  std::cout << "Packing" << std::endl;
  for (size_t c = 0; c < 2; c++) {
    const size_t complex_offset_host = c * pixels * samples;
    const size_t complex_offset_device_packed = c * pixels_padded * samples_padded / CHAR_BIT;
    // process chunks
    for (size_t pixel_start = 0; pixel_start < pixels; pixel_start += pixels_per_chunk) {
      size_t local_npixels = pixels_per_chunk;
      // correct npixels in last chunk
      if (pixel_start + local_npixels > pixels) {
        local_npixels = pixels - pixel_start;
        // ensure any padded region is set to zero
        d_a_chunk.zero(chunk_size);
      }
      // copy chunk to device, row-by-row to handle padding
      for (size_t pixel = 0; pixel < local_npixels; pixel++) {
        const size_t d_offset = pixel * samples_padded;
        const size_t offset = (pixel_start + pixel) * samples + complex_offset_host;
        const size_t bytes_to_transfer = samples;

        cu::DeviceMemory d_a_chunk_slice(d_a_chunk, d_offset, bytes_to_transfer);
        stream.memcpyHtoDAsync(d_a_chunk_slice, static_cast<char *>(a_matrix_host) + offset, bytes_to_transfer);
      }
      // get offset for this chunk in a_packed
      cu::DeviceMemory d_a_packed_chunk(d_a_matrix_packed,
                                        pixel_start * samples_padded / CHAR_BIT + complex_offset_device_packed,
                                        local_npixels * samples_padded / CHAR_BIT);
      // run packing kernel
      ccglib::packing::Packing packing(local_npixels * samples_padded, device, stream);
      packing.Run(d_a_chunk, d_a_packed_chunk, ccglib::packing::pack, ccglib::packing::complex_first);
    }
  }

  // conjugate
  std::cout << "Conjugate" << std::endl;
  dim3 threads(256);
  dim3 grid(ccglib::helper::ceildiv(bytes_a_matrix_packed / 2, threads.x));
  cu::DeviceMemory d_a_matrix_packed_imag(d_a_matrix_packed, bytes_a_matrix_packed / 2, bytes_a_matrix_packed / 2);
  conjugate_1bit<<<grid, threads, 0, stream>>>(
      reinterpret_cast<unsigned *>(static_cast<CUdeviceptr>(d_a_matrix_packed_imag)), bytes_a_matrix_packed / 2);

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
