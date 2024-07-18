#include <cxxopts.hpp>
#include <fstream>
#include <iostream>

#include "tcbf.h"

cxxopts::Options create_commandline_parser(const char *argv[]) {
  cxxopts::Options options(argv[0], "Echoframe standalone beamformer");

  options.add_options()("a_matrix", "Full path to input prepared A matrix",
                        cxxopts::value<std::string>())(
      "rf", "Full path to input RF", cxxopts::value<std::string>())(
      "bf", "Full path to output BF", cxxopts::value<std::string>())(
      "pixels", "Number of pixels", cxxopts::value<size_t>())(
      "frames", "Number of frames", cxxopts::value<size_t>())(
      "samples", "Number of samples", cxxopts::value<size_t>())(
      "device", "GPU device ID",
      cxxopts::value<unsigned>()->default_value(std::to_string(0)))(
      "h,help", "Print help");

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

    std::vector<std::string> required_options{"a_matrix", "rf",     "bf",
                                              "pixels",   "frames", "samples"};
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
  const std::string path_a_matrix = cmdline["a_matrix"].as<std::string>();
  const std::string path_rf = cmdline["rf"].as<std::string>();
  const std::string path_bf = cmdline["bf"].as<std::string>();
  const size_t pixels = cmdline["pixels"].as<size_t>();
  const size_t frames = cmdline["frames"].as<size_t>();
  const size_t samples = cmdline["samples"].as<size_t>();
  const unsigned device_id = cmdline["device"].as<unsigned>();

  cu::init();
  cu::Device device(device_id);
  cu::Context context(CU_CTX_BLOCKING_SYNC, device);
  cu::Stream stream;

  tcbf::Beamformer beamformer(pixels, frames, samples, device, stream);
  cu::HostMemory RF(2 * frames * samples);
  cu::HostMemory BF(2 * pixels * frames * sizeof(unsigned));

  beamformer.read_A_matrix(path_a_matrix);
  beamformer.read_RF(RF, path_rf);
  beamformer.process(RF, BF);
  beamformer.write_BF(BF, path_bf);
}