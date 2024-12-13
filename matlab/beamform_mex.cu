#include "mex.h"
#include "tcbf.h"  // include the header

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 7) {
    mexErrMsgIdAndTxt("beamform:InvalidInput",
                      "Seven inputs required: a_matrix, rf, bf, pixels, frames, samples, device_id");
  }

  std::string path_a_matrix = mxArrayToString(prhs[0]);
  std::string path_rf = mxArrayToString(prhs[1]);
  std::string path_bf = mxArrayToString(prhs[2]);
  size_t pixels = mxGetScalar(prhs[3]);
  size_t frames = mxGetScalar(prhs[4]);
  size_t samples = mxGetScalar(prhs[5]);
  unsigned device_id = mxGetScalar(prhs[6]);

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

  mwSize dims[2] = {2 * pixels, frames};
  mxArray *outArray = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  int32_t *outData = static_cast<int32_t *>(mxGetData(outArray));

  // Copy the data
  std::memcpy(outData, BF, totalElements * sizeof(int32_t));

  // Assign output
  plhs[0] = outArray;

  int status = 0;

  if (nlhs > 1) {
    plhs[1] = mxCreateDoubleScalar((double)status);
  }
}
