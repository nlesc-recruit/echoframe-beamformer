#include "../src/prepare_a_matrix.h"  // include the header
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 5) {
    mexErrMsgIdAndTxt("beamformer_mex:InvalidInput",
                      "Five inputs required: a_matrix_in, a_matrix_out, pixels, samples, device_id");
  }

  // Extract arguments from MATLAB inputs
  char *a_matrix_in_c = mxArrayToString(prhs[0]);
  char *a_matrix_out_c = mxArrayToString(prhs[1]);
  double pixels_d = mxGetScalar(prhs[2]);
  double samples_d = mxGetScalar(prhs[3]);
  double device_id_d = mxGetScalar(prhs[4]);

  std::string a_matrix_in(a_matrix_in_c);
  std::string a_matrix_out(a_matrix_out_c);
  size_t pixels = static_cast<size_t>(pixels_d);
  size_t samples = static_cast<size_t>(samples_d);
  unsigned device_id = static_cast<unsigned>(device_id_d);

  mxFree(a_matrix_in_c);
  mxFree(a_matrix_out_c);

  int status = prepareAMatrix(a_matrix_in, a_matrix_out, pixels, samples, device_id);

  if (nlhs > 0) {
    plhs[0] = mxCreateDoubleScalar(static_cast<double>(status));
  }
}
