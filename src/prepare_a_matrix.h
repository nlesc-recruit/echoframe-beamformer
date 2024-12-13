#ifndef PREPARE_A_MATRIX_H
#define PREPARE_A_MATRIX_H

#include <string>
#include <cstddef> // for size_t

int prepareAMatrix(const std::string &path_a_matrix_in,
                   const std::string &path_a_matrix_out,
                   size_t pixels,
                   size_t samples,
                   unsigned device_id);

#endif // PREPARE_A_MATRIX_H
