#include "omp/morozov_e_lineare_image_filtering_block_gaussian/include/ops_omp.hpp"

#include <cmath>
#include <vector>

bool morozov_e_lineare_image_filtering_block_gaussian_omp::TestTaskOpenMP::PreProcessingImpl() {
  // Init value for input and output
  n_ = static_cast<int>(task_data->inputs_count[0]);
  m_ = static_cast<int>(task_data->inputs_count[1]);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (m_ * n_));
  res_ = std::vector<double>(n_ * m_, 0);
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_omp::TestTaskOpenMP::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[1] == task_data->outputs_count[1] && task_data->inputs_count[1] > 0;
}

bool morozov_e_lineare_image_filtering_block_gaussian_omp::TestTaskOpenMP::RunImpl() {
  // Ядро Гаусса 3x3
  // clang-format off
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16},
      {2.0 / 16, 4.0 / 16, 2.0 / 16},
      {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  // clang-format on
#pragma omp parallel for
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      if (i == 0 || j == 0 || i == n_ - 1 || j == m_ - 1) {
        res_[(i * m_) + j] = input_[(i * m_) + j];
      } else {
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[((i + ki) * m_) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        res_[(i * m_) + j] = sum;
      }
    }
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < m_; j++) {
      reinterpret_cast<double *>(task_data->outputs[0])[(i * m_) + j] = res_[(i * m_) + j];
    }
  }
  return true;
}
