#include "all/morozov_e_lineare_image_filtering_block_gaussian/include/ops_all.hpp"

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>
namespace {
// Алгоритм вычисления диапазона вычисления для каждого процесса
std::pair<int, int> GetStartEndIndices(int count_proc, int curr_runk_proc, int array_size) {
  int start = 0;
  int end = 0;
  int count = array_size / count_proc;
  int rem = array_size % count_proc;
  if (count_proc < array_size) {
    if (count_proc % array_size == 0) {
      start = curr_runk_proc * count;
      end = start + count;
    } else {
      if (curr_runk_proc < rem) {
        start = (curr_runk_proc) * (count + 1);
        end = start + count + 1;
      } else {
        start = rem * (count + 1) + (curr_runk_proc - rem) * (count);
        end = start + count;
      }
    }
  } else {
    if (curr_runk_proc < array_size) {
      start = curr_runk_proc;
      end = start + 1;
    }
  }
  return {start, end};
}

}  // namespace
bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::PreProcessingImpl() {
  n_ = static_cast<int>(task_data->inputs_count[0]);
  m_ = static_cast<int>(task_data->inputs_count[1]);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (m_ * n_));
  res_ = std::vector<double>(n_ * m_, 0);
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0 &&
           task_data->inputs_count[1] == task_data->outputs_count[1] && task_data->inputs_count[1] > 0;
  }
  return true;
}
inline double morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::ApplyGaussianFilter(int i, int j) {
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  double sum = 0.0;
  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      sum += input_[((i + ki) * m_) + (j + kj)] * kernel[ki + 1][kj + 1];
    }
  }
  return sum;
}
bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::RunImpl() {
  auto start_end_pair = GetStartEndIndices(world_.size(), world_.rank(), n_);
  int start = start_end_pair.first;
  int end = start_end_pair.second;
#pragma omp parallel for
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < m_; ++j) {
      if (i == 0 || j == 0 || i == n_ - 1 || j == m_ - 1) {
        res_[(i * m_) + j] = input_[(i * m_) + j];
      } else {
        res_[(i * m_) + j] = ApplyGaussianFilter(i, j);
      }
    }
  }
  if (world_.rank() == 0) {
    for (int p = 1; p < world_.size(); ++p) {
      int start_p = 0;
      int end_p = 0;
      world_.recv(p, 0, &start_p, 1);
      world_.recv(p, 0, &end_p, 1);
      std::vector<double> temp((end_p - start_p) * m_);
      world_.recv(p, 0, temp.data(), (end_p - start_p) * m_);
      for (int i = start_p; i < end_p; ++i) {
        for (int j = 0; j < m_; ++j) {
          res_[(i * m_) + j] = temp[((i - start_p) * m_) + j];
        }
      }
    }
  } else {
    world_.send(0, 0, &start, 1);
    world_.send(0, 0, &end, 1);
    std::vector<double> temp((end - start) * m_);
    for (int i = start; i < end; ++i) {
      for (int j = 0; j < m_; ++j) {
        temp[((i - start) * m_) + j] = res_[(i * m_) + j];
      }
    }
    world_.send(0, 0, temp.data(), (end - start) * m_);
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < m_; j++) {
        reinterpret_cast<double *>(task_data->outputs[0])[(i * m_) + j] = res_[(i * m_) + j];
      }
    }
  }
  return true;
}
