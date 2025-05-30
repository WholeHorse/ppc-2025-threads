#include "stl/morozov_e_lineare_image_filtering_block_gaussian/include/ops_stl.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
// clang-format off
  constexpr std::array<std::array<double, 3>, 3> kErnel = {{
    {{1.0 / 16, 2.0 / 16, 1.0 / 16}},
    {{2.0 / 16, 4.0 / 16, 2.0 / 16}},
    {{1.0 / 16, 2.0 / 16, 1.0 / 16}}
}};
// clang-format on
inline void ThreadTask(const std::vector<double> &in_vec, int n, int m, std::vector<double> &res_vec, int begin_pos,
                       int end_pos) {
  std::cout << "begin = " << begin_pos << ", end = " << end_pos << '\n';
  for (int i = begin_pos; i < end_pos; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
        res_vec[(i * m) + j] = in_vec[(i * m) + j];
      } else {
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += in_vec[((i + ki) * m) + (j + kj)] * kErnel[ki + 1][kj + 1];
          }
        }
        res_vec[(i * m) + j] = sum;
      }
    }
  }
}
}  // namespace

bool morozov_e_lineare_image_filtering_block_gaussian_stl::TestTaskSTL::PreProcessingImpl() {
  n_ = static_cast<int>(task_data->inputs_count[0]);
  m_ = static_cast<int>(task_data->inputs_count[1]);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (m_ * n_));
  res_ = std::vector<double>(n_ * m_, 0);
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[1] == task_data->outputs_count[1] && task_data->inputs_count[1] > 0;
}

bool morozov_e_lineare_image_filtering_block_gaussian_stl::TestTaskSTL::RunImpl() {
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  int count_el = n_ / num_threads;
  int count_el_remain = n_ % num_threads;
  int cur_count = 0;
  for (int i = 0; i < num_threads; ++i) {
    int end_index = count_el + cur_count + (i < count_el_remain ? 1 : 0);
    threads[i] = std::thread(ThreadTask, std::cref(input_), n_, m_, std::ref(res_), cur_count, end_index);
    if (i < count_el_remain) {
      cur_count++;
    }
    cur_count += count_el;
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_stl::TestTaskSTL::PostProcessingImpl() {
  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < m_; j++) {
      reinterpret_cast<double *>(task_data->outputs[0])[(i * m_) + j] = res_[(i * m_) + j];
    }
  }
  return true;
}
