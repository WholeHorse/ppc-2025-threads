#include "stl/zaytsev_d_sobel/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool zaytsev_d_sobel_stl::TestTaskSTL::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);

  auto *size_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  width_ = size_ptr[0];
  height_ = size_ptr[1];

  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

bool zaytsev_d_sobel_stl::TestTaskSTL::ValidationImpl() {
  auto *size_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = size_ptr[0];
  int height = size_ptr[1];
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && (width >= 3) && (height >= 3) &&
         ((width * height) == int(task_data->inputs_count[0]));
}

bool zaytsev_d_sobel_stl::TestTaskSTL::RunImpl() {
  const int gxkernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int gykernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int rows = height_ - 2;
  int cols = width_ - 2;
  int total = rows * cols;

  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  auto worker = [&](int start, int end) {
    for (int idx = start; idx < end; ++idx) {
      int i = 1 + (idx / cols);
      int j = 1 + (idx % cols);

      int sumgx = 0;
      int sumgy = 0;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          int kr = di + 1;
          int kc = dj + 1;
          int pix = input_[(ni * width_) + nj];
          sumgx += pix * gxkernel[kr][kc];
          sumgy += pix * gykernel[kr][kc];
        }
      }

      int mag = static_cast<int>(std::sqrt((sumgx * sumgx) + (sumgy * sumgy)));
      output_[(i * width_) + j] = std::min(mag, 255);
    }
  };

  int base = total / num_threads;
  int rem = total % num_threads;
  int offset = 0;

  for (int t = 0; t < num_threads; ++t) {
    int chunk = base + (t < rem ? 1 : 0);
    int start = offset;
    int end = offset + chunk;
    threads.emplace_back(worker, start, end);
    offset += chunk;
  }

  for (auto &th : threads) {
    th.join();
  }

  return true;
}

bool zaytsev_d_sobel_stl::TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
