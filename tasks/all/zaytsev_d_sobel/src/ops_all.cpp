#include "all/zaytsev_d_sobel/include/ops_all.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives/reduce.hpp>
#include <cmath>
#include <functional>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

bool zaytsev_d_sobel_all::TestTaskALL::PreProcessingImpl() {
  int rank = world_.rank();

  int in_count = 0;
  if (rank == 0) {
    auto* size_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    width_ = size_ptr[0];
    height_ = size_ptr[1];
    in_count = static_cast<int>(task_data->inputs_count[0]);
  }

  boost::mpi::broadcast(world_, width_, 0);
  boost::mpi::broadcast(world_, height_, 0);
  boost::mpi::broadcast(world_, in_count, 0);

  input_.resize(in_count);
  if (rank == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + in_count, input_.begin());
  }
  boost::mpi::broadcast(world_, input_.data(), in_count, 0);

  int out_count = 0;
  if (rank == 0) {
    out_count = static_cast<int>(task_data->outputs_count[0]);
  }
  boost::mpi::broadcast(world_, out_count, 0);
  output_.assign(out_count, 0);

  return true;
}

bool zaytsev_d_sobel_all::TestTaskALL::ValidationImpl() {
  int rank = world_.rank();
  bool validation = false;

  if (rank == 0) {
    if (task_data->inputs.size() >= 2 && task_data->inputs_count.size() >= 2 && !task_data->outputs_count.empty()) {
      auto* size_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
      width_ = size_ptr[0];
      height_ = size_ptr[1];

      validation = (task_data->inputs_count[0] == task_data->outputs_count[0]) && (width_ >= 3 && height_ >= 3) &&
                   (width_ * height_ == static_cast<int>(task_data->inputs_count[0]));
    }
  }

  boost::mpi::broadcast(world_, validation, 0);
  return validation;
}

bool zaytsev_d_sobel_all::TestTaskALL::RunImpl() {
  static constexpr int kGxkernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  static constexpr int kGykernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int rows = height_ - 2;
  int cols = width_ - 2;
  int total_pixels = rows * cols;
  int rank = world_.rank();
  int size = world_.size();

  int chunk = total_pixels / size;
  int rem = total_pixels % size;
  int start_idx = (rank * chunk) + std::min(rank, rem);
  int end_idx = start_idx + chunk + static_cast<int>(rank < rem);

  tbb::parallel_for(tbb::blocked_range<int>(start_idx, end_idx), [&](const tbb::blocked_range<int>& r) {
    for (int idx = r.begin(); idx < r.end(); ++idx) {
      int i = 1 + (idx / cols);
      int j = 1 + (idx % cols);

      int sumgx = 0;
      int sumgy = 0;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          sumgx += input_[(ni * width_) + nj] * kGxkernel[di + 1][dj + 1];
          sumgy += input_[(ni * width_) + nj] * kGykernel[di + 1][dj + 1];
        }
      }

      int mag = static_cast<int>(std::sqrt((sumgx * sumgx) + (sumgy * sumgy)));
      output_[(i * width_) + j] = std::min(mag, 255);
    }
  });

  if (rank == 0) {
    std::vector<int> combined(output_.size());
    boost::mpi::reduce(world_, output_, combined, std::plus<>(), 0);
    output_.swap(combined);
  } else {
    boost::mpi::reduce(world_, output_, std::plus<>(), 0);
  }

  return true;
}
bool zaytsev_d_sobel_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}
