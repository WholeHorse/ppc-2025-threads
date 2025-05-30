#include "all/kapustin_i_jarv_alg/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

std::pair<int, int> kapustin_i_jarv_alg_all::TestTaskAll::FindLocalBestOMP(size_t start, size_t end,
                                                                           size_t current_index,
                                                                           const std::pair<int, int>& init_best) {
  std::pair<int, int> local_best = init_best;

#pragma omp parallel
  {
    std::pair<int, int> thread_best = local_best;

#pragma omp for nowait
    for (int i = static_cast<int>(start); i < static_cast<int>(end); ++i) {
      if (static_cast<size_t>(i) == current_index) {
        continue;
      }

      int orient = Orientation(input_[current_index], thread_best, input_[i]);
      if (orient == 0) {
        int dist_best = CalculateDistance(input_[current_index], thread_best);
        int dist_i = CalculateDistance(input_[current_index], input_[i]);
        if (dist_i > dist_best) {
          thread_best = input_[i];
        }
      } else if (orient > 0) {
        thread_best = input_[i];
      }
    }

#pragma omp critical
    {
      int orient = Orientation(input_[current_index], local_best, thread_best);
      if (orient > 0 || (orient == 0 && CalculateDistance(input_[current_index], thread_best) >
                                            CalculateDistance(input_[current_index], local_best))) {
        local_best = thread_best;
      }
    }
  }

  return local_best;
}

int kapustin_i_jarv_alg_all::TestTaskAll::CalculateDistance(const std::pair<int, int>& p1,
                                                            const std::pair<int, int>& p2) {
  return static_cast<int>(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

int kapustin_i_jarv_alg_all::TestTaskAll::Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                                      const std::pair<int, int>& r) {
  int val = ((q.second - p.second) * (r.first - q.first)) - ((q.first - p.first) * (r.second - q.second));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : -1;
}

bool kapustin_i_jarv_alg_all::TestTaskAll::PreProcessingImpl() {
  std::vector<std::pair<int, int>> points;

  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    auto* data = reinterpret_cast<std::pair<int, int>*>(task_data->inputs[i]);
    size_t count = task_data->inputs_count[i];
    points.assign(data, data + count);
  }

  input_ = points;

  leftmost_index_ = 0;
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i].first < input_[leftmost_index_].first) {
      leftmost_index_ = i;
    }
  }

  current_point_ = input_[leftmost_index_];
  return true;
}

bool kapustin_i_jarv_alg_all::TestTaskAll::ValidationImpl() { return !task_data->inputs.empty(); }

bool kapustin_i_jarv_alg_all::TestTaskAll::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::pair<int, int> start_point = current_point_;
  size_t current_index = leftmost_index_;
  output_.clear();
  output_.push_back(start_point);

  do {
    size_t next_index = (current_index + 1) % input_.size();
    std::pair<int, int> local_best = input_[next_index];

    size_t chunk_size = input_.size() / size;
    size_t remainder = input_.size() % size;
    size_t start = (rank * chunk_size) + std::min(static_cast<size_t>(rank), remainder);
    size_t end = start + chunk_size + (rank < static_cast<int>(remainder) ? 1 : 0);

    local_best = FindLocalBestOMP(start, end, current_index, local_best);

    int local_data[3] = {local_best.first, local_best.second, CalculateDistance(current_point_, local_best)};
    int* all_data = new int[3 * size];

    MPI_Allgather(local_data, 3, MPI_INT, all_data, 3, MPI_INT, MPI_COMM_WORLD);

    std::pair<int, int> global_best = local_best;
    int best_dist = local_data[2];

    for (int i = 0; i < size; ++i) {
      int x = all_data[i * 3];
      int y = all_data[(i * 3) + 1];
      int dist = all_data[(i * 3) + 2];
      std::pair<int, int> candidate = {x, y};

      int orient = Orientation(current_point_, global_best, candidate);
      if (orient > 0 || (orient == 0 && dist > best_dist)) {
        global_best = candidate;
        best_dist = dist;
      }
    }

    delete[] all_data;

    if (!output_.empty() && global_best == output_.front()) {
      break;
    }

    current_point_ = global_best;
    output_.push_back(current_point_);

    for (size_t i = 0; i < input_.size(); ++i) {
      if (input_[i] == current_point_) {
        current_index = i;
        break;
      }
    }

  } while (current_point_ != start_point);

  return true;
}

bool kapustin_i_jarv_alg_all::TestTaskAll::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0]);
  std::ranges::copy(output_, result_ptr);
  return true;
}
