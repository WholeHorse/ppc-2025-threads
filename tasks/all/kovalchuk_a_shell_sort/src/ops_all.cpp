#include "../include/ops_all.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"
#include "core/task/include/task.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace kovalchuk_a_shell_sort_all {

ShellSortAll::ShellSortAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  }
  return true;
}

bool ShellSortAll::ValidationImpl() {
  return world_.rank() != 0 ||
         (!task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 && !task_data->inputs.empty());
}

bool ShellSortAll::RunImpl() {
  const int rank = world_.rank();
  int total_size = rank == 0 ? static_cast<int>(input_.size()) : 0;

  boost::mpi::broadcast(world_, total_size, 0);

  if (rank >= total_size) {
    world_.split(1);
    return true;
  }

  group_ = world_.split(0);

  int num_procs = group_.size();
  counts_ = std::vector<int>(num_procs, total_size / num_procs);
  std::vector<int> displs(num_procs, 0);
  for (int i = 0; i < total_size % num_procs; ++i) {
    counts_[i]++;
  }
  for (int i = 1; i < num_procs; ++i) {
    displs[i] = displs[i - 1] + counts_[i - 1];
  }

  const int n = counts_[rank];

  std::vector<int> buffer(n);
  boost::mpi::scatterv(group_, input_.data(), counts_, displs, buffer.data(), n, 0);

  for (int gap = n / 2; gap > 0; gap /= 2) {
    tbb::parallel_for(0, gap, [&](int k) {
      for (int i = k + gap; i < n; i += gap) {
        int temp = buffer[i];
        int j = i;
        while (j >= gap && buffer[j - gap] > temp) {
          buffer[j] = buffer[j - gap];
          j -= gap;
        }
        buffer[j] = temp;
      }
    });
  }
  input_ = buffer;

  std::vector<int> gathered;
  if (group_.rank() == 0) {
    gathered.resize(task_data->inputs_count[0]);
  }

  boost::mpi::gatherv(group_, input_, gathered.data(), counts_, 0);

  if (group_.rank() == 0) {
    SimpleMerge(num_procs, gathered, displs, total_size);
  }

  return true;
}

void ShellSortAll::SimpleMerge(int num_procs, const std::vector<int>& gathered, const std::vector<int>& displs,
                               int total_size) {
  using Element = std::tuple<int, int, int>;
  auto comp = [](const Element& a, const Element& b) { return std::get<0>(a) > std::get<0>(b); };
  std::priority_queue<Element, std::vector<Element>, decltype(comp)> min_heap(comp);

  for (int i = 0; i < num_procs; ++i) {
    if (counts_[i] > 0) {
      min_heap.emplace(gathered[displs[i]], i, 0);
    }
  }

  result_.clear();
  result_.reserve(total_size);

  while (!min_heap.empty()) {
    auto [val, proc_idx, idx_in_block] = min_heap.top();
    min_heap.pop();
    result_.push_back(val);

    if (idx_in_block + 1 < counts_[proc_idx]) {
      int next_idx = idx_in_block + 1;
      int next_val = gathered[displs[proc_idx] + next_idx];
      min_heap.emplace(next_val, proc_idx, next_idx);
    }
  }
}

bool ShellSortAll::PostProcessingImpl() {
  if (group_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(result_.begin(), result_.end(), output_ptr);
  }
  return true;
}

}  // namespace kovalchuk_a_shell_sort_all
