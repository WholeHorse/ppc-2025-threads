#include "tbb/kovalchuk_a_shell_sort/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace kovalchuk_a_shell_sort_tbb {

ShellSortTBB::ShellSortTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortTBB::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortTBB::ValidationImpl() {
  return !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortTBB::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortTBB::ShellSort() {
  if (input_.empty()) {
    return;
  }

  int n = static_cast<int>(input_.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    tbb::parallel_for(0, gap, [&](int k) {
      for (int i = k + gap; i < n; i += gap) {
        int temp = input_[i];
        int j = i;
        while (j >= gap && input_[j - gap] > temp) {
          input_[j] = input_[j - gap];
          j -= gap;
        }
        input_[j] = temp;
      }
    });
  }
}

bool ShellSortTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_, output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort_tbb
