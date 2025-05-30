#include "omp/kovalchuk_a_shell_sort/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort_omp {

ShellSortOMP::ShellSortOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortOMP::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortOMP::ValidationImpl() {
  return !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortOMP::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortOMP::ShellSort() {
  if (input_.empty()) {
    return;
  }

  std::vector<int>& arr = input_;
  auto n = static_cast<int>(arr.size());

  for (int gap = n / 2; gap > 0; gap /= 2) {
#pragma omp parallel for
    for (int group = 0; group < gap; ++group) {
      for (int i = group + gap; i < n; i += gap) {
        int temp = arr[i];
        int j = i;

        while (j >= gap && arr[j - gap] > temp) {
          arr[j] = arr[j - gap];
          j -= gap;
        }
        arr[j] = temp;
      }
    }
  }
}

bool ShellSortOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_, output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort_omp