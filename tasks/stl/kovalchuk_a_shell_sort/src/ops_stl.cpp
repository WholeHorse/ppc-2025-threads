#include "stl/kovalchuk_a_shell_sort/include/ops_stl.hpp"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "../modules/core/util/include/util.hpp"
#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort_stl {

ShellSortSTL::ShellSortSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortSTL::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortSTL::ValidationImpl() {
  return !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortSTL::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortSTL::ShellSort() {
  if (input_.empty()) {
    return;
  }

  int n = static_cast<int>(input_.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    int num_threads = std::min(ppc::util::GetPPCNumThreads(), gap);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t, num_threads, gap]() {
        for (int k = t; k < gap; k += num_threads) {
          for (int i = k + gap; i < n; i += gap) {
            int temp = input_[i];
            int j = i;
            while (j >= gap && input_[j - gap] > temp) {
              input_[j] = input_[j - gap];
              j -= gap;
            }
            input_[j] = temp;
          }
        }
      });
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }
}

bool ShellSortSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_, output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort_stl
