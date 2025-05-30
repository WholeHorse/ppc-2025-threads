#pragma once
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort_stl {

class ShellSortSTL : public ppc::core::Task {
 public:
  explicit ShellSortSTL(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  void ShellSort();
};

}  // namespace kovalchuk_a_shell_sort_stl