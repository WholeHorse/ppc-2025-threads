#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermilova_d_shell_sort_batcher_even_odd_merger_all {

class AllTask : public ppc::core::Task {
 public:
  explicit AllTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_;
};

}  // namespace ermilova_d_shell_sort_batcher_even_odd_merger_all