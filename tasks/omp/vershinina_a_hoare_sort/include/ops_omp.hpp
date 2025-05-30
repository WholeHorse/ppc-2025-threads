#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_hoare_sort_omp {
class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> res_;
};

}  // namespace vershinina_a_hoare_sort_omp