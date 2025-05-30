#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_congrad_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;  // Coefficient matrix
  std::vector<double> b_;  // Right-hand side vector
  std::vector<double> x_;  // Solution vector
  size_t size_{};
};

}  // namespace karaseva_e_congrad_omp