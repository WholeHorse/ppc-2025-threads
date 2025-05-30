#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace morozov_e_lineare_image_filtering_block_gaussian_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, res_;
  int n_, m_;
};

}  // namespace morozov_e_lineare_image_filtering_block_gaussian_omp