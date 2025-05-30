#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaytsev_d_sobel_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int width_, height_;
};

}  // namespace zaytsev_d_sobel_stl
