#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_bitwise_integer_sort_stl {

class SortSTL : public ppc::core::Task {
 public:
  explicit SortSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<int>& GetInput() const { return input_; }
  [[nodiscard]] int GetMaxValue() const { return max_value_; }
  [[nodiscard]] const std::vector<int>& GetOutput() const { return output_; }

 private:
  std::vector<int> input_, output_;
  int max_value_{};
};

}  // namespace mezhuev_m_bitwise_integer_sort_stl