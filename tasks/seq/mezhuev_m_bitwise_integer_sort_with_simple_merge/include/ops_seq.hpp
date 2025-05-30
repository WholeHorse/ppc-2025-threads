#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_bitwise_integer_sort_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
};

}  // namespace mezhuev_m_bitwise_integer_sort_seq
