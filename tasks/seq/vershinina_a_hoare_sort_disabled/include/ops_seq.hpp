#pragma once
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_hoare_sort_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int *input_{};
  int n_{};
  std::vector<int> output_;
  int value_;
  void HoareSort(int *s_vec, int first, int last);
  int Partition(int *s_vec, int first, int last);
};

}  // namespace vershinina_a_hoare_sort_seq