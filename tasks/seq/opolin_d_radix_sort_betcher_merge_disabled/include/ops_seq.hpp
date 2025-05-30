#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_betcher_sort_seq {
void SortByDigit(std::vector<int>& array, int digit_place);

class RadixBetcherSortTaskSequential : public ppc::core::Task {
 public:
  explicit RadixBetcherSortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  int size_;
};

}  // namespace opolin_d_radix_betcher_sort_seq