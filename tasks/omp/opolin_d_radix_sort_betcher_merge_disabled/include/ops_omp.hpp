#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_omp {
void SortByDigit(std::vector<int>& array, int digit_place);
void BatcherOddEvenMerge(std::vector<int>& array, int start, int mid, int end);
void RadixSort(std::vector<int>& input, int start, int end);

class RadixBatcherSortTaskOpenMP : public ppc::core::Task {
 public:
  explicit RadixBatcherSortTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  int size_;
};
}  // namespace opolin_d_radix_batcher_sort_omp