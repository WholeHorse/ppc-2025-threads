#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_all {
uint32_t ConvertIntToUint(int num);
int ConvertUintToInt(uint32_t unum);
void RadixSort(std::vector<uint32_t>& uns_vec);
void BatcherOddEvenMerge(std::vector<int>& vec, int low, int high);

class RadixBatcherSortTaskAll : public ppc::core::Task {
 public:
  explicit RadixBatcherSortTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<uint32_t> unsigned_data_;
  int size_;
  boost::mpi::communicator world_;
};
}  // namespace opolin_d_radix_batcher_sort_all