#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_all {

class RadixALL : public ppc::core::Task {
 public:
  explicit RadixALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  std::vector<int> local_data_;
  boost::mpi::communicator world_;

  // Core radix sort functions
  static void RadixSortLocal(std::vector<int>& arr);
  static void CountingSortByDigit(std::vector<int>& arr, int exp);
  static void RadixSortPositive(std::vector<int>& arr);

  // Helper functions for signed integers
  static void SplitBySign(const std::vector<int>& arr, std::vector<int>& negatives, std::vector<int>& positives);
  static void MergeResults(std::vector<int>& result, const std::vector<int>& negatives,
                           const std::vector<int>& positives);

  // MPI distribution and merging functions
  std::vector<int> DistributeData(const std::vector<int>& data, int rank, int size);
  std::vector<int> GatherAndMerge(const std::vector<int>& local_sorted, int rank, int size);

  // Simple merge function
  static std::vector<int> MergeTwoSorted(const std::vector<int>& left, const std::vector<int>& right);
};

}  // namespace burykin_m_radix_all