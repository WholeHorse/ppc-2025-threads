#ifndef OPS_ALL_HPP
#define OPS_ALL_HPP

#include <omp.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_all {

class RadixBatcherMergesortParallelAll : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortParallelAll(std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void Sort(std::span<Bigint> arr);

 private:
  vector<Bigint> array_;            // input unsorted numbers array
  vector<Bigint> procchunk_;        // local subarray for each MPI process
  size_t n_ = 0;                    // total array size
  boost::mpi::communicator world_;  // MPI communicator

  static void RadixSort(vector<Bigint>& arr, bool invert);
  static void CountingSort(vector<Bigint>& arr, Bigint digit_place);
  static int GetNumberDigitCapacity(Bigint num);

  void SortParallel(vector<Bigint>& arr, boost::mpi::communicator& comm);
  static void BatcherMergeParallel(vector<Bigint>& arr, int num_threads);
  void MergeAcross(boost::mpi::communicator& group);
};

}  // namespace belov_a_radix_batcher_mergesort_all

#endif  // OPS_ALL_HPP