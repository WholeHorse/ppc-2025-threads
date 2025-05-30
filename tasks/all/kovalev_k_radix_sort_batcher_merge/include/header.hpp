#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_radix_sort_batcher_merge_all {

class TestTaskAll : public ppc::core::Task {
 private:
  std::vector<long long int> mas_, tmp_, loc_, loc_tmp_;
  unsigned int n_, n_input_, loc_proc_lenght_;
  int effective_num_procs_;
  boost::mpi::communicator world_;

 public:
  explicit TestTaskAll(const std::shared_ptr<ppc::core::TaskData> &task_data) : Task(task_data) {
    if (world_.rank() == 0) {
      n_input_ = task_data->inputs_count[0];
    };
  }
  static bool RadixUnsigned(unsigned long long *, unsigned long long *, unsigned int);
  bool RadixSigned(unsigned int, unsigned int);
  static bool Countbyte(unsigned long long *, int *, unsigned int, unsigned int);
  static bool OddEvenMergeOMP(long long int *, long long int *, const long long int *, unsigned int);
  bool OddEvenMergeMPI(unsigned int);
  bool FinalMergeOMP(unsigned int);
  bool BatcherSortOMP();
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace kovalev_k_radix_sort_batcher_merge_all