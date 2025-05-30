#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all {

class ShellSortAll : public ppc::core::Task {
 private:
  std::vector<long long int> mas_, tmp_, loc_, loc_tmp_;
  unsigned int n_, n_input_, loc_proc_lenght_;
  bool flag_;
  int effective_num_procs_;
  boost::mpi::communicator world_;

 public:
  explicit ShellSortAll(const std::shared_ptr<ppc::core::TaskData> &task_data) : Task(task_data) {
    if (world_.rank() == 0) {
      n_input_ = task_data->inputs_count[0];
    };
  }

  bool ShellSort(unsigned int, unsigned int);
  static bool OddEvenMergeOMP(long long int *, long long int *, const long long int *, unsigned int);
  bool OddEvenMergeMPI(unsigned int);
  bool FinalMergeOMP(unsigned int);
  bool BatcherSortOMP();
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all
