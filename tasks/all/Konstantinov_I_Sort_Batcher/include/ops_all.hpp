#pragma once

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_sort_batcher_all {

class RadixSortBatcherall : public ppc::core::Task {
 public:
  explicit RadixSortBatcherall(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<double> mas_;
  std::vector<double> output_;
};

}  // namespace konstantinov_i_sort_batcher_all