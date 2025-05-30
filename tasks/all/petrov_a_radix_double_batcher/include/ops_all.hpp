#pragma once

#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi.h"

namespace petrov_a_radix_double_batcher_all {

class TestTaskParallelOmpMpi : public ppc::core::Task {
 public:
  explicit TestTaskParallelOmpMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank_);
  }

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::span<double> in_;
  std::vector<double> res_;
  int global_rank_;
};

}  // namespace petrov_a_radix_double_batcher_all