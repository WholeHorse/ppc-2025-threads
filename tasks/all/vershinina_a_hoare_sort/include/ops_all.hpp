#pragma once

#include <boost/mpi/collectives.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi.h"

namespace vershinina_a_hoare_sort_mpi {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_size(MPI_COMM_WORLD, &ws_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  }
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;

  int ws_;
  int rank_;
};
}  // namespace vershinina_a_hoare_sort_mpi