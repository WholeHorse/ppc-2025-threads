#pragma once

#include <omp.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_congrad_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> a_local_;  // Local part of matrix
  std::vector<double> b_;        // Full RHS vector
  std::vector<double> x_;        // Solution vector
  uint64_t global_size_{};       // Global system size
  int rank_ = 0;                 // Process rank
  int world_size_ = 1;           // Total processes
};

}  // namespace karaseva_e_congrad_mpi