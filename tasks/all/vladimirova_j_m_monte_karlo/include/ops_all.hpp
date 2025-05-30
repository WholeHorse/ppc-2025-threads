#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vladimirova_j_m_monte_karlo_all {

struct BoundariesIntegral {
  double min;
  double max;
};

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  bool (*func_)(std::vector<double>, size_t);
  std::vector<BoundariesIntegral> var_integr_;
  size_t var_size_{};
  size_t accuracy_;
  boost::mpi::communicator world_;
};

}  // namespace vladimirova_j_m_monte_karlo_all
