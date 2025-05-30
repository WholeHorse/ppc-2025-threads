#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SetFunction(std::function<double(const std::vector<double>&)> func) { function_ = std::move(func); }
  [[nodiscard]] int GetRank() const { return world_.rank(); }

 private:
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  std::vector<int> steps_;
  double result_{};
  int dimensions_{};
  std::function<double(const std::vector<double>&)> function_;
  boost::mpi::communicator world_;
};

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all