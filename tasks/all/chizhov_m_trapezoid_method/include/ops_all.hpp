#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chizhov_m_trapezoid_method_all {
using Function = std::function<double(const std::vector<double>&)>;

double TrapezoidMethod(Function& f, size_t div, size_t dim, std::vector<double>& lower_limits,
                       std::vector<double>& upper_limits, const boost::mpi::communicator& world);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void SetFunc(Function f);

 private:
  Function f_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  size_t div_;
  size_t dim_;
  double res_;

  boost::mpi::communicator world_;
};
}  // namespace chizhov_m_trapezoid_method_all