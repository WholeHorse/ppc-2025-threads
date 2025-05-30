#pragma once

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace polikanov_v_rectangles {

using FunctionExecutor = std::function<double(const std::vector<double> &)>;
using FunctionArguments = std::vector<double>;
using IntegrationBound = std::pair<double, double>;

class TaskAll : public ppc::core::Task {
 public:
  explicit TaskAll(ppc::core::TaskDataPtr task_data, FunctionExecutor function)
      : Task(std::move(task_data)), function_(std::move(function)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  FunctionExecutor function_;
  std::size_t discretization_;
  std::vector<IntegrationBound> bounds_;
  double result_;
  boost::mpi::communicator world_;
};

}  // namespace polikanov_v_rectangles