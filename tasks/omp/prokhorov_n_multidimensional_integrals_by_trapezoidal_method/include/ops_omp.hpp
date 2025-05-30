#pragma once

#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp {

double ParallelIntegration(const std::function<double(const std::vector<double>&)>& func,
                           const std::vector<double>& lower, const std::vector<double>& upper,
                           const std::vector<int>& steps);

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SetFunction(const std::function<double(const std::vector<double>&)>& func) { function_ = func; }

 private:
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  std::vector<int> steps_;
  double result_{};
  int dimensions_{};
  std::function<double(const std::vector<double>&)> function_;
};

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp