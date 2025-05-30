#include "omp/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp {

double ParallelIntegration(const std::function<double(const std::vector<double>&)>& func,
                           const std::vector<double>& lower, const std::vector<double>& upper,
                           const std::vector<int>& steps) {
  int total_points = 1;
  std::vector<double> h(steps.size());
  for (size_t i = 0; i < steps.size(); ++i) {
    h[i] = (upper[i] - lower[i]) / steps[i];
    total_points *= (steps[i] + 1);
  }

  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (int idx = 0; idx < total_points; ++idx) {
    std::vector<double> point(steps.size());
    int temp = idx;
    double weight = 1.0;

    for (int dim = static_cast<int>(steps.size()) - 1; dim >= 0; --dim) {
      int i = temp % (steps[dim] + 1);
      temp /= (steps[dim] + 1);

      point[dim] = lower[dim] + i * h[dim];

      if (i == 0 || i == steps[dim]) {
        weight *= 0.5;
      }
    }

    sum += weight * func(point);
  }

  double volume = 1.0;
  for (size_t i = 0; i < steps.size(); ++i) {
    volume *= h[i];
  }

  return sum * volume;
}

TestTaskOpenMP::TestTaskOpenMP(::ppc::core::TaskDataPtr task_data) : ::ppc::core::Task(std::move(task_data)) {}

bool TestTaskOpenMP::PreProcessingImpl() {
  auto* lower_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  dimensions_ = static_cast<int>(task_data->inputs_count[0] / sizeof(double));

  lower_limits_ = std::vector<double>(lower_ptr, lower_ptr + dimensions_);
  upper_limits_ = std::vector<double>(upper_ptr, upper_ptr + dimensions_);
  steps_ = std::vector<int>(steps_ptr, steps_ptr + dimensions_);

  result_ = 0.0;
  return true;
}

bool TestTaskOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

bool TestTaskOpenMP::RunImpl() {
  result_ = ParallelIntegration(function_, lower_limits_, upper_limits_, steps_);
  return true;
}

bool TestTaskOpenMP::PostProcessingImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  if (output_ptr == nullptr) {
    return false;
  }
  *output_ptr = result_;
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp