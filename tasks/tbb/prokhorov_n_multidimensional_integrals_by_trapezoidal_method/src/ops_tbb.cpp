// ops_tbb.cpp
#include "tbb/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb {
namespace {

double ParallelTrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                                      const std::vector<double>& lower, const std::vector<double>& upper,
                                      const std::vector<int>& steps, size_t current_dim, std::vector<double>& point) {
  if (lower.size() != upper.size() || lower.size() != steps.size()) {
    return 0.0;
  }

  if (current_dim == lower.size()) {
    return func(point);
  }

  double h = (upper[current_dim] - lower[current_dim]) / steps[current_dim];
  double sum = 0.0;

  if (point.size() <= current_dim) {
    point.resize(current_dim + 1);
  }

  sum = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, steps[current_dim] + 1), 0.0,
      [&](const oneapi::tbb::blocked_range<int>& r, double local_sum) {
        for (int i = r.begin(); i != r.end(); ++i) {
          auto local_point = point;
          if (local_point.size() <= current_dim) {
            local_point.resize(current_dim + 1);
          }
          local_point[current_dim] = lower[current_dim] + i * h;
          double weight = (i == 0 || i == steps[current_dim]) ? 0.5 : 1.0;
          local_sum += weight * ParallelTrapezoidalIntegration(func, lower, upper, steps, current_dim + 1, local_point);
        }
        return local_sum;
      },
      std::plus<>());

  return sum * h;
}

}  // namespace

TestTaskTBB::TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool TestTaskTBB::PreProcessingImpl() {
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

bool TestTaskTBB::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

bool TestTaskTBB::RunImpl() {
  std::vector<double> point;
  result_ = ParallelTrapezoidalIntegration(function_, lower_limits_, upper_limits_, steps_, 0, point);
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
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

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb