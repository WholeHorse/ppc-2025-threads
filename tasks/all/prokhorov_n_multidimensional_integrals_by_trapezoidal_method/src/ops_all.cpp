#include "all/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all {

bool TestTaskALL::PreProcessingImpl() {
  if (!task_data || task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr ||
      task_data->inputs[2] == nullptr) {
    return false;
  }

  auto* lower_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  dimensions_ = static_cast<int>(task_data->inputs_count[0] / sizeof(double));
  if (dimensions_ == 0) {
    return false;
  }

  if (task_data->inputs_count[0] < dimensions_ * sizeof(double) ||
      task_data->inputs_count[1] < dimensions_ * sizeof(double) ||
      task_data->inputs_count[2] < dimensions_ * sizeof(int)) {
    return false;
  }

  lower_limits_ = std::vector<double>(lower_ptr, lower_ptr + dimensions_);
  upper_limits_ = std::vector<double>(upper_ptr, upper_ptr + dimensions_);
  steps_ = std::vector<int>(steps_ptr, steps_ptr + dimensions_);

  result_ = 0.0;
  return true;
}

bool TestTaskALL::ValidationImpl() {
  if (task_data->inputs_count[0] <= 0 || task_data->inputs_count[1] <= 0 || task_data->inputs_count[2] <= 0) {
    return false;
  }

  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

namespace {

double SequentialTrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                                        const std::vector<double>& lower, const std::vector<double>& upper,
                                        const std::vector<int>& steps, const std::vector<double>& point) {
  size_t current_dim = point.size();
  if (current_dim == lower.size()) {
    return func(point);
  }

  double h = (upper[current_dim] - lower[current_dim]) / steps[current_dim];
  double local_sum = 0.0;

  for (int i = 0; i <= steps[current_dim]; ++i) {
    double x = lower[current_dim] + (i * h);
    auto new_point = point;
    new_point.push_back(x);
    double weight = (i == 0 || i == steps[current_dim]) ? 0.5 : 1.0;
    local_sum += weight * SequentialTrapezoidalIntegration(func, lower, upper, steps, new_point);
  }

  return local_sum;
}

double ParallelTrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                                      const std::vector<double>& lower, const std::vector<double>& upper,
                                      const std::vector<int>& steps, const boost::mpi::communicator& world) {
  int rank = world.rank();
  int size = world.size();

  int total_points = steps[0] + 1;

  int points_per_process = total_points / size;
  int remainder = total_points % size;

  int start = (rank * points_per_process) + std::min(rank, remainder);
  int end = start + points_per_process + (rank < remainder ? 1 : 0);

  double h = (upper[0] - lower[0]) / steps[0];
  double local_sum = 0.0;

#pragma omp parallel for reduction(+ : local_sum)
  for (int i = start; i < end; ++i) {
    double x = lower[0] + (i * h);
    std::vector<double> point = {x};
    double weight = (i == 0 || i == steps[0]) ? 0.5 : 1.0;
    local_sum += weight * SequentialTrapezoidalIntegration(func, lower, upper, steps, point);
  }

  double global_sum = 0.0;
  boost::mpi::all_reduce(world, local_sum, global_sum, std::plus<>());

  return global_sum;
}

}  // namespace

bool TestTaskALL::RunImpl() {
  if (!function_) {
    return false;
  }

  double raw_sum = ParallelTrapezoidalIntegration(function_, lower_limits_, upper_limits_, steps_, world_);

  double volume = 1.0;
  for (size_t i = 0; i < lower_limits_.size(); ++i) {
    volume *= (upper_limits_[i] - lower_limits_[i]) / steps_[i];
  }

  result_ = raw_sum * volume;

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  }
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all