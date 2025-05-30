#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/lopatin_i_monte_carlo/include/lopatinMonteCarloSTL.hpp"

namespace lopatin_i_monte_carlo_stl {

std::vector<double> GenerateBounds(double min_val, double max_val, int dimensions) {
  std::vector<double> bounds;
  for (int i = 0; i < dimensions; ++i) {
    bounds.push_back(min_val);
    bounds.push_back(max_val);
  }
  return bounds;
}
}  // namespace lopatin_i_monte_carlo_stl

TEST(lopatin_i_monte_carlo_stl, test_pipeline_run) {
  const int dimensions = 5;
  const int iterations = 10000000;

  std::vector<double> bounds = lopatin_i_monte_carlo_stl::GenerateBounds(-3.0, 3.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // exp(x1 + x2 + x3 + x4 + x5)
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::exp(x[0] + x[1] + x[2] + x[3] + x[4]);
  };

  auto test_task = std::make_shared<lopatin_i_monte_carlo_stl::TestTaskSTL>(task_data, function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected = std::pow(std::exp(3.0) - std::exp(-3.0), 5);  // = 3.219e6
  const double tolerance = 0.05 * expected;                             // 5% error
  EXPECT_NEAR(result, expected, tolerance);
}

TEST(lopatin_i_monte_carlo_stl, test_task_run) {
  const int dimensions = 5;
  const int iterations = 10000000;

  std::vector<double> bounds = lopatin_i_monte_carlo_stl::GenerateBounds(-3.0, 3.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // exp(x1 + x2 + x3 + x4 + x5)
  std::function<double(const std::vector<double>&)> function = [](const std::vector<double>& x) {
    return std::exp(x[0] + x[1] + x[2] + x[3] + x[4]);
  };

  auto test_task = std::make_shared<lopatin_i_monte_carlo_stl::TestTaskSTL>(task_data, function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected = std::pow(std::exp(3.0) - std::exp(-3.0), 5);  // = 3.219e6
  const double tolerance = 0.05 * expected;                             // 5% error
  EXPECT_NEAR(result, expected, tolerance);
}