#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "../include/integrate_seq.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/khasanyanov_k_trapezoid_method/include/integrator.hpp"

using namespace khasanyanov_k_trapezoid_method_seq;

TEST(khasanyanov_k_trapezoid_method_seq, test_pipeline_run) {
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double {
    return (x[0] * x[0]) - (3 * x[1] * x[1] * x[2] * x[2]) + x[2];
  };

  IntegrationBounds bounds = {{2.0, 4.0}, {3.0, 5.0}, {1.0, 2.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodSequential::CreateTaskData(task_data_seq, context, &result);

  auto task = std::make_shared<TrapezoidalMethodSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(khasanyanov_k_trapezoid_method_seq, test_task_run) {
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double {
    return (x[0] * x[0]) - (3 * x[1] * x[1] * x[2] * x[2]) + x[2];
  };

  IntegrationBounds bounds = {{2.0, 4.0}, {3.0, 5.0}, {1.0, 2.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodSequential::CreateTaskData(task_data_seq, context, &result);

  auto task = std::make_shared<TrapezoidalMethodSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}