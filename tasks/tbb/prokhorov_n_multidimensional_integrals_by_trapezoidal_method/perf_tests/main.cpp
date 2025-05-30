#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_tbb.hpp"

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_pipeline_run) {
  std::vector<double> lower = {0.0, 0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {300, 300, 300};
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  auto test_task_tbb =
      std::make_shared<prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB>(task_data_tbb);

  test_task_tbb->SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_NEAR(result, 0.125, 1e-4);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_task_run) {
  std::vector<double> lower = {0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {2000, 2000};
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  auto test_task_tbb =
      std::make_shared<prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB>(task_data_tbb);

  test_task_tbb->SetFunction([](const std::vector<double>& point) { return point[0] + point[1]; });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_NEAR(result, 1.0, 1e-4);
}