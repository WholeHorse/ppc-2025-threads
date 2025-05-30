#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "all/kapustin_i_jarv_alg/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(kapustin_i_jarv_alg_all, test_pipeline_run) {
  constexpr int kCount = 10000000;

  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {1000, 0}, {1000, 1000}, {0, 1000}};

  std::vector<std::pair<int, int>> input_points(kCount, {500, 500});
  input_points[0] = {0, 0};
  input_points[1] = {1000, 0};
  input_points[2] = {1000, 1000};
  input_points[3] = {0, 1000};

  std::vector<std::pair<int, int>> output_result(kCount);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  auto test_task_all = std::make_shared<kapustin_i_jarv_alg_all::TestTaskAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  output_result.resize(expected_result.size());

  ASSERT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(output_result[i], expected_result[i]);
  }
}

TEST(kapustin_i_jarv_alg_all, test_task_run) {
  constexpr int kCount = 10000000;

  std::vector<std::pair<int, int>> input_points(kCount, {500, 500});
  input_points[0] = {0, 0};
  input_points[1] = {1000, 0};
  input_points[2] = {1000, 1000};
  input_points[3] = {0, 1000};

  std::vector<std::pair<int, int>> output_result(4);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  auto test_task_all = std::make_shared<kapustin_i_jarv_alg_all::TestTaskAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {1000, 0}, {1000, 1000}, {0, 1000}};

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(output_result[i], expected_result[i]);
  }
}
