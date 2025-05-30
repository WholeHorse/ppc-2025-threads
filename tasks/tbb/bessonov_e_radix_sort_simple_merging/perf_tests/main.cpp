#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/bessonov_e_radix_sort_simple_merging/include/ops_tbb.hpp"

TEST(bessonov_e_radix_sort_simple_merging_tbb, test_pipeline_run) {
  const int n = 5000000;
  std::vector<double> input_vector(n);
  for (int i = 0; i < n; i++) {
    input_vector[i] = static_cast<double>(n - i);
  }
  std::vector<double> output_vector(n, 0.0);

  std::vector<double> result_vector = input_vector;
  std::ranges::sort(result_vector);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  auto test_task = std::make_shared<bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, test_task_run) {
  const int n = 5000000;
  std::vector<double> input_vector(n);
  for (int i = 0; i < n; i++) {
    input_vector[i] = static_cast<double>(n - i);
  }
  std::vector<double> output_vector(n, 0.0);

  std::vector<double> result_vector = input_vector;
  std::ranges::sort(result_vector);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  auto test_task = std::make_shared<bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output_vector, result_vector);
}
