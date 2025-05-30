#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(shuravina_o_hoare_simple_merger_stl, test_pipeline_run) {
  const int count = 1000000;
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);

  for (int i = 0; i < count; i++) {
    in[i] = count - i;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point.time_since_epoch()).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (int i = 0; i < count - 1; i++) {
    ASSERT_LE(out[i], out[i + 1]);
  }
}

TEST(shuravina_o_hoare_simple_merger_stl, test_task_run) {
  const int count = 1000000;
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);

  for (int i = 0; i < count; i++) {
    in[i] = count - i;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point.time_since_epoch()).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (int i = 0; i < count - 1; i++) {
    ASSERT_LE(out[i], out[i + 1]);
  }
}