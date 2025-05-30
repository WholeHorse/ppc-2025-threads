#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/laganina_e_component_labeling/include/ops_tbb.hpp"

TEST(laganina_e_component_labeling_tbb, test_pipeline_run) {
  constexpr int kCount = 2000;

  // Create data
  std::vector<int> in(kCount * kCount, 0);
  std::vector<int> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = 1;
  }

  // Create task_data 6
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.emplace_back(kCount);
  task_data_tbb->inputs_count.emplace_back(kCount);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.emplace_back(kCount);
  task_data_tbb->outputs_count.emplace_back(kCount);

  // Create Task
  auto test_task_tbb = std::make_shared<laganina_e_component_labeling_tbb::TestTaskTBB>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 15;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  laganina_e_component_labeling_tbb::NormalizeLabels(out);
  ASSERT_EQ(in, out);
}

TEST(laganina_e_component_labeling_tbb, test_task_run) {
  constexpr int kCount = 2000;

  // Create data
  std::vector<int> in(kCount * kCount, 0);
  std::vector<int> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = 1;
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.emplace_back(kCount);
  task_data_tbb->inputs_count.emplace_back(kCount);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.emplace_back(kCount);
  task_data_tbb->outputs_count.emplace_back(kCount);

  // Create Task
  auto test_task_tbb = std::make_shared<laganina_e_component_labeling_tbb::TestTaskTBB>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  laganina_e_component_labeling_tbb::NormalizeLabels(out);
  ASSERT_EQ(in, out);
}