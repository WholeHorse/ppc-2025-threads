#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/petrov_o_vertical_image_filtration/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(petrov_o_vertical_image_filtration_all, test_pipeline_run) {
  constexpr int kCount = 5000;

  // Create data
  std::vector<int> in(static_cast<size_t>(kCount) * kCount, 1);
  std::vector<int> out(static_cast<size_t>(kCount - 2) * (kCount - 2), 0);
  std::vector<int> expected_out(static_cast<size_t>(kCount - 2) * (kCount - 2), 1);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kCount);
  task_data_all->inputs_count.emplace_back(kCount);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_all = std::make_shared<petrov_o_vertical_image_filtration_all::TaskAll>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  // std::cout << "num_running = " << perf_attr->num_running << std::endl;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (test_task_all->GetRank() == 0) {
    ASSERT_EQ(expected_out, out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_task_run) {
  constexpr int kCount = 5000;

  // Create data
  std::vector<int> in(static_cast<size_t>(kCount) * kCount, 1);
  std::vector<int> out(static_cast<size_t>(kCount - 2) * (kCount - 2), 0);
  std::vector<int> expected_out(static_cast<size_t>(kCount - 2) * (kCount - 2), 1);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kCount);
  task_data_all->inputs_count.emplace_back(kCount);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_all = std::make_shared<petrov_o_vertical_image_filtration_all::TaskAll>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (test_task_all->GetRank() == 0) {
    ASSERT_EQ(expected_out, out);
  }
}
