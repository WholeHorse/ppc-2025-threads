#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "all/muradov_m_rect_int/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(muradov_m_rect_int_all, test_pipeline_run) {
  int iterations = 475;
  std::vector<std::pair<double, double>> bounds(3, {-3.0, 3.0});
  double out = 0.0;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&iterations));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_all->inputs_count.emplace_back(1);
  task_data_all->inputs_count.emplace_back(bounds.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data_all->outputs_count.emplace_back(1);

  auto test_task_alluential = std::make_shared<muradov_m_rect_int_all::RectIntTaskMpiStlPar>(
      task_data_all, [](const auto &args) { return (args[0] * args[1]) + (args[1] * args[1]); });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_NEAR(out, 648, 0.3);
}

TEST(muradov_m_rect_int_all, test_task_run) {
  int iterations = 475;
  std::vector<std::pair<double, double>> bounds(3, {-3.0, 3.0});
  double out = 0.0;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&iterations));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_all->inputs_count.emplace_back(1);
  task_data_all->inputs_count.emplace_back(bounds.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data_all->outputs_count.emplace_back(1);

  auto test_task_alluential = std::make_shared<muradov_m_rect_int_all::RectIntTaskMpiStlPar>(
      task_data_all, [](const auto &args) { return (args[0] * args[1]) + (args[1] * args[1]); });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_NEAR(out, 648, 0.3);
}