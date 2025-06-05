#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/naumov_b_marc_on_bin_image/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(naumov_b_marc_on_bin_image_all, test_pipeline_run) {
  constexpr int kCo = 5000;

  std::vector<int> in(kCo * kCo, 0);
  std::vector<int> out(kCo * kCo, 0);

  for (size_t i = 0; i < kCo; i++) {
    in[(i * kCo) + i] = 1;
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kCo);
  task_data_all->inputs_count.emplace_back(kCo);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(kCo);
  task_data_all->outputs_count.emplace_back(kCo);

  auto test_task_stl = std::make_shared<naumov_b_marc_on_bin_image_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 15;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < kCo * kCo; ++i) {
    if (in[i] == 0) {
      ASSERT_EQ(out[i], 0);
    } else {
      ASSERT_GT(out[i], 0);
    }
  }
}

TEST(naumov_b_marc_on_bin_image_all, test_task_run) {
  constexpr int kCo = 7500;

  std::vector<int> in(kCo * kCo, 0);
  std::vector<int> out(kCo * kCo, 0);

  for (size_t i = 0; i < kCo; i++) {
    in[(i * kCo) + (kCo - i - 1)] = 1;
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kCo);
  task_data_all->inputs_count.emplace_back(kCo);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(kCo);
  task_data_all->outputs_count.emplace_back(kCo);

  auto test_task_stl = std::make_shared<naumov_b_marc_on_bin_image_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < kCo * kCo; ++i) {
    if (in[i] == 0) {
      ASSERT_EQ(out[i], 0);
    } else {
      ASSERT_GT(out[i], 0);
    }
  }
}