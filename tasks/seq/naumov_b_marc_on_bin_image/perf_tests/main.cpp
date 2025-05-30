#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/naumov_b_marc_on_bin_image/include/ops_seq.hpp"

TEST(naumov_b_marc_on_bin_image_seq, test_pipeline_run) {
  constexpr int kCo = 5000;

  std::vector<int> in(kCo * kCo, 0);
  std::vector<int> out(kCo * kCo, 0);

  for (size_t i = 0; i < kCo; i++) {
    in[(i * kCo) + i] = 1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCo);
  task_data_seq->inputs_count.emplace_back(kCo);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCo);
  task_data_seq->outputs_count.emplace_back(kCo);

  auto test_task_sequential = std::make_shared<naumov_b_marc_on_bin_image_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 15;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_GT(*std::ranges::max_element(out), 0);
}

TEST(naumov_b_marc_on_bin_image_seq, test_task_run) {
  constexpr int kCo = 7500;

  std::vector<int> in(kCo * kCo, 0);
  std::vector<int> out(kCo * kCo, 0);

  for (size_t i = 0; i < kCo; i++) {
    in[(i * kCo) + (kCo - i - 1)] = 1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kCo);
  task_data_seq->inputs_count.emplace_back(kCo);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCo);
  task_data_seq->outputs_count.emplace_back(kCo);

  auto test_task_sequential = std::make_shared<naumov_b_marc_on_bin_image_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_GT(*std::ranges::max_element(out), 0);
}
