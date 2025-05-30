#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_seq.hpp"

TEST(mezhuev_m_bitwise_integer_sort_seq, test_pipeline_run) {
  constexpr int kCount = 1500;

  std::vector<std::pair<int, int>> ranges = {{-500, 500}, {-1000, 1000}, {-200, 200}};

  for (const auto &range : ranges) {
    int min_value = range.first;
    int max_value = range.second;

    std::vector<int> in(kCount * kCount, 0);
    std::vector<int> out(kCount * kCount, 0);

    for (size_t i = 0; i < kCount * kCount; i++) {
      in[i] = std::rand() % (max_value - min_value + 1) + min_value;
    }

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    auto test_task_sequential = std::make_shared<mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential>(task_data_seq);

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 20;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<int> expected = in;
    std::ranges::sort(expected);

    ASSERT_EQ(expected, out);
  }
}

TEST(mezhuev_m_bitwise_integer_sort_seq, test_task_run) {
  constexpr int kCount = 1500;

  std::vector<std::pair<int, int>> ranges = {{-500, 500}, {-1000, 1000}, {-200, 200}};

  for (const auto &range : ranges) {
    int min_value = range.first;
    int max_value = range.second;

    std::vector<int> in(kCount * kCount, 0);
    std::vector<int> out(kCount * kCount, 0);

    for (size_t i = 0; i < kCount * kCount; i++) {
      in[i] = std::rand() % (max_value - min_value + 1) + min_value;
    }

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    auto test_task_sequential = std::make_shared<mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential>(task_data_seq);

    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 20;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<int> expected = in;
    std::ranges::sort(expected);

    ASSERT_EQ(expected, out);
  }
}