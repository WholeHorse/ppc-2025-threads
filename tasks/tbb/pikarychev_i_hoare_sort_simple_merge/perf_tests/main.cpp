#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "../include/ops_tbb.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
template <typename T>
std::vector<T> GenerateReverseSortedVector(size_t size) {
  std::vector<T> vec(size);
  std::iota(vec.rbegin(), vec.rend(), 0);
  return vec;
}
}  // namespace

TEST(pikarychev_i_hoare_sort_simple_merge_tbb, test_pipeline_run) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-5000, 5000);

  auto in = GenerateReverseSortedVector<int>(30303);

  std::vector<int> out(in.size());
  bool reverse = false;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&reverse));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = pikarychev_i_hoare_sort_simple_merge::HoareThreadBB<int>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count()) *
           1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(std::make_shared<decltype(task)>(std::move(task)));
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(std::ranges::is_sorted(out, std::greater<>{}));
}

TEST(pikarychev_i_hoare_sort_simple_merge_tbb, test_task_run) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-5000, 5000);

  auto in = GenerateReverseSortedVector<int>(30303);

  std::vector<int> out(in.size());
  bool reverse = false;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&reverse));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = pikarychev_i_hoare_sort_simple_merge::HoareThreadBB<int>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count()) *
           1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(std::make_shared<decltype(task)>(std::move(task)));
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(std::ranges::is_sorted(out, std::greater<>{}));
}