#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

namespace {
constexpr std::size_t kLargeSize = 1000000;
constexpr std::size_t kMediumSize = 1000000;
}  // namespace

TEST(shuravina_o_hoare_simple_merger_tbb, perf_pipeline_run) {
  std::vector<int> input(kLargeSize);
  std::iota(input.rbegin(), input.rend(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->outputs_count.emplace_back(input.size());

  auto sorter = std::make_shared<shuravina_o_hoare_simple_merger_tbb::HoareSortTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sorter);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(input));
}

TEST(shuravina_o_hoare_simple_merger_tbb, perf_task_run) {
  std::vector<int> input(kMediumSize);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(1, 10000);
  std::ranges::generate(input, [&] { return dist(gen); });

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->outputs_count.emplace_back(input.size());

  auto sorter = std::make_shared<shuravina_o_hoare_simple_merger_tbb::HoareSortTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sorter);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(input));
}