#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "all/opolin_d_radix_sort_batcher_merge/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_all {
namespace {
void GenDataRadixSort(size_t size, std::vector<int> &vec, std::vector<int> &expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-1000, 1000);
  vec.clear();
  expected.clear();
  vec.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    vec.push_back(dis(gen));
  }
  expected = vec;
  std::ranges::sort(expected);
}
}  // namespace
}  // namespace opolin_d_radix_batcher_sort_all

TEST(opolin_d_radix_batcher_sort_all, test_pipeline_run) {
  boost::mpi::communicator world;
  // Create data
  int size = 1100000;
  std::vector<int> input;
  std::vector<int> expected;
  std::vector<int> out(size, 0);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    opolin_d_radix_batcher_sort_all::GenDataRadixSort(size, input, expected);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(out.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll>(task_data_all);
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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(opolin_d_radix_batcher_sort_all, test_task_run) {
  boost::mpi::communicator world;
  int size = 1100000;
  std::vector<int> input;
  std::vector<int> expected;
  std::vector<int> out(size, 0);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    opolin_d_radix_batcher_sort_all::GenDataRadixSort(size, input, expected);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(out.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  auto test_task_all = std::make_shared<opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll>(task_data_all);
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
  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}