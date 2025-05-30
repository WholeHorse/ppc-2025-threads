#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "all/shlyakov_m_shell_sort/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenerateRandomArray(std::size_t size) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<int> distribution_range(-1000, 1000);
  int min_val = distribution_range(generator);
  int max_val = distribution_range(generator);

  if (min_val > max_val) {
    std::swap(min_val, max_val);
  }

  std::uniform_int_distribution<int> distribution(min_val, max_val);

  std::vector<int> arr(size);
  for (std::size_t i = 0; i < size; ++i) {
    arr[i] = distribution(generator);
  }
  return arr;
}

bool IsSorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (std::size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr std::size_t kCount = 100000;

  std::vector<int> in = GenerateRandomArray(kCount);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<shlyakov_m_shell_sort_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 99;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);

  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    EXPECT_TRUE(IsSorted(out));
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, test_task_run) {
  boost::mpi::communicator world;
  constexpr std::size_t kCount = 100000;

  std::vector<int> in = GenerateRandomArray(kCount);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<shlyakov_m_shell_sort_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 99;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);

  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    EXPECT_TRUE(IsSorted(out));
  }
}