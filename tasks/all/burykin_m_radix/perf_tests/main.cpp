#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/burykin_m_radix/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<int> GenerateRandomVector(size_t size, int min_val = -10000, int max_val = 10000) {
  std::vector<int> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min_val, max_val);
  for (auto &elem : vec) {
    elem = dis(gen);
  }
  return vec;
}

}  // namespace

TEST(burykin_m_radix_all, test_pipeline_run) {
  boost::mpi::communicator world;

  constexpr size_t kNumElements = 10000000;

  std::vector<int> input;
  std::vector<int> expected;
  std::vector<int> output;

  if (world.rank() == 0) {
    input = GenerateRandomVector(kNumElements);
    expected = input;
    std::ranges::sort(expected);
    output.resize(kNumElements, 0);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data->inputs_count.emplace_back(static_cast<std::uint32_t>(input.size()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(output.size()));
  }

  auto task = std::make_shared<burykin_m_radix_all::RadixALL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  world.barrier();
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_EQ(output, expected);
  }
}

TEST(burykin_m_radix_all, test_task_run) {
  boost::mpi::communicator world;

  constexpr size_t kNumElements = 100000000;

  std::vector<int> input;
  std::vector<int> expected;
  std::vector<int> output;

  if (world.rank() == 0) {
    input = GenerateRandomVector(kNumElements);
    expected = input;
    std::ranges::sort(expected);
    output.resize(kNumElements, 0);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data->inputs_count.emplace_back(static_cast<std::uint32_t>(input.size()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(output.size()));
  }

  auto task = std::make_shared<burykin_m_radix_all::RadixALL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  world.barrier();
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_EQ(output, expected);
  }
}
