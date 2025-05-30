#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenerateRandomVector(size_t size, int lower_bound = -1000, int upper_bound = 1000) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = static_cast<int>(lower_bound + (gen() % (upper_bound - lower_bound + 1)));
  }
  return vec;
}
}  // namespace

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_pipeline_run) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(50000);
    out = std::vector(in.size(), 0);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto task = std::make_shared<ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_task_run) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(50000);
    out = std::vector(in.size(), 0);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto task = std::make_shared<ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(std::ranges::is_sorted(out));
}