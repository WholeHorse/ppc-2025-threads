#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi.h"

namespace {
std::vector<double> CreateVector(size_t size) {
  std::vector<double> vec(size);
  std::iota(vec.rbegin(), vec.rend(), 0);
  return vec;
}
}  // namespace

TEST(petrov_a_radix_double_batcher_all, test_pipeline_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto in = CreateVector(22222000);
  std::vector<double> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out));
  }
}

TEST(petrov_a_radix_double_batcher_all, test_task_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto in = CreateVector(22222000);
  std::vector<double> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out));
  }
}