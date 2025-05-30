#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "all/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
void GetRandomVector(std::vector<long long> &v, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dis(gen);
  }
}
}  // namespace

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_pipeline_run) {
  constexpr int kSizeOfVector = 50000;
  boost::mpi::communicator world;
  // Create data
  std::vector<long long int> in(kSizeOfVector);
  std::vector<long long int> answer(kSizeOfVector);
  if (world.rank() == 0) {
    GetRandomVector(in, -1000, 1000);
    answer = in;
    std::ranges::sort(answer);
  }
  std::vector<long long int> out(in);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto test_task_sequential =
      std::make_shared<volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    ASSERT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_task_run) {
  constexpr int kSizeOfVector = 50000;
  boost::mpi::communicator world;

  // Create data
  std::vector<long long int> in(kSizeOfVector);
  std::vector<long long int> answer(kSizeOfVector);
  if (world.rank() == 0) {
    GetRandomVector(in, -1000, 1000);
    answer = in;
    std::ranges::sort(answer);
  }
  std::vector<long long int> out(in);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto test_task_sequential =
      std::make_shared<volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    ASSERT_EQ(answer, out);
  }
}
