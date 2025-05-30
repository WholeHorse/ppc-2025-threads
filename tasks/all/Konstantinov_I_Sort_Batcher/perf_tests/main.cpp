#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(Konstantinov_I_Sort_Batcher_all, test_pipeline_run) {
  constexpr int kCount = 300000;

  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    for (size_t i = 0; i < kCount; ++i) {
      in[i] = static_cast<double>(kCount - i);
      exp_out[i] = static_cast<double>(i + 1);
    }
  }
  std::vector<double> out(kCount);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  auto test_task = std::make_shared<konstantinov_i_sort_batcher_all::RadixSortBatcherall>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_task_run) {
  constexpr int kCount = 300000;

  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    for (size_t i = 0; i < kCount; ++i) {
      in[i] = static_cast<double>(kCount - i);
      exp_out[i] = static_cast<double>(i + 1);
    }
  }
  std::vector<double> out(kCount);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  auto test_task = std::make_shared<konstantinov_i_sort_batcher_all::RadixSortBatcherall>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(exp_out, out);
  }
}