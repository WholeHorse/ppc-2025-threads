#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/karaseva_e_congrad/include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(karaseva_e_congrad_mpi, test_pipeline_run) {
  constexpr int kSize = 10000;

  // Create data.
  std::vector<double> a(kSize * kSize, 0.0);
  std::vector<double> b(kSize, 1.0);
  std::vector<double> x(kSize, 0.0);

  for (int i = 0; i < kSize; ++i) {
    a[(i * kSize) + i] = 1.0;
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data->inputs_count.emplace_back(a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data->inputs_count.emplace_back(b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_congrad_mpi::TestTaskMPI>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(b, x);
}

TEST(karaseva_e_congrad_mpi, test_task_run) {
  constexpr int kSize = 10000;

  // Create data
  std::vector<double> a(kSize * kSize, 0.0);
  std::vector<double> b(kSize, 1.0);
  std::vector<double> x(kSize, 0.0);

  for (int i = 0; i < kSize; ++i) {
    a[(i * kSize) + i] = 1.0;
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data->inputs_count.emplace_back(a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data->inputs_count.emplace_back(b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_congrad_mpi::TestTaskMPI>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(b, x);
}