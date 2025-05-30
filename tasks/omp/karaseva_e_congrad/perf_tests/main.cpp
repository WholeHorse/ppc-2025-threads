#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/karaseva_e_congrad/include/ops_omp.hpp"

TEST(karaseva_e_congrad_omp, test_pipeline_run) {
  constexpr int kCount = 10000;
  constexpr double kThreshold = 1e-6;

  // Create matrix a (identity matrix)
  std::vector<double> a(kCount * kCount, 0.0);
  for (size_t i = 0; i < kCount; ++i) {
    a[(i * kCount) + i] = 1.0;
  }

  // Create vector b
  std::vector<double> b(kCount, 5.0);
  std::vector<double> out(kCount, 0.0);

  // Create task data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create task
  auto test_task_omp = std::make_shared<karaseva_e_congrad_omp::TestTaskOpenMP>(task_data_omp);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Verify results
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_NEAR(out[i], b[i], kThreshold);
  }
}

TEST(karaseva_e_congrad_omp, test_task_run) {
  constexpr int kCount = 10000;
  constexpr double kThreshold = 1e-6;

  // Create diagonal matrix a with elements 2.0
  std::vector<double> a(kCount * kCount, 0.0);
  for (size_t i = 0; i < kCount; ++i) {
    a[(i * kCount) + i] = 2.0;
  }

  // Create vector b
  std::vector<double> b(kCount, 4.0);
  std::vector<double> out(kCount, 0.0);

  // Expected solution
  std::vector<double> expected(kCount, 2.0);

  // Create task data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create task
  auto test_task_omp = std::make_shared<karaseva_e_congrad_omp::TestTaskOpenMP>(task_data_omp);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Verify results
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_NEAR(out[i], expected[i], kThreshold);
  }
}