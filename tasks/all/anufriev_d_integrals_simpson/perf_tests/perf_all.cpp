#include <gtest/gtest.h>
#define OMPI_SKIP_MPICXX
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/anufriev_d_integrals_simpson/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(anufriev_d_integrals_simpson_all, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {2, 0.0, 1.0, 2000, 0.0, 1.0, 2000, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_all->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.push_back(static_cast<uint32_t>(in.size() * sizeof(double)));
    task_data_all->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.push_back(static_cast<uint32_t>(out.size() * sizeof(double)));
  }

  auto task_all = std::make_shared<anufriev_d_integrals_simpson_all::IntegralsSimpsonAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const auto t_global_start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t_global_start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_all);

  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    double result = out[0];
    EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {2, 0.0, 1.0, 2000, 0.0, 1.0, 2000, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data_all->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.push_back(static_cast<uint32_t>(in.size() * sizeof(double)));
    task_data_all->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.push_back(static_cast<uint32_t>(out.size() * sizeof(double)));
  }

  auto task_all = std::make_shared<anufriev_d_integrals_simpson_all::IntegralsSimpsonAll>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const auto t_global_start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t_global_start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_all);

  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    double result = out[0];
    EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
  }
}