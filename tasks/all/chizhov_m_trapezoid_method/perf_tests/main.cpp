#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "all/chizhov_m_trapezoid_method/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(chizhov_m_trapezoid_method_all, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> res(1, 0);

  int div = 200;
  int dim = 3;
  std::vector<double> limits = {0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0};

  auto f = [](const std::vector<double> &f_val) {
    return std::sin((f_val[0] * f_val[0]) + (f_val[1] * f_val[1]) + (f_val[2] * f_val[2]));
  };

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
    task_data_mpi->inputs_count.emplace_back(sizeof(div));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    task_data_mpi->inputs_count.emplace_back(sizeof(dim));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_mpi->inputs_count.emplace_back(limits.size());

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  auto test_task_mpi = std::make_shared<chizhov_m_trapezoid_method_all::TestTaskMPI>(task_data_mpi);
  test_task_mpi->SetFunc(f);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(-1405.6, res[0], 0.3);
  }
}

TEST(chizhov_m_trapezoid_method_all, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> res(1, 0);

  int div = 200;
  int dim = 3;
  std::vector<double> limits = {0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0};

  auto f = [](const std::vector<double> &f_val) {
    return std::sin((f_val[0] * f_val[0]) + (f_val[1] * f_val[1]) + (f_val[2] * f_val[2]));
  };

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
    task_data_mpi->inputs_count.emplace_back(sizeof(div));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    task_data_mpi->inputs_count.emplace_back(sizeof(dim));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_mpi->inputs_count.emplace_back(limits.size());

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  auto test_task_mpi = std::make_shared<chizhov_m_trapezoid_method_all::TestTaskMPI>(task_data_mpi);
  test_task_mpi->SetFunc(f);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(-1405.6, res[0], 0.3);
  }
}
