#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/durynichev_d_integrals_simpson_method/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(durynichev_d_integrals_simpson_method_all, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<double> in = {-10.0, 10.0, -10.0, 10.0, 29000};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI>(task_data);

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_NEAR(out[0], 80000.0 / 3.0, 1e-2);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_task_run) {
  boost::mpi::communicator world;

  std::vector<double> in = {-10.0, 10.0, -10.0, 10.0, 29000};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI>(task_data);

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_NEAR(out[0], 80000.0 / 3.0, 1e-2);
  }
}