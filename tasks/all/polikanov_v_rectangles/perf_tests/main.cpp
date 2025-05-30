#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
template <bool Pipeline>
void RunPerfTest() {
  boost::mpi::communicator world;

  std::vector<polikanov_v_rectangles::IntegrationBound> bounds(4, {-1.0, 1.0});
  std::size_t discretization = 70;
  polikanov_v_rectangles::FunctionExecutor function = [](const auto &args) {
    return (args[0] * args[1]) + std::pow(args[1], 2);
  };
  double out = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&discretization));
    task_data->inputs_count.emplace_back(bounds.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  auto task = std::make_shared<polikanov_v_rectangles::TaskAll>(task_data, function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  ppc::core::Perf perf_analyzer(task);
  if constexpr (Pipeline) {
    perf_analyzer.PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer.TaskRun(perf_attr, perf_results);
  }

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_NEAR(out, 5.33, 0.2);
  }
}
}  // namespace

TEST(polikanov_v_rectangles_all, test_pipeline_run) { RunPerfTest<true>(); }

TEST(polikanov_v_rectangles_all, test_task_run) { RunPerfTest<false>(); }
