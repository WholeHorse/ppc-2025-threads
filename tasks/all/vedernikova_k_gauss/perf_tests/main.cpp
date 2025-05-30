#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "all/vedernikova_k_gauss/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

using Pars = std::tuple<uint32_t, uint32_t, uint32_t, uint8_t>;

TEST(vedernikova_k_gauss_test_all, test_pipeline_run) {
  boost::mpi::communicator world;
  const auto &[width, height, channels, brightness] = Pars(1000, 300, 4, 128);
  const uint32_t size = width * height * channels;
  std::vector<uint8_t> in(size, brightness);
  std::vector<uint8_t> out(size);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->inputs_count.emplace_back(channels);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<vedernikova_k_gauss_all::Gauss>(task_data_all);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(in, out);
  }
}

TEST(vedernikova_k_gauss_test_all, test_task_run) {
  boost::mpi::communicator world;
  const auto &[width, height, channels, brightness] = Pars(1000, 300, 4, 128);
  const uint32_t size = width * height * channels;
  std::vector<uint8_t> in(size, brightness);
  std::vector<uint8_t> out(size);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->inputs_count.emplace_back(channels);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<vedernikova_k_gauss_all::Gauss>(task_data_all);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(in, out);
  }
}
}  // namespace