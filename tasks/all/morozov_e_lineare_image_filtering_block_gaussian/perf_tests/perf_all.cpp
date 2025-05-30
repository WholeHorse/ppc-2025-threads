#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/morozov_e_lineare_image_filtering_block_gaussian/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(morozov_e_lineare_image_filtering_block_gaussian_all, test_pipeline_run) {
  int n = 1000;
  int m = 1000;
  std::vector<double> image(n * m, 1.0);
  std::vector<double> image_res(n * m, 1.0);
  std::vector real_res(n * m, 1.0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_all->outputs_count.emplace_back(n);
  task_data_all->outputs_count.emplace_back(m);

  // Create Task
  auto test_task_all =
      std::make_shared<morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL>(task_data_all);

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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  if (world.rank() == 0) {
    ASSERT_EQ(image_res.size(), real_res.size());
    for (size_t i = 0; i < image_res.size(); ++i) {
      ASSERT_NEAR(image_res[i], real_res[i], 0.0000001);
    }
  }
}

TEST(morozov_e_lineare_image_filtering_block_gaussian_all, test_task_run) {
  int n = 1000;
  int m = 1000;
  std::vector<double> image(n * m, 1.0);
  std::vector<double> image_res(n * m, 1.0);
  std::vector real_res(n * m, 1.0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_all->outputs_count.emplace_back(n);
  task_data_all->outputs_count.emplace_back(m);

  // Create Task
  auto test_task_all =
      std::make_shared<morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL>(task_data_all);

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
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  if (world.rank() == 0) {
    ASSERT_EQ(image_res, real_res);
  }
}
