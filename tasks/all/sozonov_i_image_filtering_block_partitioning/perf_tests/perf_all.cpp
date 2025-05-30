#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace sozonov_i_image_filtering_block_partitioning_all {

std::vector<double> ZeroEdges(std::vector<double> img, int wdth, int hght) {
  for (int i = 0; i < wdth; ++i) {
    img[i] = 0;
    img[((hght - 1) * wdth) + i] = 0;
  }
  for (int i = 1; i < hght - 1; ++i) {
    img[i * wdth] = 0.75;
    img[(i * wdth) + wdth - 1] = 0.75;
  }

  return img;
}

}  // namespace sozonov_i_image_filtering_block_partitioning_all

TEST(sozonov_i_image_filtering_block_partitioning_all, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 3000;
  const int height = 3000;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<sozonov_i_image_filtering_block_partitioning_all::TestTaskALL>(task_data_all);

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_task_run) {
  boost::mpi::communicator world;

  const int width = 3000;
  const int height = 3000;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<sozonov_i_image_filtering_block_partitioning_all::TestTaskALL>(task_data_all);

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out, ans);
  }
}
