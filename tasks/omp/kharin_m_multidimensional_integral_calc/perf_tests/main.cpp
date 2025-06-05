#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/kharin_m_multidimensional_integral_calc/include/ops_omp.hpp"

TEST(kharin_m_multidimensional_integral_calc_omp, test_pipeline_run) {
  constexpr size_t kDim = 10000;

  std::vector<double> in(kDim * kDim, 1.0);
  std::vector<size_t> grid_sizes = {kDim, kDim};
  std::vector<double> step_sizes = {0.01, 0.01};
  std::vector<double> out(1, 0.0);
  double expected_out = static_cast<double>(kDim * kDim) * 0.01 * 0.01;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const double t0 = omp_get_wtime();
  perf_attr->current_timer = [t0] { return omp_get_wtime() - t0; };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}

TEST(kharin_m_multidimensional_integral_calc_omp, test_task_run) {
  constexpr size_t kDim = 10000;

  std::vector<double> in(kDim * kDim, 1.0);
  std::vector<size_t> grid_sizes = {kDim, kDim};
  std::vector<double> step_sizes = {0.01, 0.01};
  std::vector<double> out(1, 0.0);
  double expected_out = static_cast<double>(kDim * kDim) * 0.01 * 0.01;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const double t0 = omp_get_wtime();
  perf_attr->current_timer = [t0] { return omp_get_wtime() - t0; };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}