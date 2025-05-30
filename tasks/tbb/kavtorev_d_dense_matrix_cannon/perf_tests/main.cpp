// Copyright 2025 Kavtorev Dmitry
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/kavtorev_d_dense_matrix_cannon/include/ops_tbb.hpp"

namespace {
std::vector<double> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(1.0, 20.0);

  std::vector<double> matrix(rows * cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[(i * cols) + j] = dis(gen);
    }
  }

  return matrix;
}
}  // namespace

TEST(kavtorev_d_dense_matrix_cannon_tbb, test_pipeline_run) {
  int n = 500;
  int m = 500;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  auto test_task_omp = std::make_shared<kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, test_task_run) {
  int n = 500;
  int m = 500;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  auto test_task_omp = std::make_shared<kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}
