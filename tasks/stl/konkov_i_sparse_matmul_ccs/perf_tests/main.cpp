#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/konkov_i_sparse_matmul_ccs/include/ops_stl.hpp"

TEST(konkov_i_SparseMatmulPerfTest_stl, test_pipeline_run) {
  constexpr int kSize = 5000;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<konkov_i_sparse_matmul_ccs_stl::SparseMatmulTask>(task_data);

  std::vector<double> a_values(kSize, 2.0);
  std::vector<int> a_row_indices(kSize);
  std::vector<int> a_col_ptr(kSize + 1);

  std::vector<double> b_values(kSize, 3.0);
  std::vector<int> b_row_indices(kSize);
  std::vector<int> b_col_ptr(kSize + 1);

  for (int i = 0; i < kSize; i++) {
    a_row_indices[i] = i;
    a_col_ptr[i] = i;
    b_row_indices[i] = i;
    b_col_ptr[i] = i;
  }
  a_col_ptr[kSize] = kSize;
  b_col_ptr[kSize] = kSize;

  task->A_values = a_values;
  task->A_row_indices = a_row_indices;
  task->A_col_ptr = a_col_ptr;
  task->rowsA = kSize;
  task->colsA = kSize;

  task->B_values = b_values;
  task->B_row_indices = b_row_indices;
  task->B_col_ptr = b_col_ptr;
  task->rowsB = kSize;
  task->colsB = kSize;

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

  const double expected_value = 6.0;
  for (const auto& val : task->C_values) {
    ASSERT_NEAR(val, expected_value, 1e-9);
  }
  ASSERT_EQ(task->C_col_ptr.back(), kSize);
}

TEST(konkov_i_SparseMatmulPerfTest_stl, test_task_run) {
  constexpr int kSize = 5000;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<konkov_i_sparse_matmul_ccs_stl::SparseMatmulTask>(task_data);

  std::vector<double> a_values(kSize, 2.0);
  std::vector<int> a_row_indices(kSize);
  std::vector<int> a_col_ptr(kSize + 1);

  std::vector<double> b_values(kSize, 3.0);
  std::vector<int> b_row_indices(kSize);
  std::vector<int> b_col_ptr(kSize + 1);

  for (int i = 0; i < kSize; i++) {
    a_row_indices[i] = i;
    a_col_ptr[i] = i;
    b_row_indices[i] = i;
    b_col_ptr[i] = i;
  }
  a_col_ptr[kSize] = kSize;
  b_col_ptr[kSize] = kSize;

  task->A_values = a_values;
  task->A_row_indices = a_row_indices;
  task->A_col_ptr = a_col_ptr;
  task->rowsA = kSize;
  task->colsA = kSize;

  task->B_values = b_values;
  task->B_row_indices = b_row_indices;
  task->B_col_ptr = b_col_ptr;
  task->rowsB = kSize;
  task->colsB = kSize;

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

  const double expected_value = 6.0;
  for (const auto& val : task->C_values) {
    ASSERT_NEAR(val, expected_value, 1e-9);
  }
  ASSERT_EQ(task->C_col_ptr.back(), kSize);
}
