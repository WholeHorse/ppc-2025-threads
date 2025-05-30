#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace korneeva_e_all = korneeva_e_sparse_matrix_mult_complex_ccs_all;

namespace {
korneeva_e_all::SparseMatrixCCS CreateRandomMatrix(int rows, int cols, int max_nnz) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  korneeva_e_all::SparseMatrixCCS matrix(rows, cols, 0);
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  std::uniform_int_distribution<> row_dis(0, rows - 1);

  std::vector<std::vector<korneeva_e_all::Complex>> temp(rows, std::vector<korneeva_e_all::Complex>(cols, {0.0, 0.0}));
  int nnz = 0;
  while (nnz < max_nnz && nnz < rows * cols) {
    int r = row_dis(gen);
    int c = row_dis(gen) % cols;
    if (temp[r][c] == korneeva_e_all::Complex(0.0, 0.0)) {
      temp[r][c] = korneeva_e_all::Complex(dis(gen), dis(gen));
      nnz++;
    }
  }

  matrix.nnz = nnz;
  matrix.values.reserve(nnz);
  matrix.row_indices.reserve(nnz);
  matrix.col_offsets.resize(cols + 1, 0);

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (temp[i][j] != korneeva_e_all::Complex(0.0, 0.0)) {
        matrix.values.push_back(temp[i][j]);
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

bool AreComplexNumbersApproxEqual(const korneeva_e_all::Complex& c1, const korneeva_e_all::Complex& c2,
                                  double tolerance = 1e-6) {
  return std::abs(c1.real() - c2.real()) < tolerance && std::abs(c1.imag() - c2.imag()) < tolerance;
}

korneeva_e_all::SparseMatrixCCS SequentialMatrixMultiply(const korneeva_e_all::SparseMatrixCCS& m1,
                                                         const korneeva_e_all::SparseMatrixCCS& m2) {
  korneeva_e_all::SparseMatrixCCS result(m1.rows, m2.cols, 0);
  result.col_offsets.resize(m2.cols + 1, 0);
  std::vector<std::vector<korneeva_e_all::Complex>> temp(m1.rows,
                                                         std::vector<korneeva_e_all::Complex>(m2.cols, {0.0, 0.0}));

  for (int j = 0; j < m2.cols; j++) {
    for (int k = m2.col_offsets[j]; k < m2.col_offsets[j + 1]; k++) {
      int row_m2 = m2.row_indices[k];
      for (int i = m1.col_offsets[row_m2]; i < m1.col_offsets[row_m2 + 1]; i++) {
        int row_m1 = m1.row_indices[i];
        temp[row_m1][j] += m1.values[i] * m2.values[k];
      }
    }
  }

  for (int j = 0; j < m2.cols; j++) {
    for (int i = 0; i < m1.rows; i++) {
      if (temp[i][j] != korneeva_e_all::Complex(0.0, 0.0)) {
        result.values.push_back(temp[i][j]);
        result.row_indices.push_back(i);
      }
    }
    result.col_offsets[j + 1] = static_cast<int>(result.values.size());
  }
  result.nnz = static_cast<int>(result.values.size());
  return result;
}

void AssertVectorValuesEqual(const std::vector<korneeva_e_all::Complex>& result_values,
                             const std::vector<korneeva_e_all::Complex>& expected_values,
                             const std::vector<int>& result_indices, const std::vector<int>& expected_indices) {
  ASSERT_EQ(result_values.size(), expected_values.size());
  ASSERT_EQ(result_indices.size(), expected_indices.size());
  for (size_t i = 0; i < result_values.size(); i++) {
    ASSERT_TRUE(AreComplexNumbersApproxEqual(result_values[i], expected_values[i]));
    ASSERT_EQ(result_indices[i], expected_indices[i]);
  }
}

void AssertVectorOffsetsEqual(const std::vector<int>& result_offsets, const std::vector<int>& expected_offsets) {
  ASSERT_EQ(result_offsets.size(), expected_offsets.size());
  for (size_t i = 0; i < result_offsets.size(); i++) {
    ASSERT_EQ(result_offsets[i], expected_offsets[i]);
  }
}

void AssertMatricesEqual(const korneeva_e_all::SparseMatrixCCS& result,
                         const korneeva_e_all::SparseMatrixCCS& expected) {
  ASSERT_EQ(result.rows, expected.rows);
  ASSERT_EQ(result.cols, expected.cols);
  ASSERT_EQ(result.nnz, expected.nnz);

  AssertVectorValuesEqual(result.values, expected.values, result.row_indices, expected.row_indices);
  AssertVectorOffsetsEqual(result.col_offsets, expected.col_offsets);
}
}  // namespace

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_pipeline_run) {
  static boost::mpi::communicator world;
  constexpr int kRowsCols = 500;
  constexpr int kMaxNnzMatrix = 500;
  constexpr int kMaxNnzMatrix2 = 500;

  korneeva_e_all::SparseMatrixCCS matrix1({kRowsCols, kRowsCols});
  korneeva_e_all::SparseMatrixCCS matrix2({kRowsCols, kRowsCols});
  korneeva_e_all::SparseMatrixCCS result({kRowsCols, kRowsCols});

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix);
    matrix2 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix2);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix1));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix2));
    task_data_all->inputs_count.emplace_back(2);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data_all->outputs_count.emplace_back(1);
  }

  auto test_task_all = std::make_shared<korneeva_e_all::SparseMatrixMultComplexCCS>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    korneeva_e_all::SparseMatrixCCS expected = SequentialMatrixMultiply(matrix1, matrix2);
    AssertMatricesEqual(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_task_run) {
  static boost::mpi::communicator world;
  constexpr int kRowsCols = 500;
  constexpr int kMaxNnzMatrix = 500;
  constexpr int kMaxNnzMatrix2 = 500;

  korneeva_e_all::SparseMatrixCCS matrix1({kRowsCols, kRowsCols});
  korneeva_e_all::SparseMatrixCCS matrix2({kRowsCols, kRowsCols});
  korneeva_e_all::SparseMatrixCCS result({kRowsCols, kRowsCols});

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix);
    matrix2 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix2);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix1));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix2));
    task_data_all->inputs_count.emplace_back(2);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data_all->outputs_count.emplace_back(1);
  }

  auto test_task_all = std::make_shared<korneeva_e_all::SparseMatrixMultComplexCCS>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    korneeva_e_all::SparseMatrixCCS expected = SequentialMatrixMultiply(matrix1, matrix2);
    AssertMatricesEqual(result, expected);
  }
}