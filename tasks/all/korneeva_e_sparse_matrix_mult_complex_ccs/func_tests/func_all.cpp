#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace korneeva_e_all = korneeva_e_sparse_matrix_mult_complex_ccs_all;

namespace {
void RunTask(korneeva_e_all::SparseMatrixCCS& m1, korneeva_e_all::SparseMatrixCCS& m2,
             korneeva_e_all::SparseMatrixCCS& result, boost::mpi::communicator& world) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->inputs_count.emplace_back(2);
    task_data->outputs_count.emplace_back(1);
  }

  korneeva_e_all::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

void ExpectMatrixValuesEq(const korneeva_e_all::SparseMatrixCCS& result,
                          const korneeva_e_all::SparseMatrixCCS& expected, double epsilon) {
  ASSERT_EQ(result.values.size(), expected.values.size());
  for (size_t i = 0; i < result.values.size(); i++) {
    EXPECT_NEAR(std::abs(result.values[i] - expected.values[i]), 0.0, epsilon);
  }
}

void ExpectMatrixEq(const korneeva_e_all::SparseMatrixCCS& result, const korneeva_e_all::SparseMatrixCCS& expected,
                    double epsilon = 1e-6) {
  EXPECT_EQ(result.rows, expected.rows);
  EXPECT_EQ(result.cols, expected.cols);
  EXPECT_EQ(result.nnz, expected.nnz);
  EXPECT_EQ(result.col_offsets, expected.col_offsets);
  EXPECT_EQ(result.row_indices, expected.row_indices);
  ExpectMatrixValuesEq(result, expected, epsilon);
}

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

korneeva_e_all::SparseMatrixCCS CreateCcsFromDense(const std::vector<std::vector<korneeva_e_all::Complex>>& dense) {
  int rows = static_cast<int>(dense.size());
  int cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  korneeva_e_all::SparseMatrixCCS matrix(rows, cols, 0);

  matrix.col_offsets.resize(cols + 1, 0);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (dense[i][j] != korneeva_e_all::Complex(0.0, 0.0)) {
        matrix.values.push_back(dense[i][j]);
        matrix.row_indices.push_back(i);
        matrix.nnz++;
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}
}  // namespace

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_incompatible_sizes) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 3, 0);
  korneeva_e_all::SparseMatrixCCS m2(2, 2, 0);
  korneeva_e_all::SparseMatrixCCS result;

  if (world.rank() == 0) {
    m1.col_offsets = {0, 0, 0, 0};
    m2.col_offsets = {0, 0, 0};
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->inputs_count.emplace_back(2);
    task_data->outputs_count.emplace_back(1);
  }

  korneeva_e_all::SparseMatrixMultComplexCCS task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_negative_dimensions) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(-1, 2, 0);
  korneeva_e_all::SparseMatrixCCS m2(2, 2, 0);
  korneeva_e_all::SparseMatrixCCS result;

  if (world.rank() == 0) {
    m1.col_offsets = {0, 0, 0};
    m2.col_offsets = {0, 0, 0};
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
    task_data->inputs_count.emplace_back(2);
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }

  korneeva_e_all::SparseMatrixMultComplexCCS task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_empty_input) {
  static boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  korneeva_e_all::SparseMatrixCCS result;

  if (world.rank() == 0) {
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }

  korneeva_e_all::SparseMatrixMultComplexCCS task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_no_output) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2, 0);
  korneeva_e_all::SparseMatrixCCS m2(2, 2, 0);

  if (world.rank() == 0) {
    m1.col_offsets = {0, 0, 0};
    m2.col_offsets = {0, 0, 0};
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
    task_data->inputs_count.emplace_back(2);
  }

  korneeva_e_all::SparseMatrixMultComplexCCS task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_identity_mult) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                                        {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_left_identity_mult) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS i(3, 3);
  korneeva_e_all::SparseMatrixCCS a(3, 2);
  korneeva_e_all::SparseMatrixCCS result(3, 2);

  if (world.rank() == 0) {
    i = CreateCcsFromDense(
        {{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
         {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
         {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
    a = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 2.0), korneeva_e_all::Complex(3.0, 4.0)},
                            {korneeva_e_all::Complex(5.0, 0.0), korneeva_e_all::Complex(7.0, 8.0)},
                            {korneeva_e_all::Complex(9.0, 10.0), korneeva_e_all::Complex(11.0, 12.0)}});
  }

  RunTask(i, a, result, world);

  if (world.rank() == 0) {
    ExpectMatrixEq(result, a);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_zero_matrix) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                                        {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_full_zero_matrix) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                                        {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_diagonal_matrices) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(2.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(3.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(4.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(5.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(8.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                                        {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(15.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_complex_numbers) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(1, 1);
  korneeva_e_all::SparseMatrixCCS m2(1, 1);
  korneeva_e_all::SparseMatrixCCS result(1, 1);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 1.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, -1.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_large_complex_values) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(1, 1);
  korneeva_e_all::SparseMatrixCCS m2(1, 1);
  korneeva_e_all::SparseMatrixCCS result(1, 1);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1e10, 1e10)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(1e10, -1e10)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(2e20, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_rectangular_matrices) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 3);
  korneeva_e_all::SparseMatrixCCS m2(3, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense(
        {{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(2.0, 0.0), korneeva_e_all::Complex(3.0, 0.0)},
         {korneeva_e_all::Complex(4.0, 0.0), korneeva_e_all::Complex(5.0, 0.0), korneeva_e_all::Complex(6.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(7.0, 0.0), korneeva_e_all::Complex(8.0, 0.0)},
                             {korneeva_e_all::Complex(9.0, 0.0), korneeva_e_all::Complex(10.0, 0.0)},
                             {korneeva_e_all::Complex(11.0, 0.0), korneeva_e_all::Complex(12.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(58.0, 0.0), korneeva_e_all::Complex(64.0, 0.0)},
                                        {korneeva_e_all::Complex(139.0, 0.0), korneeva_e_all::Complex(154.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_matrix_vector_mult) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS vec(2, 1);
  korneeva_e_all::SparseMatrixCCS result(2, 1);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 2.0), korneeva_e_all::Complex(3.0, 4.0)},
                             {korneeva_e_all::Complex(5.0, 6.0), korneeva_e_all::Complex(7.0, 8.0)}});
    vec = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0)}, {korneeva_e_all::Complex(2.0, 0.0)}});
  }

  RunTask(m1, vec, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(7.0, 10.0)}, {korneeva_e_all::Complex(19.0, 22.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_vector_matrix_mult) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS vec(1, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(1, 2);

  if (world.rank() == 0) {
    vec = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(3.0, 0.0), korneeva_e_all::Complex(4.0, 0.0)},
                             {korneeva_e_all::Complex(5.0, 0.0), korneeva_e_all::Complex(6.0, 0.0)}});
  }

  RunTask(vec, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(13.0, 0.0), korneeva_e_all::Complex(16.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_matrix_unit_vector) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS vec(2, 1);
  korneeva_e_all::SparseMatrixCCS result(2, 1);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 1.0), korneeva_e_all::Complex(2.0, 2.0)},
                             {korneeva_e_all::Complex(3.0, 3.0), korneeva_e_all::Complex(4.0, 4.0)}});
    vec = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0)}, {korneeva_e_all::Complex(0.0, 0.0)}});
  }

  RunTask(m1, vec, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 1.0)}, {korneeva_e_all::Complex(3.0, 3.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_sparse_matrices) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 3);
  korneeva_e_all::SparseMatrixCCS m2(3, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense(
        {{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
         {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(3.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(4.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(4.0, 0.0)},
                                        {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_mixed_values) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(3.0, 0.0)},
                             {korneeva_e_all::Complex(4.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(3.0, 0.0)},
                                        {korneeva_e_all::Complex(8.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_dense_sparse_mult) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)},
                             {korneeva_e_all::Complex(3.0, 0.0), korneeva_e_all::Complex(4.0, 0.0)}});
    m2 = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                             {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    auto expected = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                                        {korneeva_e_all::Complex(3.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)}});
    ExpectMatrixEq(result, expected);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_random_matrices1) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(2, 2);
  korneeva_e_all::SparseMatrixCCS m2(2, 2);
  korneeva_e_all::SparseMatrixCCS result(2, 2);

  if (world.rank() == 0) {
    m1 = CreateRandomMatrix(2, 2, 2);
    m2 = CreateRandomMatrix(2, 2, 2);
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    ASSERT_EQ(result.rows, 2);
    ASSERT_EQ(result.cols, 2);
    EXPECT_LE(result.nnz, 4);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_random_matrices2) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS m1(100, 100);
  korneeva_e_all::SparseMatrixCCS m2(100, 100);
  korneeva_e_all::SparseMatrixCCS result(100, 100);

  if (world.rank() == 0) {
    m1 = CreateRandomMatrix(100, 100, 500);
    m2 = CreateRandomMatrix(100, 100, 500);
  }

  RunTask(m1, m2, result, world);

  if (world.rank() == 0) {
    ASSERT_EQ(result.rows, 100);
    ASSERT_EQ(result.cols, 100);
    EXPECT_LE(result.nnz, 10000);
  }
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_all, test_associativity) {
  static boost::mpi::communicator world;
  korneeva_e_all::SparseMatrixCCS a(2, 2);
  korneeva_e_all::SparseMatrixCCS b(2, 2);
  korneeva_e_all::SparseMatrixCCS c(2, 2);
  korneeva_e_all::SparseMatrixCCS ab(2, 2);
  korneeva_e_all::SparseMatrixCCS ab_c(2, 2);
  korneeva_e_all::SparseMatrixCCS bc(2, 2);
  korneeva_e_all::SparseMatrixCCS a_bc(2, 2);

  if (world.rank() == 0) {
    a = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)},
                            {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(3.0, 0.0)}});
    b = CreateCcsFromDense({{korneeva_e_all::Complex(4.0, 0.0), korneeva_e_all::Complex(0.0, 0.0)},
                            {korneeva_e_all::Complex(5.0, 0.0), korneeva_e_all::Complex(6.0, 0.0)}});
    c = CreateCcsFromDense({{korneeva_e_all::Complex(1.0, 0.0), korneeva_e_all::Complex(2.0, 0.0)},
                            {korneeva_e_all::Complex(0.0, 0.0), korneeva_e_all::Complex(1.0, 0.0)}});
  }

  RunTask(a, b, ab, world);
  RunTask(ab, c, ab_c, world);
  RunTask(b, c, bc, world);
  RunTask(a, bc, a_bc, world);

  if (world.rank() == 0) {
    ExpectMatrixEq(ab_c, a_bc);
  }
}