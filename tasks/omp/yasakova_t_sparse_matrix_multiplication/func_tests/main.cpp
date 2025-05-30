#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication/include/ops_omp.hpp"

namespace {
MatrixStructure GenerateRandomMatrix(uint32_t num_rows, uint32_t num_cols, double non_zero_percentage) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> distribution(-10000, 10000);
  MatrixStructure result{
      .num_rows = num_rows, .num_cols = num_cols, .task_data = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.task_data, [&]() {
    const auto value = distribution(rng);
    const auto real_component =
        (value < (distribution.min() + ((distribution.max() - distribution.min()) * non_zero_percentage))) ? value : 0;

    std::complex<double> complex_number;
    complex_number.real(real_component);
    if (real_component != 0.0) {
      complex_number.imag(distribution(rng));
    }

    return complex_number;
  });
  return result;
}
void TestCRSMatrixMultiplication(MatrixStructure &&matrix_left, MatrixStructure &&matrix_right) {
  SparseMatrixFormat compressed_row_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat compressed_row_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat compressed_row_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&compressed_row_left),
                       reinterpret_cast<uint8_t *>(&compressed_row_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&compressed_row_result)};
  task_data->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier task(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  MatrixStructure actual_result = ConvertFromCRS(compressed_row_result);
  EXPECT_EQ(actual_result, MatrixMultiply(matrix_left, matrix_right));
}
}  // namespace

TEST(yasakova_t_sparse_matrix_multiplication_omp, MultiplySquareMatrices) {
  MatrixStructure matrix_left{.num_rows = 5, .num_cols = 4, .task_data = {43, 46, 21, 21, 39, 26, 82, 65, 97, 47,
                                                                          32, 16, 76, 43, 78, 50, 18, 14, 84, 22}};
  MatrixStructure matrix_right{.num_rows = 4, .num_cols = 5, .task_data = {43, 46, 21, 21, 87, 39, 26, 82, 65, 62,
                                                                           97, 47, 32, 16, 61, 76, 43, 78, 50, 63}};
  MatrixStructure ref{
      .num_rows = 5, .num_cols = 5, .task_data = {7276,  5064,  6985,  5279, 9197, 15585, 9119,  10645, 7071,
                                                  14102, 10324, 7876,  8163, 6404, 14313, 16311, 10430, 11518,
                                                  8139,  17186, 11140, 6086, 5930, 3732,  8944}};
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, MultiplyRectangularMatrices) {
  MatrixStructure matrix_left{.num_rows = 5, .num_cols = 4, .task_data = {43, 46, 21, 21, 39, 26, 82, 65, 97, 47,
                                                                          32, 16, 76, 43, 78, 50, 18, 14, 84, 22}};
  MatrixStructure matrix_right{.num_rows = 4, .num_cols = 5, .task_data = {43, 46, 21, 21, 87, 39, 26, 82, 65, 62,
                                                                           97, 47, 32, 16, 61, 76, 43, 78, 50, 63}};
  MatrixStructure ref{
      .num_rows = 5, .num_cols = 5, .task_data = {7276,  5064,  6985,  5279, 9197, 15585, 9119,  10645, 7071,
                                                  14102, 10324, 7876,  8163, 6404, 14313, 16311, 10430, 11518,
                                                  8139,  17186, 11140, 6086, 5930, 3732,  8944}};
  EXPECT_EQ(MatrixMultiply(matrix_left, matrix_right), ref);
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_FullyDense) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 30, .0), GenerateRandomMatrix(30, 30, .0));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_20PercentNonZero) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 30, .20), GenerateRandomMatrix(30, 30, .20));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_MixedDensity) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 30, .20), GenerateRandomMatrix(30, 30, .50));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x40_VaryingDensity) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 40, .70), GenerateRandomMatrix(40, 30, .60));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x23_70PercentNonZero) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 23, .70), GenerateRandomMatrix(23, 30, .63));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrix30x1_VeryHighDensity) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 1, .70), GenerateRandomMatrix(1, 30, .63));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrix30x1_LowDensity) {
  TestCRSMatrixMultiplication(GenerateRandomMatrix(30, 1, .38), GenerateRandomMatrix(1, 30, .63));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, InverseMatrixMultiplication) {
  MatrixStructure matrix_left{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, 1, 0, 1}};
  MatrixStructure matrix_right{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, -1, 0, 1}};
  MatrixStructure ref{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
  EXPECT_EQ(MatrixMultiply(matrix_left, matrix_right), ref);
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_inv) {
  TestCRSMatrixMultiplication({.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, 1, 0, 1}},
                              {.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, -1, 0, 1}});
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, ValidationFailure_IncompatibleDimensions) {
  const auto matrix_left = GenerateRandomMatrix(30, 40, .70);
  const auto matrix_right = GenerateRandomMatrix(50, 50, .70);

  SparseMatrixFormat compressed_row_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat compressed_row_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat compressed_row_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&compressed_row_left),
                       reinterpret_cast<uint8_t *>(&compressed_row_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&compressed_row_result)};
  task_data->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(YasakovaSparseMatrixMultiplicationOMP, IdentityMatrixMultiplication) {
  MatrixStructure identity{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
  MatrixStructure random_matrix = GenerateRandomMatrix(3, 3, 0.5);
  EXPECT_EQ(MatrixMultiply(identity, random_matrix), random_matrix);
}

TEST(YasakovaSparseMatrixMultiplicationOMP, ZeroMatrixMultiplication) {
  MatrixStructure zero{.num_rows = 3, .num_cols = 3, .task_data = {0, 0, 0, 0, 0, 0, 0, 0, 0}};
  MatrixStructure random_matrix = GenerateRandomMatrix(3, 3, 0.5);
  EXPECT_EQ(MatrixMultiply(zero, random_matrix), zero);
}