#pragma once

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct MatrixStructure {
  uint32_t num_rows;
  uint32_t num_cols;
  std::vector<std::complex<double>> task_data;

  std::complex<double>& AccessElement(uint32_t row, uint32_t col) { return task_data[(row * num_cols) + col]; }

  bool operator==(const MatrixStructure& other) const noexcept {
    return num_rows == other.num_rows && num_cols == other.num_cols && task_data == other.task_data;
  }
};

struct SparseMatrixFormat {
  std::vector<std::complex<double>> task_data;

  uint32_t columns;
  std::vector<uint32_t> row_pointers;
  std::vector<uint32_t> column_indices;

  [[nodiscard]] uint32_t RowCount() const { return row_pointers.size() - 1; }
  [[nodiscard]] uint32_t ColumnCount() const { return columns; }

  bool operator==(const SparseMatrixFormat& other) const noexcept {
    return columns == other.columns && row_pointers == other.row_pointers && column_indices == other.column_indices &&
           task_data == other.task_data;
  }
};

inline MatrixStructure MatrixMultiply(MatrixStructure& matrix_left, MatrixStructure& matrix_right) {
  MatrixStructure result{.num_rows = matrix_left.num_rows,
                         .num_cols = matrix_right.num_cols,
                         .task_data = std::vector<std::complex<double>>(matrix_left.num_rows * matrix_right.num_cols)};
  for (uint32_t i = 0; i < matrix_left.num_rows; i++) {
    for (uint32_t j = 0; j < matrix_right.num_cols; j++) {
      result.AccessElement(i, j) = 0;
      for (uint32_t k = 0; k < matrix_right.num_rows; k++) {
        result.AccessElement(i, j) += matrix_left.AccessElement(i, k) * matrix_right.AccessElement(k, j);
      }
    }
  }
  return result;
}

inline SparseMatrixFormat ConvertToCRS(const MatrixStructure& matrix) {
  SparseMatrixFormat result;
  result.row_pointers.resize(matrix.num_rows + 1);
  result.columns = matrix.num_cols;

  uint32_t i = 0;
  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    uint32_t nz = 0;
    for (uint32_t col = 0; col < matrix.num_cols; ++col) {
      if (const auto& element = matrix.task_data[i++]; element != 0.0) {
        ++nz;
        result.column_indices.push_back(col);
        result.task_data.push_back(element);
      }
    }
    result.row_pointers[row + 1] = result.row_pointers[row] + nz;
  }

  return result;
}

inline MatrixStructure ConvertFromCRS(const SparseMatrixFormat& input_matrix) {
  MatrixStructure matrix{
      .num_rows = input_matrix.RowCount(),
      .num_cols = input_matrix.ColumnCount(),
      .task_data = std::vector<std::complex<double>>(input_matrix.RowCount() * input_matrix.ColumnCount())};
  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    for (uint32_t i = input_matrix.row_pointers[row]; i < input_matrix.row_pointers[row + 1]; ++i) {
      matrix.AccessElement(row, input_matrix.column_indices[i]) = input_matrix.task_data[i];
    }
  }
  return matrix;
}

inline SparseMatrixFormat CreateRandomSparseMatrix(uint32_t size, uint32_t non_zero_elements) {
  SparseMatrixFormat matrix;
  matrix.columns = size;
  matrix.row_pointers.resize(size + 1);
  std::srand(std::time(nullptr));
  for (uint32_t i = 0; i < non_zero_elements; i++) {
    uint32_t row = rand() % size;
    uint32_t col = rand() % size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    matrix.task_data.push_back(value);
    matrix.column_indices.push_back(col);
    matrix.row_pointers[row + 1]++;
  }

  // Calculate row pointers
  for (uint32_t i = 1; i <= size; i++) {
    matrix.row_pointers[i] += matrix.row_pointers[i - 1];
  }

  return matrix;
}

namespace yasakova_t_sparse_matrix_multiplication_omp {

class SparseMatrixMultiplier : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplier(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixFormat left_matrix_;
  SparseMatrixFormat transposed_right_matrix_;
  SparseMatrixFormat result_matrix_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication_omp