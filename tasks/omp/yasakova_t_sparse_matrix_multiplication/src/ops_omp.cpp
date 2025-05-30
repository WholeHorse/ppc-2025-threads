#include "omp/yasakova_t_sparse_matrix_multiplication/include/ops_omp.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {
SparseMatrixFormat TransposeMatrixCRS(const SparseMatrixFormat &input_matrix) {
  const auto new_cols = input_matrix.RowCount();

  SparseMatrixFormat transposed_matrix;
  transposed_matrix.columns = new_cols;
  transposed_matrix.row_pointers.resize(input_matrix.ColumnCount() + 2);
  transposed_matrix.column_indices.resize(input_matrix.column_indices.size(), 0);
  transposed_matrix.task_data.resize(input_matrix.task_data.size(), 0);

  for (uint32_t i = 0; i < input_matrix.task_data.size(); ++i) {
    ++transposed_matrix.row_pointers[input_matrix.column_indices[i] + 2];
  }
  for (uint32_t i = 2; i < transposed_matrix.row_pointers.size(); ++i) {
    transposed_matrix.row_pointers[i] += transposed_matrix.row_pointers[i - 1];
  }
  for (uint32_t i = 0; i < new_cols; ++i) {
    for (uint32_t j = input_matrix.row_pointers[i]; j < input_matrix.row_pointers[i + 1]; ++j) {
      const auto new_index = transposed_matrix.row_pointers[input_matrix.column_indices[j] + 1]++;
      transposed_matrix.task_data[new_index] = input_matrix.task_data[j];
      transposed_matrix.column_indices[new_index] = i;
    }
  }
  transposed_matrix.row_pointers.pop_back();

  return transposed_matrix;
}
}  // namespace

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::ValidationImpl() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::PreProcessingImpl() {
  left_matrix_ = *reinterpret_cast<SparseMatrixFormat *>(task_data->inputs[0]);
  transposed_right_matrix_ = TransposeMatrixCRS(*reinterpret_cast<SparseMatrixFormat *>(task_data->inputs[1]));
  result_matrix_ = {};
  result_matrix_.row_pointers.resize(left_matrix_.RowCount() + 1);
  result_matrix_.columns = transposed_right_matrix_.RowCount();
  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::RunImpl() {
  const auto num_rows = left_matrix_.RowCount();
  const auto num_cols = transposed_right_matrix_.RowCount();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> non_zero_elements(num_rows);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_rows); ++i) {
    for (uint32_t j = 0; j < num_cols; ++j) {
      auto left_element_ptr = left_matrix_.row_pointers[i];
      auto right_element_ptr = transposed_right_matrix_.row_pointers[j];
      std::complex<double> product_sum = 0.0;
      while (left_element_ptr < left_matrix_.row_pointers[i + 1] &&
             right_element_ptr < transposed_right_matrix_.row_pointers[j + 1]) {
        if (left_matrix_.column_indices[left_element_ptr] <
            transposed_right_matrix_.column_indices[right_element_ptr]) {
          ++left_element_ptr;
        } else if (left_matrix_.column_indices[left_element_ptr] >
                   transposed_right_matrix_.column_indices[right_element_ptr]) {
          ++right_element_ptr;
        } else {
          product_sum +=
              left_matrix_.task_data[left_element_ptr++] * transposed_right_matrix_.task_data[right_element_ptr++];
        }
      }
      if (product_sum != 0.0) {
        non_zero_elements[i].emplace_back(product_sum, j);
      }
    }
  }

  for (uint32_t i = 0; i < num_rows; i++) {
    result_matrix_.row_pointers[i + 1] = result_matrix_.row_pointers[i];
    for (const auto &[product_sum, j] : non_zero_elements[i]) {
      result_matrix_.task_data.push_back(product_sum);
      result_matrix_.column_indices.push_back(j);
      ++result_matrix_.row_pointers[i + 1];
    }
  }

  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixFormat *>(task_data->outputs[0]) = result_matrix_;
  return true;
}