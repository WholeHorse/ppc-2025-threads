#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace yasakova_t_sparse_matrix_multiplication_seq {

// Функция для сравнения двух разреженных матриц
bool AreMatricesEqual(const SparseMatrixCRS& matrix1, const SparseMatrixCRS& matrix2) {
  if (matrix1.columnCount != matrix2.columnCount || matrix1.rowCount != matrix2.rowCount) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)matrix1.rowCount; ++i) {
    unsigned int row_start1 = matrix1.rowPointers[i];
    unsigned int row_end1 = matrix1.rowPointers[i + 1];
    unsigned int row_start2 = matrix2.rowPointers[i];
    unsigned int row_end2 = matrix2.rowPointers[i + 1];

    if ((row_end1 - row_start1) != (row_end2 - row_start2)) {
      return false;
    }

    for (unsigned int j = row_start1; j < row_end1; ++j) {
      bool found = false;
      for (unsigned int k = row_start2; k < row_end2; ++k) {
        if (matrix1.columnIndices[j] == matrix2.columnIndices[k] && matrix1.data[j] == matrix2.data[k]) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

// Функция для преобразования вектора в разреженную матрицу
SparseMatrixCRS ConvertVectorToMatrix(std::vector<Complex>& vector) {
  SparseMatrixCRS result;
  result.rowCount = (int)vector[0].real();
  result.columnCount = (int)vector[1].real();
  auto data_size = (unsigned int)vector[2].real();
  auto col_indices_size = (unsigned int)vector[3].real();
  auto row_ptr_size = (unsigned int)vector[4].real();
  result.data.reserve(data_size);
  result.columnIndices.reserve(col_indices_size);
  result.rowPointers.reserve(row_ptr_size);
  for (unsigned int i = 0; i < data_size; i++) {
    result.data.emplace_back(vector[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    result.columnIndices.emplace_back((int)vector[5 + data_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    result.rowPointers.emplace_back((int)vector[5 + data_size + col_indices_size + i].real());
  }
  return result;
}

// Функция для преобразования разреженной матрицы в вектор
std::vector<Complex> ConvertMatrixToVector(const SparseMatrixCRS& matrix) {
  std::vector<Complex> result = {};
  result.reserve(5 + matrix.data.size() + matrix.columnIndices.size() + matrix.rowPointers.size());
  result.emplace_back((double)matrix.rowCount);
  result.emplace_back((double)matrix.columnCount);
  result.emplace_back((double)matrix.data.size());
  result.emplace_back((double)matrix.columnIndices.size());
  result.emplace_back((double)matrix.rowPointers.size());
  for (unsigned int i = 0; i < (unsigned int)matrix.data.size(); i++) {
    result.emplace_back(matrix.data[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)matrix.columnIndices.size(); i++) {
    result.emplace_back(matrix.columnIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)matrix.rowPointers.size(); i++) {
    result.emplace_back(matrix.rowPointers[i]);
  }
  return result;
}

// Реализация класса SequentialMatrixMultiplicationTest
bool SequentialMatrixMultiplicationTest::PreProcessingImpl() {
  // Инициализация входных данных
  unsigned int input_data_size = task_data->inputs_count[0];
  auto* input_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_data_ = std::vector<Complex>(input_ptr, input_ptr + input_data_size);
  std::vector<Complex> matrix_a = {};
  std::vector<Complex> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(input_data_[2].real() + input_data_[3].real() + input_data_[4].real()));
  matrix_b.reserve(input_data_.size() -
                   (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real()));
  for (unsigned int i = 0;
       i < (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real()); i++) {
    matrix_a.emplace_back(input_data_[i]);
  }
  for (auto i = (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real());
       i < (unsigned int)input_data_.size(); i++) {
    matrix_b.emplace_back(input_data_[i]);
  }
  matrixA_ = ConvertVectorToMatrix(matrix_a);
  matrixB_ = ConvertVectorToMatrix(matrix_b);
  return true;
}

bool SequentialMatrixMultiplicationTest::ValidationImpl() {
  // Проверка корректности входных данных
  unsigned int input_data_size = task_data->inputs_count[0];
  auto* input_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  std::vector<Complex> vector = std::vector<Complex>(input_ptr, input_ptr + input_data_size);
  return !(vector[1] != vector[5 + (int)(vector[2].real() + vector[3].real() + vector[4].real())].real());
}

bool SequentialMatrixMultiplicationTest::RunImpl() {
  // Умножение матриц
  bool is_quadric = (matrixA_.rowCount == matrixB_.columnCount);
  SparseMatrixCRS result(matrixA_.rowCount, is_quadric, matrixB_.columnCount);
  for (unsigned int i = 0; i < (unsigned int)matrixA_.rowCount; ++i) {
    for (unsigned int j = matrixA_.rowPointers[i]; j < (unsigned int)matrixA_.rowPointers[i + 1]; ++j) {
      unsigned int col_a = matrixA_.columnIndices[j];
      Complex value_a = matrixA_.data[j];
      for (unsigned int k = matrixB_.rowPointers[col_a]; k < (unsigned int)matrixB_.rowPointers[col_a + 1]; ++k) {
        unsigned int col_b = matrixB_.columnIndices[k];
        Complex value_b = matrixB_.data[k];

        result.InsertElement((int)i, value_a * value_b, (int)col_b);
      }
    }
  }
  output_data_ = ConvertMatrixToVector(result);
  return true;
}

bool SequentialMatrixMultiplicationTest::PostProcessingImpl() {
  // Сохранение результата
  for (size_t i = 0; i < output_data_.size(); i++) {
    reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_data_[i];
  }
  return true;
}

}  // namespace yasakova_t_sparse_matrix_multiplication_seq