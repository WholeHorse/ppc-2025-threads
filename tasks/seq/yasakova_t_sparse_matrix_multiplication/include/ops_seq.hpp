#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Complex = std::complex<double>;

namespace yasakova_t_sparse_matrix_multiplication_seq {

struct SparseMatrixCRS {
  std::vector<Complex> data;
  std::vector<int> columnIndices;
  std::vector<int> rowPointers;
  int rowCount;
  int columnCount;

  // Конструктор по умолчанию
  SparseMatrixCRS() : data({}), columnIndices({}), rowPointers({}), rowCount(0), columnCount(0) {}

  // Конструктор с параметрами
  SparseMatrixCRS(int rows, bool is_quadric, int cols) : rowCount(rows), columnCount(cols) {
    rowPointers.resize(rows + 1, 0);  // Инициализация указателей на строки
  }

  // Вставка элемента в матрицу
  void InsertElement(int row, Complex value, int col) {
    // Поиск элемента в строке
    for (int j = rowPointers[row]; j < rowPointers[row + 1]; ++j) {
      if (columnIndices[j] == col) {
        data[j] += value;  // Если элемент уже существует, добавляем значение
        return;
      }
    }
    // Если элемент не найден, добавляем его
    columnIndices.emplace_back(col);
    data.emplace_back(value);
    // Обновляем указатели на строки
    for (int i = row + 1; i <= rowCount; ++i) {
      rowPointers[i]++;
    }
  }

  // Конструктор копирования
  SparseMatrixCRS(const SparseMatrixCRS& other) = default;

  // Оператор присваивания
  SparseMatrixCRS& operator=(const SparseMatrixCRS& other) = default;
};

// Преобразование матрицы в вектор
std::vector<Complex> ConvertMatrixToVector(const SparseMatrixCRS& matrix);

// Преобразование вектора в матрицу
SparseMatrixCRS ConvertVectorToMatrix(std::vector<Complex>& vector);

// Сравнение двух матриц
bool AreMatricesEqual(const SparseMatrixCRS& matrix1, const SparseMatrixCRS& matrix2);

// Класс для тестирования умножения матриц
class SequentialMatrixMultiplicationTest : public ppc::core::Task {
 public:
  explicit SequentialMatrixMultiplicationTest(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Complex> input_data_;
  std::vector<Complex> output_data_;
  SparseMatrixCRS matrixA_;
  SparseMatrixCRS matrixB_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication_seq