#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_real_matrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(3, true, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(1, 0), 0);
  mat_a.InsertElement(0, Complex(2, 0), 2);
  mat_a.InsertElement(1, Complex(3, 0), 1);
  mat_a.InsertElement(2, Complex(4, 0), 0);
  mat_a.InsertElement(2, Complex(5, 0), 1);

  mat_b.InsertElement(0, Complex(6, 0), 1);
  mat_b.InsertElement(1, Complex(7, 0), 0);
  mat_b.InsertElement(2, Complex(8, 0), 2);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }
  expected_result.InsertElement(0, Complex(6, 0), 1);
  expected_result.InsertElement(0, Complex(16, 0), 2);
  expected_result.InsertElement(1, Complex(21, 0), 0);
  expected_result.InsertElement(2, Complex(24, 0), 1);
  expected_result.InsertElement(2, Complex(35, 0), 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_imaginary_parts) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(3, true, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(1, 1), 0);
  mat_a.InsertElement(0, Complex(2, 2), 2);
  mat_a.InsertElement(1, Complex(3, 3), 1);
  mat_a.InsertElement(2, Complex(4, 4), 0);
  mat_a.InsertElement(2, Complex(5, 5), 1);

  mat_b.InsertElement(0, Complex(6, 6), 1);
  mat_b.InsertElement(1, Complex(7, 7), 0);
  mat_b.InsertElement(2, Complex(8, 8), 2);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }
  expected_result.InsertElement(0, Complex(0, 12), 1);
  expected_result.InsertElement(0, Complex(0, 32), 2);
  expected_result.InsertElement(1, Complex(0, 42), 0);
  expected_result.InsertElement(2, Complex(0, 48), 1);
  expected_result.InsertElement(2, Complex(0, 70), 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_rectangular_matrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(2, false, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, false, 4);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(2, false, 4);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(1, 0), 1);
  mat_a.InsertElement(0, Complex(2, 0), 2);
  mat_a.InsertElement(1, Complex(3, 0), 1);

  mat_b.InsertElement(0, Complex(3, 0), 2);
  mat_b.InsertElement(1, Complex(5, 0), 0);
  mat_b.InsertElement(1, Complex(4, 0), 3);
  mat_b.InsertElement(2, Complex(7, 0), 0);
  mat_b.InsertElement(2, Complex(8, 0), 1);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }
  expected_result.InsertElement(0, Complex(19, 0), 0);
  expected_result.InsertElement(0, Complex(4, 0), 3);
  expected_result.InsertElement(0, Complex(16, 0), 1);
  expected_result.InsertElement(1, Complex(15, 0), 0);
  expected_result.InsertElement(1, Complex(12, 0), 3);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_negative_elements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(2, true, 2);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(-1, -1), 0);
  mat_a.InsertElement(1, Complex(3, 3), 1);

  mat_b.InsertElement(0, Complex(6, 6), 1);
  mat_b.InsertElement(1, Complex(-7, -7), 0);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }
  expected_result.InsertElement(0, Complex(0, -12), 1);
  expected_result.InsertElement(1, Complex(0, -42), 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_zero_elements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(2, true, 2);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(0, 0), 0);
  mat_a.InsertElement(1, Complex(0, 0), 1);

  mat_b.InsertElement(0, Complex(0, 0), 1);
  mat_b.InsertElement(1, Complex(0, 0), 0);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }
  expected_result.InsertElement(0, Complex(0, 0), 1);
  expected_result.InsertElement(1, Complex(0, 0), 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_different_dimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(5, false, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  mat_a.InsertElement(0, Complex(1, 0), 0);
  mat_a.InsertElement(0, Complex(2, 0), 2);
  mat_a.InsertElement(1, Complex(3, 0), 1);
  mat_a.InsertElement(2, Complex(4, 0), 0);
  mat_a.InsertElement(2, Complex(5, 0), 1);

  mat_b.InsertElement(0, Complex(6, 0), 1);
  mat_b.InsertElement(1, Complex(7, 0), 0);
  mat_b.InsertElement(2, Complex(8, 0), 2);
  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_zero_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(3, true, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  // mat_a is zero matrix
  mat_b.InsertElement(0, Complex(1, 0), 0);
  mat_b.InsertElement(1, Complex(2, 0), 1);
  mat_b.InsertElement(2, Complex(3, 0), 2);

  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Expected result is zero matrix
  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_identity_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(3, true, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  // mat_a is identity matrix
  mat_a.InsertElement(0, Complex(1, 0), 0);
  mat_a.InsertElement(1, Complex(1, 0), 1);
  mat_a.InsertElement(2, Complex(1, 0), 2);

  mat_b.InsertElement(0, Complex(2, 0), 0);
  mat_b.InsertElement(1, Complex(3, 0), 1);
  mat_b.InsertElement(2, Complex(4, 0), 2);

  // Expected result is mat_b
  expected_result.InsertElement(0, Complex(2, 0), 0);
  expected_result.InsertElement(1, Complex(3, 0), 1);
  expected_result.InsertElement(2, Complex(4, 0), 2);

  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrix_with_single_non_zero_element) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(3, true, 3);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  // mat_a has only one non-zero element at (1, 1)
  mat_a.InsertElement(1, Complex(5, 0), 1);

  // mat_b is a regular matrix
  mat_b.InsertElement(0, Complex(1, 0), 0);
  mat_b.InsertElement(1, Complex(2, 0), 1);
  mat_b.InsertElement(2, Complex(3, 0), 2);

  // Expected result: only one non-zero row (row 1)
  expected_result.InsertElement(1, Complex(10, 0), 1);

  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_large_matrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(100, true, 100);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(100, true, 100);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(100, true, 100);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  // Fill matrices with some values
  for (int i = 0; i < 100; ++i) {
    mat_a.InsertElement(i, Complex(i + 1, 0), i);
    mat_b.InsertElement(i, Complex(i + 1, 0), i);
    expected_result.InsertElement(i, Complex((i + 1) * (i + 1), 0), i);
  }

  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_real_matrices_complex_result) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_a(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS mat_b(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(2, true, 2);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vec_a;
  std::vector<Complex> vec_b;
  std::vector<Complex> output_data(mat_a.columnCount * mat_b.rowCount * 100, 0);

  // Matrix A (Real part only)
  mat_a.InsertElement(0, Complex(1, 0), 0);
  mat_a.InsertElement(0, Complex(2, 0), 1);
  mat_a.InsertElement(1, Complex(3, 0), 0);
  mat_a.InsertElement(1, Complex(4, 0), 1);

  // Matrix B (Real part only)
  mat_b.InsertElement(0, Complex(5, 0), 0);
  mat_b.InsertElement(0, Complex(6, 0), 1);
  mat_b.InsertElement(1, Complex(7, 0), 0);
  mat_b.InsertElement(1, Complex(8, 0), 1);

  vec_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_a);
  vec_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(mat_b);
  input_data.reserve(vec_a.size() + vec_b.size());
  for (unsigned int i = 0; i < vec_a.size(); i++) {
    input_data.emplace_back(vec_a[i]);
  }
  for (unsigned int i = 0; i < vec_b.size(); i++) {
    input_data.emplace_back(vec_b[i]);
  }

  // Expected Result (Real part only)
  expected_result.InsertElement(0, Complex(19, 0), 0);  // 1*5 + 2*7 = 19
  expected_result.InsertElement(0, Complex(22, 0), 1);  // 1*6 + 2*8 = 22
  expected_result.InsertElement(1, Complex(43, 0), 0);  // 3*5 + 4*7 = 43
  expected_result.InsertElement(1, Complex(50, 0), 1);  // 3*6 + 4*8 = 50

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expected_result));
}