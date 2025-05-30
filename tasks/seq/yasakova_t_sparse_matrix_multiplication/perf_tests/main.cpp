#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_pipeline_run) {
  const int matrix_size = 1000;
  const int non_zero_elements = 10000;
  const int num_runs = 100;
  std::srand(std::time(nullptr));

  // Create random sparse matrices
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_a(matrix_size, true, non_zero_elements);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_b(matrix_size, true, non_zero_elements);

  // Fill matrix A with random elements
  for (int i = 0; i < non_zero_elements; i++) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    sparse_matrix_a.InsertElement(row, Complex(-50 + (rand() % 100), -50 + (rand() % 100)), col);
  }

  // Fill matrix B with random elements
  for (int i = 0; i < non_zero_elements; i++) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    sparse_matrix_b.InsertElement(row, Complex(-50 + (rand() % 100), -50 + (rand() % 100)), col);
  }

  // Convert matrices to vectors
  std::vector<Complex> vector_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_a);
  std::vector<Complex> vector_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_b);

  // Prepare input data
  std::vector<Complex> input_data;
  input_data.reserve(vector_a.size() + vector_b.size());
  input_data.insert(input_data.end(), vector_a.begin(), vector_a.end());
  input_data.insert(input_data.end(), vector_b.begin(), vector_b.end());

  // Prepare result vector
  std::vector<Complex> result_vector(matrix_size * matrix_size, Complex(0, 0));

  // Create task data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task_data->outputs_count.emplace_back(result_vector.size());

  // Create Task
  auto sequential_task =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(task_data);

  // Create Performance attributes
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = num_runs;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(sequential_task);
  performance_analyzer->PipelineRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  // Convert result and verify
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS final_result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(result_vector);

  ASSERT_EQ(final_result.rowCount, sparse_matrix_a.rowCount);
  ASSERT_EQ(final_result.columnCount, sparse_matrix_b.columnCount);

  bool is_result_non_zero = false;
  for (const auto& elem : result_vector) {
    if (elem != Complex(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_task_run) {
  const int matrix_size = 1000;
  const int non_zero_elements = 10000;
  const int num_runs = 100;

  // Initialize random number generator
  std::srand(std::time(nullptr));

  // Create random sparse matrices
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_a(matrix_size, true, non_zero_elements);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_b(matrix_size, true, non_zero_elements);

  // Fill matrix A with random elements
  for (int i = 0; i < non_zero_elements; i++) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    sparse_matrix_a.InsertElement(row, Complex(-50 + (rand() % 100), -50 + (rand() % 100)), col);
  }

  // Fill matrix B with random elements
  for (int i = 0; i < non_zero_elements; i++) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    sparse_matrix_b.InsertElement(row, Complex(-50 + (rand() % 100), -50 + (rand() % 100)), col);
  }

  // Convert matrices to vectors
  std::vector<Complex> vector_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_a);
  std::vector<Complex> vector_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_b);

  // Prepare input data
  std::vector<Complex> input_data;
  input_data.reserve(vector_a.size() + vector_b.size());
  input_data.insert(input_data.end(), vector_a.begin(), vector_a.end());
  input_data.insert(input_data.end(), vector_b.begin(), vector_b.end());

  // Prepare result vector
  std::vector<Complex> result_vector(matrix_size * matrix_size, Complex(0, 0));

  // Create task data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task_data->outputs_count.emplace_back(result_vector.size());

  // Create Task
  auto sequential_task =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(task_data);

  // Create Performance attributes
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = num_runs;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(sequential_task);
  performance_analyzer->TaskRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  // Convert result and verify
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS final_result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(result_vector);

  ASSERT_EQ(final_result.rowCount, sparse_matrix_a.rowCount);
  ASSERT_EQ(final_result.columnCount, sparse_matrix_b.columnCount);

  bool is_result_non_zero = false;
  for (const auto& elem : result_vector) {
    if (elem != Complex(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}