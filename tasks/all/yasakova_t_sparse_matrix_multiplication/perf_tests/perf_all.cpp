#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(yasakova_t_sparse_matrix_mult_task_all, test_pipeline_run) {
  srand(time(nullptr));
  const int matrix_size = 500;
  const int non_zero_elements = 1000;
  const int num_runs = 100;

  // Инициализация матриц
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(matrix_size, matrix_size);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(matrix_size, matrix_size);

  // Заполняем матрицы случайными элементами
  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    ComplexNum value(-50 + (rand() % 100), -50 + (rand() % 100));
    a.InsertElement(row, value, col);
  }
  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    ComplexNum value(-50 + (rand() % 100), -50 + (rand() % 100));
    b.InsertElement(row, value, col);
  }

  // Конвертируем матрицы в векторы для передачи в задачу
  std::vector<ComplexNum> in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  std::vector<ComplexNum> in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);

  // Подготавливаем входные данные (конкатенация двух векторов)
  std::vector<ComplexNum> input_data;
  input_data.reserve(in_a.size() + in_b.size());
  input_data.insert(input_data.end(), in_a.begin(), in_a.end());
  input_data.insert(input_data.end(), in_b.begin(), in_b.end());

  // Подготавливаем вектор для результата
  std::vector<ComplexNum> output_data(matrix_size * matrix_size, ComplexNum(0, 0));

  // Создаем данные задачи
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_all->inputs_count.emplace_back(input_data.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_all->outputs_count.emplace_back(output_data.size());

  // Создаем задачу
  auto test_task_all = std::make_shared<yasakova_t_sparse_matrix_mult_all::TestTaskALL>(task_data_all);

  // Создаем атрибуты для измерения производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_runs;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Запуск теста производительности
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Конвертируем результат обратно в матрицу
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS actual_result =
      yasakova_t_sparse_matrix_mult_all::ConvertToSparse(output_data);

  // Проверяем размеры результата
  ASSERT_EQ(actual_result.total_rows, matrix_size);
  ASSERT_EQ(actual_result.total_cols, matrix_size);

  // Проверяем, что результат не нулевой
  bool is_result_non_zero = false;
  for (const auto &elem : output_data) {
    if (elem != ComplexNum(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}

TEST(yasakova_t_sparse_matrix_mult_task_all, test_task_run) {
  srand(time(nullptr));
  const int matrix_size = 500;
  const int non_zero_elements = 1000;
  const int num_runs = 100;

  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(matrix_size, matrix_size);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(matrix_size, matrix_size);

  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    ComplexNum value(-50 + (rand() % 100), -50 + (rand() % 100));
    a.InsertElement(row, value, col);
  }
  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    ComplexNum value(-50 + (rand() % 100), -50 + (rand() % 100));
    b.InsertElement(row, value, col);
  }

  std::vector<ComplexNum> in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  std::vector<ComplexNum> in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);

  std::vector<ComplexNum> input_data;
  input_data.reserve(in_a.size() + in_b.size());
  input_data.insert(input_data.end(), in_a.begin(), in_a.end());
  input_data.insert(input_data.end(), in_b.begin(), in_b.end());

  std::vector<ComplexNum> output_data(matrix_size * matrix_size, ComplexNum(0, 0));

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_all->inputs_count.emplace_back(input_data.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_all->outputs_count.emplace_back(output_data.size());

  auto test_task_all = std::make_shared<yasakova_t_sparse_matrix_mult_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_runs;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS actual_result =
      yasakova_t_sparse_matrix_mult_all::ConvertToSparse(output_data);

  ASSERT_EQ(actual_result.total_rows, matrix_size);
  ASSERT_EQ(actual_result.total_cols, matrix_size);

  bool is_result_non_zero = false;
  for (const auto &elem : output_data) {
    if (elem != ComplexNum(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}
