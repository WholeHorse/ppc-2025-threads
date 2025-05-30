#include <gtest/gtest.h>
#include <omp.h>  // Добавлен заголовочный файл OpenMP

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(yasakova_t_sparse_matrix_mult_all, test_matmul_only_real) {
  // Инициализация MPI
  boost::mpi::communicator world;

  // Создание данных
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(3, 3);
  std::vector<ComplexNum> in = {};
  std::vector<ComplexNum> in_a;
  std::vector<ComplexNum> in_b;
  std::vector<ComplexNum> out(a.total_cols * b.total_rows * 100, 0);

  // Заполнение матриц
  a.InsertElement(0, ComplexNum(1, 0), 0);
  a.InsertElement(0, ComplexNum(2, 0), 2);
  a.InsertElement(1, ComplexNum(3, 0), 1);
  a.InsertElement(2, ComplexNum(4, 0), 0);
  a.InsertElement(2, ComplexNum(5, 0), 1);

  b.InsertElement(0, ComplexNum(6, 0), 1);
  b.InsertElement(1, ComplexNum(7, 0), 0);
  b.InsertElement(2, ComplexNum(8, 0), 2);

  // Подготовка входных данных
  in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);
  in.reserve(in_a.size() + in_b.size());
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  // Ожидаемый результат
  c.InsertElement(0, ComplexNum(6, 0), 1);
  c.InsertElement(0, ComplexNum(16, 0), 2);
  c.InsertElement(1, ComplexNum(21, 0), 0);
  c.InsertElement(2, ComplexNum(24, 0), 1);
  c.InsertElement(2, ComplexNum(35, 0), 0);

  // Создание задачи
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Выполнение теста
  yasakova_t_sparse_matrix_mult_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();

  // Установка числа потоков OpenMP
  omp_set_num_threads(omp_get_max_threads());

  test_task_all.Run();
  test_task_all.PostProcessing();

  // Проверка результатов только на нулевом процессе
  if (world.rank() == 0) {
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res = yasakova_t_sparse_matrix_mult_all::ConvertToSparse(out);
    ASSERT_TRUE(yasakova_t_sparse_matrix_mult_all::CompareMatrices(res, c));
  }
}

TEST(yasakova_t_sparse_matrix_mult_all, test_empty_matrices) {
  boost::mpi::communicator world;

  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(3, 3);

  std::vector<ComplexNum> in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  std::vector<ComplexNum> in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);
  std::vector<ComplexNum> in;
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  std::vector<ComplexNum> out(a.total_cols * b.total_rows * 100, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  yasakova_t_sparse_matrix_mult_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();

  omp_set_num_threads(omp_get_max_threads());
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res = yasakova_t_sparse_matrix_mult_all::ConvertToSparse(out);
    ASSERT_TRUE(yasakova_t_sparse_matrix_mult_all::CompareMatrices(res, c));
  }
}

TEST(yasakova_t_sparse_matrix_mult_all, test_identity_matrix) {
  boost::mpi::communicator world;

  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(3, 3);

  // Заполняем матрицу a
  a.InsertElement(0, ComplexNum(1, 0), 0);
  a.InsertElement(0, ComplexNum(2, 0), 1);
  a.InsertElement(1, ComplexNum(3, 0), 1);
  a.InsertElement(2, ComplexNum(4, 0), 2);

  // Создаем единичную матрицу b
  b.InsertElement(0, ComplexNum(1, 0), 0);
  b.InsertElement(1, ComplexNum(1, 0), 1);
  b.InsertElement(2, ComplexNum(1, 0), 2);

  std::vector<ComplexNum> in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  std::vector<ComplexNum> in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);
  std::vector<ComplexNum> in;
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  std::vector<ComplexNum> out(a.total_cols * b.total_rows * 100, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  yasakova_t_sparse_matrix_mult_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();

  omp_set_num_threads(omp_get_max_threads());
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res = yasakova_t_sparse_matrix_mult_all::ConvertToSparse(out);
    ASSERT_TRUE(yasakova_t_sparse_matrix_mult_all::CompareMatrices(res, a));
  }
}

TEST(yasakova_t_sparse_matrix_mult_all, test_large_matrices) {
  boost::mpi::communicator world;

  const int size = 100;
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(size, size);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(size, size);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(size, size);

  // Заполняем диагональные матрицы
  for (int i = 0; i < size; ++i) {
    a.InsertElement(i, ComplexNum(i + 1, 0), i);
    b.InsertElement(i, ComplexNum(size - i, 0), i);
    c.InsertElement(i, ComplexNum((i + 1) * (size - i), 0), i);
  }

  std::vector<ComplexNum> in_a = yasakova_t_sparse_matrix_mult_all::ConvertToDense(a);
  std::vector<ComplexNum> in_b = yasakova_t_sparse_matrix_mult_all::ConvertToDense(b);
  std::vector<ComplexNum> in;
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  std::vector<ComplexNum> out(a.total_cols * b.total_rows * 100, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  yasakova_t_sparse_matrix_mult_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();

  omp_set_num_threads(omp_get_max_threads());
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res = yasakova_t_sparse_matrix_mult_all::ConvertToSparse(out);
    ASSERT_TRUE(yasakova_t_sparse_matrix_mult_all::CompareMatrices(res, c));
  }
}