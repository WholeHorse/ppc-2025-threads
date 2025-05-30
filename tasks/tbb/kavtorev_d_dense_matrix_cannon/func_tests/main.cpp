// Copyright 2025 Kavtorev Dmitry
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kavtorev_d_dense_matrix_cannon/include/ops_tbb.hpp"

namespace {
std::vector<double> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-100.0, 100.0);

  std::vector<double> matrix(rows * cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[(i * cols) + j] = dis(gen);
    }
  }

  return matrix;
}
}  // namespace

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_3x3) {
  int n = 3;
  int m = 3;

  std::vector<double> in_mtrx_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> in_mtrx_b{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> result = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_2x2) {
  int n = 2;
  int m = 2;

  std::vector<double> in_mtrx_a{1, 2, 3, 4};
  std::vector<double> in_mtrx_b{6, 7, 8, 9};
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_5x5) {
  int n = 5;
  int m = 5;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);

  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_4x4) {
  int n = 4;
  int m = 4;

  std::vector<double> in_mtrx_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> in_mtrx_b{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  std::vector<double> expected_c = {80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386};

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], expected_c[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_6x6) {
  int n = 6;
  int m = 6;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  // Проверка корректности результата
  std::vector<double> expected_c(n * m, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < m; ++k) {
        expected_c[(i * m) + j] += in_mtrx_a[(i * m) + k] * in_mtrx_b[(k * m) + j];
      }
    }
  }

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_NEAR(res[i], expected_c[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_10x10) {
  int n = 10;
  int m = 10;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  // Проверка корректности результата
  std::vector<double> expected_c(n * m, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < m; ++k) {
        expected_c[(i * m) + j] += in_mtrx_a[(i * m) + k] * in_mtrx_b[(k * m) + j];
      }
    }
  }

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_NEAR(res[i], expected_c[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_AllOnes) {
  int n = 4;
  int m = 4;

  std::vector<double> in_mtrx_a(n * m, 1.0);
  std::vector<double> in_mtrx_b(n * m, 1.0);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  std::vector<double> expected_c(n * m, static_cast<double>(n));

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], expected_c[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_WithAllOnesMatrix) {
  int n = 3;
  int m = 3;

  std::vector<double> in_mtrx_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> in_mtrx_b(n * m, 1.0);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  std::vector<double> expected_c = {6, 6, 6, 15, 15, 15, 24, 24, 24};

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], expected_c[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_0x0) {
  int n = 0;
  int m = 0;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_100x100) {
  int n = 100;
  int m = 100;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n, m);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n, m);
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_1x1) {
  int n = 1;
  int m = 1;

  std::vector<double> in_mtrx_a{2};
  std::vector<double> in_mtrx_b{3};
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  ASSERT_EQ(res[0], 6);
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Validation_ZeroSizes) {
  int n = 0;
  int m = 0;

  std::vector<double> in_mtrx_a;
  std::vector<double> in_mtrx_b;
  std::vector<double> out;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Validation_MismatchedSizes) {
  int n = 3;
  int m = 3;

  std::vector<double> in_mtrx_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> in_mtrx_b{1, 2, 3, 4, 5, 6};  // Размеры не совпадают
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(kavtorev_d_dense_matrix_cannon_tbb, Multiplication_WithZeros) {
  int n = 3;
  int m = 3;

  std::vector<double> in_mtrx_a{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> in_mtrx_b{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> out(n * m);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_omp->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  std::vector<double> res = kavtorev_d_dense_matrix_cannon_tbb::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n, m);

  kavtorev_d_dense_matrix_cannon_tbb::TestTaskSequential test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], 0);
  }
}
