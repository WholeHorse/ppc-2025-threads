#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/karaseva_e_congrad/include/ops_omp.hpp"

namespace {

// Function to generate a random symmetric positive-definite matrix of size matrix_size x matrix_size
// The matrix is computed as A = R^T * R
std::vector<double> GenerateRandomSPDMatrix(size_t matrix_size, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> r_matrix(matrix_size * matrix_size);
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    r_matrix[i] = dist(gen);
  }
  std::vector<double> a_matrix(matrix_size * matrix_size, 0.0);
  // Compute a_matrix = R^T * R
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        a_matrix[(i * matrix_size) + j] += (r_matrix[(k * matrix_size) + i] * r_matrix[(k * matrix_size) + j]);
      }
    }
  }
  // Add diagonal dominance
  for (size_t i = 0; i < matrix_size; ++i) {
    a_matrix[(i * matrix_size) + i] += static_cast<double>(matrix_size);
  }
  return a_matrix;
}

// Helper function to multiply a_matrix (size matrix_size x matrix_size) by vector x (length matrix_size)
std::vector<double> MultiplyMatrixVector(const std::vector<double>& a_matrix, const std::vector<double>& x,
                                         size_t matrix_size) {
  std::vector<double> result(matrix_size, 0.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      result[i] += (a_matrix[(i * matrix_size) + j] * x[j]);
    }
  }
  return result;
}

}  // namespace

TEST(karaseva_e_congrad_omp, test_identity_50) {
  constexpr size_t kN = 50;

  // Create identity matrix
  std::vector<double> a_matrix(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    a_matrix[(i * kN) + i] = 1.0;
  }

  // Create vector b
  std::vector<double> b(kN);
  for (size_t i = 0; i < kN; ++i) {
    b[i] = static_cast<double>(i + 1);
  }

  // Solution vector
  std::vector<double> x(kN, 0.0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_omp->inputs_count.push_back(kN * kN);
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.push_back(kN);
  task_data_omp->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_omp->outputs_count.push_back(kN);

  // Create Task
  karaseva_e_congrad_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  // Accuracy check
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], b[i], 1e-9);
  }
}

TEST(karaseva_e_congrad_omp, test_small_random_spd) {
  constexpr size_t kN = 10;
  constexpr double kTolerance = 1e-6;

  // Generate random SPD matrix
  auto a_matrix = GenerateRandomSPDMatrix(kN);

  // Generate random solution vector
  std::vector<double> x_expected(kN);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (auto& val : x_expected) {
    val = dist(gen);
  }

  // Compute b = A * x_expected
  auto b = MultiplyMatrixVector(a_matrix, x_expected, kN);

  // Task configuration
  std::vector<double> x(kN, 0.0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kN * kN, kN};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kN};

  // Create Task
  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  // Verification
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_omp, test_large_spd) {
  constexpr size_t kN = 100;
  constexpr double kTolerance = 1e-4;

  auto a_matrix = GenerateRandomSPDMatrix(kN);
  std::vector<double> x_expected(kN, 1.0);
  auto b = MultiplyMatrixVector(a_matrix, x_expected, kN);

  std::vector<double> x(kN, 0.0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kN * kN, kN};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kN};

  // Create Task
  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_omp, test_diagonal_dominant) {
  constexpr size_t kN = 20;
  constexpr double kTolerance = 1e-8;

  // Create diagonally dominant matrix
  std::vector<double> a_matrix(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    a_matrix[(i * kN) + i] = 100.0 + static_cast<double>(i);
    for (size_t j = i + 1; j < kN; ++j) {
      a_matrix[(i * kN) + j] = a_matrix[(j * kN) + i] = 0.1;
    }
  }

  // Fill x_expected with values 1.0, 2.0, ..., kN
  std::vector<double> x_expected(kN);
  for (size_t i = 0; i < kN; ++i) {
    x_expected[i] = static_cast<double>(i + 1);
  }

  auto b = MultiplyMatrixVector(a_matrix, x_expected, kN);

  std::vector<double> x(kN, 0.0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kN * kN, kN};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kN};

  // Create Task
  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_omp, test_zero_solution) {
  constexpr size_t kN = 30;

  auto a_matrix = GenerateRandomSPDMatrix(kN);
  std::vector<double> b(kN, 0.0);
  std::vector<double> x(kN, 1.0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kN * kN, kN};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kN};

  // Create Task
  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  // Expect zero vector
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], 0.0, 1e-10);
  }
}

TEST(karaseva_e_congrad_omp, validation_check_invalid_matrix) {
  // Test non-square matrix input
  constexpr size_t kInvalidSize = 3;
  std::vector<double> a_matrix(kInvalidSize * 2, 1.0);  // 3x2 matrix
  std::vector<double> b(kInvalidSize, 0.0);
  std::vector<double> x(kInvalidSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kInvalidSize * 2, kInvalidSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kInvalidSize};

  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(karaseva_e_congrad_omp, validation_check_invalid_output) {
  // Test output size mismatch
  constexpr size_t kValidSize = 4;
  std::vector<double> a_matrix(kValidSize * kValidSize, 1.0);
  std::vector<double> b(kValidSize, 0.0);
  std::vector<double> x(kValidSize - 1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kValidSize * kValidSize, kValidSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kValidSize - 1};

  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(karaseva_e_congrad_omp, validation_check_valid_data) {
  // Test valid input configuration
  constexpr size_t kValidSize = 5;
  std::vector<double> a_matrix(kValidSize * kValidSize, 1.0);
  std::vector<double> b(kValidSize, 0.0);
  std::vector<double> x(kValidSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kValidSize * kValidSize, kValidSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kValidSize};

  karaseva_e_congrad_omp::TestTaskOpenMP task(task_data);
  EXPECT_TRUE(task.Validation());
}