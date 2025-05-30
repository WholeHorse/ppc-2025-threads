#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "all/karaseva_e_congrad/include/ops_mpi.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> GenerateRandomSPDMatrix(size_t matrix_size, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> r_matrix(matrix_size * matrix_size);
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    r_matrix[i] = dist(gen);
  }

  std::vector<double> a_matrix(matrix_size * matrix_size, 0.0);
  const auto matrix_size_d = static_cast<double>(matrix_size);

  // Generate symmetric positive-definite matrix.
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        a_matrix[(i * matrix_size) + j] += r_matrix[(k * matrix_size) + i] * r_matrix[(k * matrix_size) + j];
      }
    }
    a_matrix[(i * matrix_size) + i] += matrix_size_d;  // Ensure diagonal dominance
  }
  return a_matrix;
}

std::vector<double> MultiplyMatrixVector(const std::vector<double>& a_matrix, const std::vector<double>& x,
                                         size_t matrix_size) {
  std::vector<double> result(matrix_size, 0.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      result[i] += a_matrix[(i * matrix_size) + j] * x[j];
    }
  }
  return result;
}

}  // namespace

TEST(karaseva_e_congrad_mpi, test_small_matrix_2x2) {
  constexpr size_t kSize = 2;
  constexpr double kTolerance = 1e-10;

  std::vector<double> a_matrix = {4.0, 1.0, 1.0, 3.0};
  std::vector<double> x_expected = {1.0, -2.0};
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);
  std::vector<double> solution(kSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data->inputs_count.emplace_back(a_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  task_data->inputs_count.emplace_back(b_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(solution.data()));
  task_data->outputs_count.emplace_back(solution.size());

  karaseva_e_congrad_mpi::TestTaskMPI test_task_mpi(task_data);
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_mpi, test_small_random_spd) {
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
  karaseva_e_congrad_mpi::TestTaskMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  // Verification
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_mpi, validation_fails_on_non_square_matrix) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  // Intentionally wrong matrix size (3 elements for 2x2 matrix)
  std::vector<double> a_matrix = {1.0, 2.0, 3.0};
  std::vector<double> b_vector = {1.0, 2.0};
  std::vector<double> solution(2, 0.0);

  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b_vector.data())};
  task_data->inputs_count = {3, 2};
  task_data->outputs = {reinterpret_cast<uint8_t*>(solution.data())};
  task_data->outputs_count = {2};

  karaseva_e_congrad_mpi::TestTaskMPI test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(karaseva_e_congrad_mpi, validation_fails_on_vector_size_mismatch) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> a_matrix(4, 1.0);  // 2x2 matrix
  std::vector<double> b_vector = {1.0};  // Wrong size (1 instead of 2)
  std::vector<double> solution(2, 0.0);

  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b_vector.data())};
  task_data->inputs_count = {4, 1};
  task_data->outputs = {reinterpret_cast<uint8_t*>(solution.data())};
  task_data->outputs_count = {2};

  karaseva_e_congrad_mpi::TestTaskMPI test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(karaseva_e_congrad_mpi, identity_matrix_solution) {
  constexpr size_t kSize = 3;
  constexpr double kTolerance = 1e-10;

  // Identity matrix
  std::vector<double> a_matrix(kSize * kSize, 0.0);
  for (size_t i = 0; i < kSize; ++i) {
    a_matrix[(i * kSize) + i] = 1.0;
  }

  std::vector<double> x_expected = {5.0, -3.0, 2.0};
  auto b_vector = x_expected;  // For identity matrix b = x
  std::vector<double> solution(kSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b_vector.data())};
  task_data->inputs_count = {kSize * kSize, kSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(solution.data())};
  task_data->outputs_count = {kSize};

  karaseva_e_congrad_mpi::TestTaskMPI test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_mpi, large_spd_matrix_100x100) {
  constexpr size_t kN = 100;
  constexpr double kTolerance = 1e-6;

  auto a_matrix = GenerateRandomSPDMatrix(kN);
  std::vector<double> x_expected(kN);
  std::iota(x_expected.begin(), x_expected.end(), 1.0);

  auto b = MultiplyMatrixVector(a_matrix, x_expected, kN);
  std::vector<double> x(kN, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b.data())};
  task_data->inputs_count = {kN * kN, kN};
  task_data->outputs = {reinterpret_cast<uint8_t*>(x.data())};
  task_data->outputs_count = {kN};

  karaseva_e_congrad_mpi::TestTaskMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_mpi, already_solved_system) {
  constexpr size_t kSize = 4;
  constexpr double kTolerance = 1e-10;

  std::vector<double> a_matrix = GenerateRandomSPDMatrix(kSize);
  std::vector<double> x_expected(kSize, 0.0);  // Zero vector
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);

  // Initial guess is already solution
  std::vector<double> solution = x_expected;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b_vector.data())};
  task_data->inputs_count = {kSize * kSize, kSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(solution.data())};
  task_data->outputs_count = {kSize};

  karaseva_e_congrad_mpi::TestTaskMPI test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance);
  }
}

TEST(karaseva_e_congrad_mpi, single_iteration_convergence) {
  constexpr size_t kSize = 3;
  constexpr double kTolerance = 1e-10;

  // Diagonal matrix with large values for instant convergence
  std::vector<double> a_matrix(kSize * kSize, 0.0);
  for (size_t i = 0; i < kSize; ++i) {
    a_matrix[(i * kSize) + i] = 1e12;
  }

  std::vector<double> x_expected = {2.0, 3.0, 4.0};
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);
  std::vector<double> solution(kSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_matrix.data()), reinterpret_cast<uint8_t*>(b_vector.data())};
  task_data->inputs_count = {kSize * kSize, kSize};
  task_data->outputs = {reinterpret_cast<uint8_t*>(solution.data())};
  task_data->outputs_count = {kSize};

  karaseva_e_congrad_mpi::TestTaskMPI test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance);
  }
}