#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/gnitienko_k_strassen_alg/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
double min_val = -100.0;
double max_val = 100.0;
static std::vector<double> GenMatrix(size_t size);
static void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                            size_t size, const boost::mpi::communicator &world);

std::vector<double> GenMatrix(size_t size) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}

std::vector<double> IdentityMatrix(size_t size) {
  std::vector<double> matrix(size * size, 0);
  for (size_t i = 0; i < size; ++i) {
    matrix[(i * size) + i] = 1;
  }
  return matrix;
}

void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, size_t size,
                     const boost::mpi::communicator &world) {
  if (world.rank() == 0) {
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        c[(i * size) + j] = 0;
        for (size_t k = 0; k < size; ++k) {
          c[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
        }
      }
    }
  }
}

void TestMultiply(std::vector<double> &a, std::vector<double> &b, std::vector<double> &out,
                  std::vector<double> &expected, const boost::mpi::communicator &world, size_t size = 0) {
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_all->inputs_count.emplace_back(a.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  gnitienko_k_strassen_algorithm_all::StrassenAlgAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(expected[i], out[i], 1e-2);
    }
  }
}
}  // namespace

TEST(gnitienko_k_strassen_alg_all, test_2x2_matrix) {
  // Create data
  size_t size = 2;
  boost::mpi::communicator world;
  std::vector<double> a = {2.4, 3.5, -4.1, 13.3};
  std::vector<double> b = {1.4, -0.5, 1.1, 2.3};
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(4);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_4x4_matrix) {
  // Create data
  size_t size = 4;
  boost::mpi::communicator world;
  std::vector<double> a = {2.4, 3.5, -4.1, 13.3, 1.4, -0.5, 1.1, 2.3, 3.2, 2.1, -1.3, 4.5, 0.9, -2.7, 3.8, -1.2};
  std::vector<double> b = {1.1, -0.8, 2.3, 0.5, -1.5, 3.2, 0.7, 1.9, 0.9, -1.1, 1.5, -0.4, 2.2, 0.6, -3.1, 1.3};
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(16);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_random_16x16) {
  // Create data
  size_t size = 16;
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_random_64x64) {
  // Create data
  size_t size = 64;
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_non_squad_7x7) {
  // Create data
  size_t size = 7;
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_empty) {
  // Create data
  boost::mpi::communicator world;
  std::vector<double> a = {};
  std::vector<double> b = {};
  std::vector<double> expected = {};
  std::vector<double> out = {};

  TestMultiply(a, b, out, expected, world, 0);
}

TEST(gnitienko_k_strassen_alg_all, test_IxA) {
  // Create data
  size_t size = 3;
  boost::mpi::communicator world;
  std::vector<double> a = {2.4, 3.5, -4.1, 13.3, 5.4, 3.2, 4.5, 0.7, 1.9};
  std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> expected = {2.4, 3.5, -4.1, 13.3, 5.4, 3.2, 4.5, 0.7, 1.9};
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_IxA_large) {
  // Create data
  size_t size = 64;
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = IdentityMatrix(size);
  std::vector<double> expected(a);
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}

TEST(gnitienko_k_strassen_alg_all, test_random_256x256) {
  // Create data
  size_t size = 256;
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  std::vector<double> expected(size * size);
  TrivialMultiply(a, b, expected, size, world);
  std::vector<double> out(size * size);

  TestMultiply(a, b, out, expected, world, size);
}
