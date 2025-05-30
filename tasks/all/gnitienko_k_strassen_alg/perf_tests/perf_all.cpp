#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/gnitienko_k_strassen_alg/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
double min_val = -100.0;
double max_val = 100.0;
static std::vector<double> GenMatrix(size_t size);
static void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                            size_t size);

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

void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      c[(i * size) + j] = 0;
      for (size_t k = 0; k < size; ++k) {
        c[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
}
}  // namespace

TEST(gnitienko_k_strassen_alg_all, test_pipeline_run) {
  size_t size = 1024;

  // Create data
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  boost::mpi::communicator world;
  std::vector<double> expected(size * size);

  std::vector<double> out(size * size);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TrivialMultiply(a, b, expected, size);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_all->inputs_count.emplace_back(a.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<gnitienko_k_strassen_algorithm_all::StrassenAlgAll>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(expected[i], out[i], 1e-2);
    }
  }
}

TEST(gnitienko_k_strassen_alg_all, test_task_run) {
  size_t size = 1024;

  // Create data
  boost::mpi::communicator world;
  std::vector<double> a = GenMatrix(size);
  std::vector<double> b = GenMatrix(size);
  std::vector<double> expected(size * size);

  std::vector<double> out(size * size);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    TrivialMultiply(a, b, expected, size);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_all->inputs_count.emplace_back(a.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<gnitienko_k_strassen_algorithm_all::StrassenAlgAll>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(expected[i], out[i], 1e-2);
    }
  }
}