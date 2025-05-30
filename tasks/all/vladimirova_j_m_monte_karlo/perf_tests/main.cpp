
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/vladimirova_j_m_monte_karlo/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> seed;
std::vector<int> GenRandomVector(size_t size, int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min, max);
  std::vector<int> random_vector(size);
  for (size_t i = 0; i < size; i++) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}
std::vector<double> GenRandomVectorArea(size_t size, int min, int r, double& ans) {
  ans = 1;
  std::random_device rd;
  std::mt19937 gen(rd());
  int max = min + r;
  std::uniform_int_distribution<int> dis(min, max);
  std::vector<double> random_vector(size * 2);
  for (size_t i = 0; i < size; i += 2) {
    random_vector[i] = dis(gen);
    random_vector[i + 1] = random_vector[i] + std::abs(dis(gen));
    ans *= (random_vector[i + 1] - random_vector[i]);
  }

  return random_vector;
}
bool RandomFunc(std::vector<double> arr, size_t size = 0) {
  double x = arr[0];
  double y = arr[1];
  double sum = 0;
  for (auto i : seed) {
    switch (i % 10) {
      case 0: {
        sum += x;
        break;
      }
      case 1: {
        sum += x * 5;
        break;
      }
      case 2: {
        sum += std::abs(x);
        break;
      }
      case 3: {
        sum += std::sqrt(x);
        break;
      }
      case 4: {
        sum += -3;
        break;
      }
      case 5: {
        sum += y;
        break;
      }
      case 6: {
        sum += y * 7;
        break;
      }
      case 7: {
        sum += std::abs(y);
        break;
      }
      case 8: {
        sum += std::sqrt(y);
        break;
      }
      case 9: {
        sum *= 2;
        break;
      }
      default: {
        break;
      }
    }
  }
  return (sum <= 0);
};
}  // namespace

TEST(vladimirova_j_m_monte_karlo_all, test_pipeline_run) {
  // Create data
  boost::mpi::communicator world;
  double ans = 1;
  std::vector<double> val_b = GenRandomVectorArea(2, -10, 30, ans);
  std::vector<double> out(1, 0);
  seed = GenRandomVector(5, 0, 10);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(RandomFunc));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(2500000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_all = std::make_shared<vladimirova_j_m_monte_karlo_all::TestTaskALL>(task_data_all);
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
    ASSERT_TRUE(out[0] <= ans);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_task_run) {
  boost::mpi::communicator world;
  // Create data
  double ans = 1;
  std::vector<double> val_b = GenRandomVectorArea(2, -10, 30, ans);
  std::vector<double> out(1, 0);
  seed = GenRandomVector(5, 0, 10);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(RandomFunc));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(2500000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_all = std::make_shared<vladimirova_j_m_monte_karlo_all::TestTaskALL>(task_data_all);

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
    ASSERT_TRUE(out[0] <= ans);
  };
}
