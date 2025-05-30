#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/belov_a_radix_sort_with_batcher_mergesort/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

using namespace belov_a_radix_batcher_mergesort_all;

namespace {
vector<Bigint> GenerateMixedValuesArray(int n) {
  random_device rd;
  mt19937 gen(rd());

  uniform_int_distribution<Bigint> small_range(-999LL, 999LL);
  uniform_int_distribution<Bigint> medium_range(-10000LL, 10000LL);
  uniform_int_distribution<Bigint> large_range(-10000000000LL, 10000000000LL);
  uniform_int_distribution<int> choice(0, 2);

  vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (choice(gen) == 0) {
      arr.push_back(small_range(gen));
    } else if (choice(gen) == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(large_range(gen));
    }
  }
  return arr;
}
}  // namespace

TEST(belov_a_radix_batcher_mergesort_all, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Create data
  int n = 4194304;
  vector<Bigint> arr = GenerateMixedValuesArray(n);

  // Create TaskData
  shared_ptr<ppc::core::TaskData> task_data_mpi_omp = make_shared<ppc::core::TaskData>();
  task_data_mpi_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_mpi_omp->inputs_count.emplace_back(arr.size());
  task_data_mpi_omp->inputs_count.emplace_back(n);
  task_data_mpi_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_mpi_omp->outputs_count.emplace_back(arr.size());

  // Create Task
  auto test_task_mpi_omp =
      std::make_shared<belov_a_radix_batcher_mergesort_all::RadixBatcherMergesortParallelAll>(task_data_mpi_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ASSERT_TRUE(test_task_mpi_omp->Validation());
  test_task_mpi_omp->PreProcessing();
  test_task_mpi_omp->Run();
  test_task_mpi_omp->PostProcessing();

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_TRUE(std::ranges::is_sorted(arr));
  }
}

TEST(belov_a_radix_batcher_mergesort_all, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Create data
  int n = 4194304;
  vector<Bigint> arr = GenerateMixedValuesArray(n);

  // Create TaskData
  shared_ptr<ppc::core::TaskData> task_data_mpi_omp = make_shared<ppc::core::TaskData>();
  task_data_mpi_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_mpi_omp->inputs_count.emplace_back(arr.size());
  task_data_mpi_omp->inputs_count.emplace_back(n);
  task_data_mpi_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data_mpi_omp->outputs_count.emplace_back(arr.size());

  // Create Task
  auto test_task_mpi_omp =
      std::make_shared<belov_a_radix_batcher_mergesort_all::RadixBatcherMergesortParallelAll>(task_data_mpi_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ASSERT_TRUE(test_task_mpi_omp->Validation());
  test_task_mpi_omp->PreProcessing();
  test_task_mpi_omp->Run();
  test_task_mpi_omp->PostProcessing();

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_TRUE(std::ranges::is_sorted(arr));
  }
}