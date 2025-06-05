#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_pipeline_run_MPI) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(1024000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 512;
  std::vector<double> output_array_mpi(1024000);
  std::vector<std::vector<double>> out_array_mpi(1, output_array_mpi);
  std::vector<double> output_array_seq(1024000);
  std::vector<std::vector<double>> out_array_seq(1, output_array_seq);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_mpi->inputs_count.emplace_back(input_array.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array_mpi.data()));
  task_data_mpi->outputs_count.emplace_back(output_array_mpi.size());
  auto hoare_sort_simple_merge_task_mpi =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI>(task_data_mpi);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer_mpi = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_mpi);
  perf_analyzer_mpi->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_solution, out_array_mpi[0]);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_seq->inputs_count.emplace_back(input_array.size());
    task_data_seq->inputs_count.emplace_back(chunk_count);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array_seq.data()));
    task_data_seq->outputs_count.emplace_back(output_array_seq.size());
    auto hoare_sort_simple_merge_task_seq =
        std::make_shared<deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential>(task_data_seq);
    perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t1 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t1).count();
      return static_cast<double>(duration) * 1e-9;
    };
    perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer_seq = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_seq);
    perf_analyzer_seq->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_solution, out_array_seq[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_task_run_MPI) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(1024000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 512;
  std::vector<double> output_array_mpi(1024000);
  std::vector<std::vector<double>> out_array_mpi(1, output_array_mpi);
  std::vector<double> output_array_seq(1024000);
  std::vector<std::vector<double>> out_array_seq(1, output_array_seq);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_mpi->inputs_count.emplace_back(input_array.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array_mpi.data()));
  task_data_mpi->outputs_count.emplace_back(output_array_mpi.size());
  auto hoare_sort_simple_merge_task_mpi =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI>(task_data_mpi);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer_mpi = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_mpi);
  perf_analyzer_mpi->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_solution, out_array_mpi[0]);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_seq->inputs_count.emplace_back(input_array.size());
    task_data_seq->inputs_count.emplace_back(chunk_count);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array_seq.data()));
    task_data_seq->outputs_count.emplace_back(output_array_seq.size());
    auto hoare_sort_simple_merge_task_seq =
        std::make_shared<deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential>(task_data_seq);
    perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t1 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t1).count();
      return static_cast<double>(duration) * 1e-9;
    };
    perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer_seq = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_seq);
    perf_analyzer_seq->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true_solution, out_array_seq[0]);
  }
}
