#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_random_array) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(1000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_double_reverse_array) {
  boost::mpi::communicator world;
  std::vector<double> input_array(1000);
  const auto half = input_array.size() / 2U;
  std::ranges::generate(input_array.begin(), input_array.end() - (long)half,
                        [value = half]() mutable { return value--; });
  std::ranges::generate(input_array.end() - (long)half, input_array.end(),
                        [value = half]() mutable { return value--; });
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_negative_elements_array) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, -1);
  std::vector<double> input_array(1000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_shuffle_array) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(1000);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 0]() mutable { return value++; });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_partially_sorted_array) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(1000);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 0]() mutable { return value++; });
  const auto half = input_array.size() / 2U;
  std::shuffle(input_array.begin() + (long)half, input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_backward_sorted_array) {
  boost::mpi::communicator world;
  std::vector<double> input_array(1000);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 800]() mutable { return value--; });
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(1000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  ASSERT_EQ(hoare_sort_task_mpi.Validation(), true);
  hoare_sort_task_mpi.PreProcessing();
  hoare_sort_task_mpi.Run();
  hoare_sort_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(true_solution, out_array[0]);
  }
}

TEST(deryabin_m_hoare_sort_simple_merge_mpi, test_invalid_array) {
  boost::mpi::communicator world;
  std::vector<double> input_array(2);
  std::vector<std::vector<double>> in_array(1, input_array);
  std::vector<double> output_array(2);
  std::vector<std::vector<double>> out_array(1, output_array);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
    task_data_mpi->inputs_count.emplace_back(input_array.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
    task_data_mpi->outputs_count.emplace_back(output_array.size());
  }

  deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI hoare_sort_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(hoare_sort_task_mpi.Validation(), false);
  }
}
