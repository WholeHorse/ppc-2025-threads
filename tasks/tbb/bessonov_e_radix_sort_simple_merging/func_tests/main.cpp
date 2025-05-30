#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/bessonov_e_radix_sort_simple_merging/include/ops_tbb.hpp"

namespace {
std::vector<double> GenerateVector(std::size_t n, double min, double max) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(min, max);
  std::vector<double> data(n);
  for (double& d : data) {
    d = dist(gen);
  }
  return data;
}
}  // namespace

TEST(bessonov_e_radix_sort_simple_merging_tbb, BasicSortingTest) {
  std::vector<double> input = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, SingleElementTest) {
  std::vector<double> input = {42.0};
  std::vector<double> output(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, input);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, NegativeAndPositiveTest) {
  std::vector<double> input = {-3.2, 1.1, -7.5, 0.0, 4.4, -2.2, 3.3};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, RandomVectorTest) {
  std::vector<double> input = GenerateVector(100, -1000.0, 1000.0);
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, AllSameElementsTest) {
  std::vector<double> input = {3.14, 3.14, 3.14, 3.14};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, ExtremeValuesTest) {
  std::vector<double> input = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0, -42.5,
                               100.0};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, TinyNumbersTest) {
  std::vector<double> input = {1e-10, -1e-10, 1e-20, -1e-20};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, InvalidInputOutputSizeTest) {
  std::vector<double> input = {1.0, 2.0, 3.0};
  std::vector<double> output(2, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, ValidationEmptyTest) {
  std::vector<double> input;
  std::vector<double> output;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_tbb, ReverseOrderTest) {
  std::vector<double> input = {9.1, 8.9, 7.8, 6.7, 5.6, 4.5, 4.3, 3.4, 3.0, 2.3, 1.5, 1.2, 1.0, 0.5, 0.2};
  std::vector<double> output(input.size(), 0.0);
  std::vector<double> expected = input;
  std::ranges::sort(expected);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_tbb::TestTaskTbb task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(output, expected);
}