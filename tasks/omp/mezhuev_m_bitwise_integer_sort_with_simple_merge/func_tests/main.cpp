#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_omp.hpp"

// tests
TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_small_input) {
  constexpr size_t kCount = 5;
  std::vector<int> input = {5, 3, 8, 1, 4};
  std::vector<int> expected_output = {1, 3, 4, 5, 8};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_empty_input) {
  constexpr size_t kCount = 0;
  std::vector<int> input(kCount, 0);
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_single_element) {
  constexpr size_t kCount = 1;
  std::vector<int> input = {42};
  std::vector<int> expected_output = {42};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_all_same_elements) {
  constexpr size_t kCount = 6;
  std::vector<int> input = {5, 5, 5, 5, 5, 5};
  std::vector<int> expected_output = {5, 5, 5, 5, 5, 5};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_large_numbers) {
  constexpr size_t kCount = 6;
  std::vector<int> input = {5000, 3000, 8000, 1000, 4000, 6000};
  std::vector<int> expected_output = {1000, 3000, 4000, 5000, 6000, 8000};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_with_negative_and_positive_numbers) {
  constexpr size_t kCount = 6;
  std::vector<int> input = {-5, 3, -8, 1, -4, 2};
  std::vector<int> expected_output = {-8, -5, -4, 1, 2, 3};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_only_negative_numbers) {
  constexpr size_t kCount = 4;
  std::vector<int> input = {-5, -3, -8, -1};
  std::vector<int> expected_output = {-8, -5, -3, -1};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_with_zero_and_negative_numbers) {
  constexpr size_t kCount = 5;
  std::vector<int> input = {-5, 0, -3, -8, 0};
  std::vector<int> expected_output = {-8, -5, -3, 0, 0};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_only_positive_numbers) {
  constexpr size_t kCount = 4;
  std::vector<int> input = {5, 3, 8, 1};
  std::vector<int> expected_output = {1, 3, 5, 8};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_sort_single_negative_number) {
  constexpr size_t kCount = 1;
  std::vector<int> input = {-42};
  std::vector<int> expected_output = {-42};
  std::vector<int> output(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_validation_fails_due_to_mismatched_sizes) {
  std::vector<int> input = {5, 3, 8, 1, 4};
  std::vector<int> output(input.size() + 1, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_FALSE(test_task_omp.Validation());
}

TEST(mezhuev_m_bitwise_integer_sort_omp, test_validation_fails_due_to_empty_outputs) {
  std::vector<int> input = {5, 3, 8, 1, 4};
  std::vector<int> output;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_omp::SortOpenMP test_task_omp(task_data_omp);

  ASSERT_FALSE(test_task_omp.Validation());
}