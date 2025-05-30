#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_seq.hpp"

TEST(mezhuev_m_bitwise_integer_sort_seq, TestEmptyArray) {
  std::vector<int> input = {};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_EQ(test_task_sequential.Validation(), true);

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestSingleElementArray) {
  std::vector<int> input = {42};
  std::vector<int> output = {42};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_EQ(test_task_sequential.Validation(), true);

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestIdenticalElementsArray) {
  std::vector<int> input = {7, 7, 7, 7, 7};
  std::vector<int> output = {7, 7, 7, 7, 7};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_EQ(test_task_sequential.Validation(), true);

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestRegularArray) {
  std::vector<int> input = {5, 2, 9, 1, 5, 6};
  std::vector<int> expected = {1, 2, 5, 5, 6, 9};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_EQ(output, expected);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestMixedArray) {
  std::vector<int> input = {5, -2, 9, -1, 5, -6};
  std::vector<int> expected = {-6, -2, -1, 5, 5, 9};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_EQ(test_task_sequential.Validation(), true);

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, expected);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestMismatchedInputOutputSize) {
  std::vector<int> input = {5, 2, 9, 1, 5, 6};
  std::vector<int> output(input.size() - 1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestNullInputPointer) {
  std::vector<int> output = {0, 0, 0, 0, 0, 0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(output.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestNullOutputPointer) {
  std::vector<int> input = {5, 2, 9, 1, 5, 6};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(input.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestRandomGeneratedArray) {
  constexpr int kCount = 1500;

  std::vector<int> input(kCount);
  std::vector<int> output(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    input[i] = std::rand() % 1000 - 500;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_EQ(test_task_sequential.Validation(), true);

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = input;
  std::ranges::sort(expected);

  EXPECT_EQ(output, expected);
}

TEST(mezhuev_m_bitwise_integer_sort_seq, TestReverseOrderArray) {
  std::vector<int> input = {10, 8, 6, 4, 2, 0, -1, -3, -5};
  std::vector<int> expected = {-5, -3, -1, 0, 2, 4, 6, 8, 10};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  mezhuev_m_bitwise_integer_sort_seq::TestTaskSequential test_task_sequential(task_data);

  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_EQ(output, expected);
}