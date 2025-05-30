#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_stl.hpp"

TEST(mezhuev_m_bitwise_integer_sort_stl, PreProcessingTest) {
  std::vector<int> input = {3, -1, 0, 5, -2, 4, -3};
  std::vector<int> expected_output = {-3, -2, -1, 0, 3, 4, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  ASSERT_TRUE(sort_task.PreProcessingImpl());

  EXPECT_EQ(sort_task.GetInput(), input);
  EXPECT_EQ(sort_task.GetMaxValue(), 5);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, ValidationTest) {
  std::vector<int> input = {3, -1, 0, 5, -2, 4, -3};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  ASSERT_TRUE(sort_task.ValidationImpl());
}

TEST(mezhuev_m_bitwise_integer_sort_stl, RunTest) {
  std::vector<int> input = {3, -1, 0, 5, -2, 4, -3};
  std::vector<int> expected_output = {-3, -2, -1, 0, 3, 4, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();

  EXPECT_EQ(sort_task.GetOutput(), expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, PostProcessingTest) {
  std::vector<int> input = {3, -1, 0, 5, -2, 4, -3};
  std::vector<int> expected_output = {-3, -2, -1, 0, 3, 4, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, EmptyInputTest) {
  std::vector<int> input = {};
  std::vector<int> expected_output = {};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, SortedInputTest) {
  std::vector<int> input = {-5, -3, 0, 1, 2, 4, 6};
  std::vector<int> expected_output = {-5, -3, 0, 1, 2, 4, 6};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, ReverseSortedInputTest) {
  std::vector<int> input = {6, 4, 2, 1, 0, -3, -5};
  std::vector<int> expected_output = {-5, -3, 0, 1, 2, 4, 6};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, InputWithDuplicatesTest) {
  std::vector<int> input = {3, 1, 2, 3, 5, 2, 3};
  std::vector<int> expected_output = {1, 2, 2, 3, 3, 3, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, InputWithNegativeValuesTest) {
  std::vector<int> input = {-3, -1, -4, -2, 5, 0, 3};
  std::vector<int> expected_output = {-4, -3, -2, -1, 0, 3, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, EdgeCaseTest) {
  std::vector<int> input = {42};
  std::vector<int> expected_output = {42};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(expected_output.data()));
  task_data->outputs_count.push_back(expected_output.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL sort_task(task_data);

  sort_task.PreProcessingImpl();
  sort_task.RunImpl();
  sort_task.PostProcessingImpl();

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::vector<int> output(out_ptr, out_ptr + input.size());

  EXPECT_EQ(output, expected_output);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, test_sort_random) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = rand() % 10000;
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), true);

  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, test_sort_reverse) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(kCount - i - 1);
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), true);

  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(mezhuev_m_bitwise_integer_sort_stl, test_sort_large) {
  constexpr size_t kCount = 10000;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = rand() % 100000;
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_stl::SortSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), true);

  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}