#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

TEST(kovalchuk_a_shell_sort_func, Test_EmptyArray) {
  std::vector<int> input = {};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task->Validation());
  }
}

TEST(kovalchuk_a_shell_sort_func, Test_AlreadySorted) {
  std::vector<int> input = {1, 2, 3, 4, 5};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(input, output);
  }
}

TEST(kovalchuk_a_shell_sort_func, Test_ReverseSorted) {
  std::vector<int> input = {10, 7, 5, 3, 1};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> expected = {1, 3, 5, 7, 10};
    EXPECT_EQ(expected, output);
  }
}

TEST(kovalchuk_a_shell_sort_func, Test_Duplicates) {
  std::vector<int> input = {5, 2, 5, 1, 2};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> expected = {1, 2, 2, 5, 5};
    EXPECT_EQ(expected, output);
  }
}

TEST(kovalchuk_a_shell_sort_func, Test_NegativeNumbers) {
  std::vector<int> input = {-5, 0, -3, 9, -1};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> expected = {-5, -3, -1, 0, 9};
    EXPECT_EQ(expected, output);
  }
}

TEST(kovalchuk_a_shell_sort_func, Test_ExtremeValues) {
  std::vector<int> input = {INT32_MIN, 0, INT32_MAX};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  if (world.rank() == 0) {
    std::vector<int> expected = {INT32_MIN, 0, INT32_MAX};
    EXPECT_EQ(expected, output);
  }
}
TEST(kovalchuk_a_shell_sort_func, Test_MixedOrder) {
  std::vector<int> input = {4, 1, 7, 2, 9, 3};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  if (world.rank() == 0) {
    std::vector<int> expected = {1, 2, 3, 4, 7, 9};
    EXPECT_EQ(expected, output);
  }
}
TEST(kovalchuk_a_shell_sort_func, Test_SingleElement) {
  std::vector<int> input = {7};
  std::vector<int> output(input.size());
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<kovalchuk_a_shell_sort_all::ShellSortAll>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(input, output);
  }
}
