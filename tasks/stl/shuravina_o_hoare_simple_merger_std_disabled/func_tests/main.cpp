#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

TEST(shuravina_o_hoare_simple_merger_stl, test_sort) {
  std::vector<int> in = {5, 2, 9, 1, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  EXPECT_TRUE(test_task->Validation());
  EXPECT_TRUE(test_task->PreProcessing());
  EXPECT_TRUE(test_task->Run());
  EXPECT_TRUE(test_task->PostProcessing());

  std::vector<int> expected = {1, 2, 5, 5, 6, 9};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_stl, test_empty_array) {
  std::vector<int> in = {};
  std::vector<int> out = {};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (!in.empty()) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  } else {
    task_data->inputs.emplace_back(nullptr);
  }
  task_data->inputs_count.emplace_back(in.size());

  if (!out.empty()) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  } else {
    task_data->outputs.emplace_back(nullptr);
  }
  task_data->outputs_count.emplace_back(out.size());
}

TEST(shuravina_o_hoare_simple_merger_stl, test_sorted_array) {
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  EXPECT_TRUE(test_task->Validation());
  EXPECT_TRUE(test_task->PreProcessing());
  EXPECT_TRUE(test_task->Run());
  EXPECT_TRUE(test_task->PostProcessing());

  std::vector<int> expected = {1, 2, 3, 4, 5};
  EXPECT_EQ(out, expected);
}