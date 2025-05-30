#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

namespace {

bool IsSorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> GenerateRandomArray(size_t size, int min_val, int max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distrib(gen);
  }
  return arr;
}

}  // namespace

TEST(shuravina_o_hoare_simple_merger_tbb, test_random_array) {
  const size_t array_size = 1000;
  const int min_val = -1000;
  const int max_val = 1000;

  std::vector<int> in = GenerateRandomArray(array_size, min_val, max_val);
  std::vector<int> out(in.size(), 0);

  ASSERT_FALSE(IsSorted(in));

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  auto sorter = std::make_shared<shuravina_o_hoare_simple_merger_tbb::HoareSortTBB>(task_data_tbb);
  ASSERT_EQ(sorter->Validation(), true);
  sorter->PreProcessing();
  sorter->Run();
  sorter->PostProcessing();

  ASSERT_TRUE(IsSorted(out));
}

TEST(shuravina_o_hoare_simple_merger_tbb, test_sorted_array) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  auto sorter = std::make_shared<shuravina_o_hoare_simple_merger_tbb::HoareSortTBB>(task_data_tbb);
  ASSERT_EQ(sorter->Validation(), true);
  sorter->PreProcessing();
  sorter->Run();
  sorter->PostProcessing();

  EXPECT_EQ(out, in);
}