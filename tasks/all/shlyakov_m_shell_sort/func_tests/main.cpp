#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "all/shlyakov_m_shell_sort/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenerateRandomArray(size_t size) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<int> distribution_range(-1000, 1000);
  int min_val = distribution_range(generator);
  int max_val = distribution_range(generator);

  if (min_val > max_val) {
    std::swap(min_val, max_val);
  }

  std::uniform_int_distribution<int> distribution(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distribution(generator);
  }
  return arr;
}

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
}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Empty_Array) {
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> out;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Already_Sorted_Array) {
  boost::mpi::communicator world;
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    ASSERT_EQ(in, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Reverse_Sorted_Array) {
  boost::mpi::communicator world;
  std::vector<int> in = {5, 4, 3, 2, 1};
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = {1, 2, 3, 4, 5};
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_Small) {
  boost::mpi::communicator world;
  std::vector<int> in = GenerateRandomArray(10);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_Large) {
  boost::mpi::communicator world;
  size_t array_size = 200;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Simple_Size) {
  boost::mpi::communicator world;
  size_t array_size = 241;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_500) {
  boost::mpi::communicator world;
  size_t array_size = 500;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_501) {
  boost::mpi::communicator world;
  size_t array_size = 501;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_1000) {
  boost::mpi::communicator world;
  size_t array_size = 1000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_1001) {
  boost::mpi::communicator world;
  size_t array_size = 1001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_999) {
  boost::mpi::communicator world;
  size_t array_size = 999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_10000) {
  boost::mpi::communicator world;
  size_t array_size = 10000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_10001) {
  boost::mpi::communicator world;
  size_t array_size = 10001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_9999) {
  boost::mpi::communicator world;
  size_t array_size = 9999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_15000) {
  boost::mpi::communicator world;
  size_t array_size = 15000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_15001) {
  boost::mpi::communicator world;
  size_t array_size = 15001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_14999) {
  boost::mpi::communicator world;
  size_t array_size = 14999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_With_Eq_Numbers) {
  boost::mpi::communicator world;
  size_t array_size = 100;
  std::vector<int> in(array_size, 3);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_With_False_Validation) {
  boost::mpi::communicator world;
  size_t array_size = 100;
  std::vector<int> in(array_size, 3);
  std::vector<int> out(in.size() - 1);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);

  if (world.rank() == 0) {
    ASSERT_FALSE(test_task_all.Validation());
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Mersen63) {
  boost::mpi::communicator world;
  size_t array_size = 63;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Mersen127) {
  boost::mpi::communicator world;
  size_t array_size = 127;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Mersen255) {
  boost::mpi::communicator world;
  size_t array_size = 14999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Mersen511) {
  boost::mpi::communicator world;
  size_t array_size = 511;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(shlyakov_m_shell_sort_all, Test_Random_Array_With_Mersen1023) {
  boost::mpi::communicator world;
  size_t array_size = 1023;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(in.data())));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  shlyakov_m_shell_sort_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  if (world.rank() == 0) {
    ASSERT_TRUE(IsSorted(out));
    std::vector<int> expected = in;
    std::ranges::sort(expected);
    ASSERT_EQ(expected, out);
  }
}