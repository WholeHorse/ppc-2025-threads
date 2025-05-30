#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "all/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenerateRandomVector(size_t size, int lower_bound = -1000, int upper_bound = 1000) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = static_cast<int>(lower_bound + (gen() % (upper_bound - lower_bound + 1)));
  }
  return vec;
}
}  // namespace

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_create_empty_input) {
  // Create data
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> out;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {};
    out = in;
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(sut.Validation());
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_create_input_and_output_with_different_size) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(6);
    out = std::vector<int>(1, 0);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(sut.Validation());
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_single_element) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;

  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(1);
    out = std::vector(in.size(), 0);
    expected = in;
    std::ranges::sort(expected);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  ASSERT_TRUE(sut.Validation());
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_small_even_size) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {3, 1, 4, 2};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_small_odd_size) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {5, 2, 3};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_positive_values) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {578, 23546, 1231, 6, 18247, 789, 2348, 3, 213980, 123345};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_negative_values) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {-578, -23546, -1231, -6, -18247, -789, -2348, -3, -213980, -123345};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_repeating_values) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {9, 10, 8, 9399, 10, 10, 546, 2387, 3728};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_doubly_decreasing_values) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_descending_sorted) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {5, 4, 3, 2, 1};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_ascending_sorted) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {1, 2, 3, 4, 5};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_all_equal_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {7, 7, 7, 7, 7};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_duplicates_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {1, 9, 7, 7, 3, 11, 11, 50, 1, 98, 31};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_10_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(10);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_100_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(100);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_1000_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(1000);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_10000_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(10000);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_8_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(8);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_127_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(127);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_347_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(347);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_with_boundary_sedgwick_gap_109) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(109);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_128_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(128);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_27_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(27);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_809_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(809);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_500_random_elements) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(500);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_with_boundary_sedgwick_gap_729) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(729);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_with_boundary_sedgwick_gap_457) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateRandomVector(457);
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_with_alternating_positive_negative) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_all, test_sort_with_max_min) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> expected;
  boost::mpi::communicator world;

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = {std::numeric_limits<int>::max(), 3456, 12, -234, std::numeric_limits<int>::min(), 1244, 0, 781, 237};
    out.resize(in.size());
    expected = in;
    std::ranges::sort(expected);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }
  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_all::AllTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected, out);
  }
}