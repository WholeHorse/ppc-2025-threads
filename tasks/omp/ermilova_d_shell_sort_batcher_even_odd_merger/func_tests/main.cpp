#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_omp.hpp"

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

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_create_empty_input) {
  // Create data
  std::vector<int> in = {};
  std::vector<int> out(in);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  ASSERT_FALSE(sut.Validation());
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_create_input_and_output_with_different_size) {
  // Create data
  std::vector<int> in = GenerateRandomVector(6);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  ASSERT_FALSE(sut.Validation());
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_single_element) {
  // Create data
  std::vector<int> in = GenerateRandomVector(1);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_small_even_size) {
  // Create data
  std::vector<int> in = {3, 1, 4, 2};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_small_odd_size) {
  // Create data
  std::vector<int> in = {5, 2, 3};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_positive_values) {
  // Create data
  std::vector<int> in = {578, 23546, 1231, 6, 18247, 789, 2348, 3, 213980, 123345};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_negative_values) {
  // Create data
  std::vector<int> in = {-578, -23546, -1231, -6, -18247, -789, -2348, -3, -213980, -123345};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_repeating_values) {
  // Create data
  std::vector<int> in = {9, 10, 8, 9399, 10, 10, 546, 2387, 3728};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_doubly_decreasing_values) {
  // Create data
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_descending_sorted) {
  // Create data
  std::vector<int> in = {5, 4, 3, 2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_ascending_sorted) {
  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_all_equal_elements) {
  // Create data
  std::vector<int> in = {7, 7, 7, 7, 7};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_duplicates_elements) {
  // Create data
  std::vector<int> in = {1, 9, 7, 7, 3, 11, 11, 50, 1, 98, 31};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_10_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(10);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_100_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(100);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_250_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(250);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_327_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(327);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_8_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(8);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_127_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(127);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_347_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(347);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_109) {
  // Create data
  std::vector<int> in = GenerateRandomVector(109);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_128_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(128);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_27_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(27);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_309_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(309);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_50_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(50);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_229) {
  // Create data
  std::vector<int> in = GenerateRandomVector(229);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_400) {
  // Create data
  std::vector<int> in = GenerateRandomVector(400);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}
