#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "all/naumov_b_marc_on_bin_image/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<int> GenerateRandomBinaryMatrix(int rows, int cols, double probability) {
  const int total_elements = rows * cols;
  const int target_ones = static_cast<int>(total_elements * probability);

  std::vector<int> matrix(total_elements, 1);

  const int zeros_needed = total_elements - target_ones;

  for (int i = 0; i < zeros_needed; ++i) {
    matrix[i] = 0;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(matrix.begin(), matrix.end(), g);

  return matrix;
}
}  // namespace

TEST(naumov_b_marc_on_bin_image_all, Validation_1) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 2, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);
  EXPECT_FALSE(test_task_all.Validation());
}

TEST(naumov_b_marc_on_bin_image_all, Validation_2) {
  int m = 3;
  int n = 3;

  std::vector<int> in;
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);
  EXPECT_FALSE(test_task_all.Validation());
}

TEST(naumov_b_marc_on_bin_image_all, Validation_3) {
  int m = 0;
  int n = 0;

  std::vector<int> in;
  std::vector<int> out;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);
  EXPECT_FALSE(test_task_all.Validation());
}

TEST(naumov_b_marc_on_bin_image_all, SingleCallonentInCorner) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, RingShape) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, MazeStructure) {
  int m = 7;
  int n = 7;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                         1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 2, 0,
                              2, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, SimpleTest_1) {
  int m = 3;
  int n = 4;

  std::vector<int> in = {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, SimpleTest_2) {
  int m = 4;
  int n = 4;

  std::vector<int> in = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, SimpleTest_3) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, SingleCallonent) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, OnlyBackground) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, MultipleCallonents) {
  int m = 4;
  int n = 4;

  std::vector<int> in = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 0, 3, 0, 4, 5, 0, 6, 0, 0, 7, 0, 8};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, SingleColumn) {
  int m = 5;
  int n = 1;

  std::vector<int> in = {1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3};
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_all, large3) {
  int m = 200;
  int n = 200;

  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(naumov_b_marc_on_bin_image_all, RandomLargeMatrix) {
  const int m = 100;
  const int n = 100;

  auto in = GenerateRandomBinaryMatrix(m, n, 0.1);
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  const size_t ones_count = static_cast<size_t>(std::count(in.begin(), in.end(), 1));
  EXPECT_GE(unique_labels.size(), static_cast<size_t>(ones_count * 0.6));
}

TEST(naumov_b_marc_on_bin_image_all, RandomSparseMatrix) {
  const int m = 50;
  const int n = 50;

  auto in = GenerateRandomBinaryMatrix(m, n, 0.1);
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  const size_t ones_count = static_cast<size_t>(std::count(in.begin(), in.end(), 1));
  EXPECT_GE(unique_labels.size(), static_cast<size_t>(ones_count * 0.6));
}

TEST(naumov_b_marc_on_bin_image_all, RandomDenseMatrix) {
  const int m = 20;
  const int n = 20;

  auto in = GenerateRandomBinaryMatrix(m, n, 0.9);
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  EXPECT_LE(unique_labels.size(), static_cast<size_t>(5));
}

TEST(naumov_b_marc_on_bin_image_all, RandomDenseMatrix2) {
  const int m = 17;
  const int n = 23;

  auto in = GenerateRandomBinaryMatrix(m, n, 0.9);
  std::vector<int> out(m * n, 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());

  std::set<int> unique_labels;
  for (int val : out) {
    if (val > 0) {
      unique_labels.insert(val);
    }
  }

  EXPECT_LE(unique_labels.size(), static_cast<size_t>(5));
}

TEST(naumov_b_marc_on_bin_image_all, ZeroByZeroMatrix) {
  int m = 0;
  int n = 0;

  std::vector<int> in;
  std::vector<int> out;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(m);
  task_data_all->inputs_count.emplace_back(n);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(m);
  task_data_all->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_all::TestTaskALL test_task_all(task_data_all);

  EXPECT_FALSE(test_task_all.Validation());
}