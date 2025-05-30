#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "all/kapustin_i_jarv_alg/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<std::pair<int, int>> GenerateRandomPoints(size_t count, int min_x, int max_x, int min_y, int max_y) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist_x(min_x, max_x);
  std::uniform_int_distribution<int> dist_y(min_y, max_y);

  std::vector<std::pair<int, int>> random_points;
  random_points.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    random_points.emplace_back(dist_x(rng), dist_y(rng));
  }

  return random_points;
}
}  // namespace

TEST(KapustinJarvAlgALLTest, FixedPointsWithRandomNoise) {
  std::vector<std::pair<int, int>> fixed_points = {{-1000, -1000}, {1000, -1000}, {1000, 1000}, {-1000, 1000}};

  auto random_points = GenerateRandomPoints(100, -900, 900, -900, 900);

  std::vector<std::pair<int, int>> input_points = fixed_points;
  input_points.insert(input_points.end(), random_points.begin(), random_points.end());

  std::vector<std::pair<int, int>> output_result(fixed_points.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());

  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  EXPECT_EQ(output_result.size(), fixed_points.size());
  for (size_t i = 0; i < fixed_points.size(); ++i) {
    EXPECT_EQ(output_result[i].first, fixed_points[i].first);
    EXPECT_EQ(output_result[i].second, fixed_points[i].second);
  }
}

TEST(KapustinJarvAlgAllTest, PureTriangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgAllTest, SquarePlusOne) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 4}, {4, 4}, {4, 0}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgAllTest, DuplicatePoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}, {0, 5}, {5, 5}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {5, 0}, {5, 5}, {0, 5}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgAllTest, SinglePoint) {
  std::vector<std::pair<int, int>> input_points = {{1, 1}};
  std::vector<std::pair<int, int>> expected_result = {{1, 1}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  ASSERT_EQ(output_result.size(), expected_result.size());
  EXPECT_EQ(output_result[0].first, expected_result[0].first);
  EXPECT_EQ(output_result[0].second, expected_result[0].second);
}

TEST(KapustinJarvAlgAllTest, CollinearPoints2) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 4}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgAllTest, TwoPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {4, 4}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 4}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  ASSERT_EQ(output_result.size(), expected_result.size());
  EXPECT_EQ(output_result[0].first, expected_result[0].first);
  EXPECT_EQ(output_result[0].second, expected_result[0].second);
  EXPECT_EQ(output_result[1].first, expected_result[1].first);
  EXPECT_EQ(output_result[1].second, expected_result[1].second);
}

TEST(KapustinJarvAlgAllTest, Circle) {
  std::vector<std::pair<int, int>> input_points = {{0, 5},  {3, 4},   {4, 3},   {5, 0},  {4, -3}, {3, -4},
                                                   {0, -5}, {-3, -4}, {-4, -3}, {-5, 0}, {-4, 3}, {-3, 4}};
  std::vector<std::pair<int, int>> expected_result = {{-5, 0}, {-4, -3}, {-3, -4}, {0, -5}, {3, -4}, {4, -3},
                                                      {5, 0},  {4, 3},   {3, 4},   {0, 5},  {-3, 4}, {-4, 3}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_all->inputs_count.emplace_back(input_points.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_all->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}