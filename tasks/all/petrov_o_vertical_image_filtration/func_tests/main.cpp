#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/petrov_o_vertical_image_filtration/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<int> GenerateRandomInput(size_t width, size_t height) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 100);

  std::vector<int> input(width * height);
  for (auto &val : input) {
    val = dist(gen);
  }
  return input;
}

std::vector<int> ComputeReference(const std::vector<int> &in, size_t width, size_t height,
                                  const std::vector<float> &kernel) {
  std::vector<int> result((width - 2) * (height - 2), 0);
  if (width < 3 || height < 3) {
    return result;
  }

  for (size_t i = 1; i < height - 1; ++i) {
    for (size_t j = 1; j < width - 1; ++j) {
      float sum = 0.0F;
      for (int ki = -1; ki <= 1; ++ki) {
        for (int kj = -1; kj <= 1; ++kj) {
          const int input_val = in[((i + ki) * width) + (j + kj)];
          const float weight = kernel[((ki + 1) * 3) + (kj + 1)];
          sum += static_cast<float>(input_val) * weight;
        }
      }
      result[((i - 1) * (width - 2)) + (j - 1)] = static_cast<int>(sum);
    }
  }
  return result;
}

}  // namespace

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_3x3) {
  constexpr size_t kWidth = 3;
  constexpr size_t kHeight = 3;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    std::vector<int> expected_out = {5};
    EXPECT_EQ(out, expected_out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_5x5) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    std::vector<int> expected_out = {7, 8, 9, 12, 13, 14, 17, 18, 19};
    EXPECT_EQ(out, expected_out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_const_data) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;

  // Create data
  std::vector<int> in(kWidth * kHeight, 3);
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);
  std::vector<int> expected_out((kWidth - 2) * (kHeight - 2), 3);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    EXPECT_EQ(out, expected_out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_large_data) {
  constexpr size_t kWidth = 500;
  constexpr size_t kHeight = 500;

  // Create data
  std::vector<int> in(kWidth * kHeight, 1);
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);
  std::vector<int> expected_out((kWidth - 2) * (kHeight - 2), 1);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    EXPECT_EQ(out, expected_out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_rectangular) {
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    std::vector<int> expected_out = {6, 7};
    EXPECT_EQ(out, expected_out);
  }
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_1x1) {
  constexpr size_t kWidth = 1;
  constexpr size_t kHeight = 1;

  // Create data
  std::vector<int> in = {1};
  std::vector<int> out(0, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);

  // Validation can be checked on all ranks, as it's a prerequisite for running.
  // If it fails on any rank, the test setup is incorrect.
  ASSERT_EQ(test_task_all.Validation(), false);
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_empty) {
  constexpr size_t kWidth = 0;
  constexpr size_t kHeight = 0;

  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), false);
}

TEST(petrov_o_vertical_image_filtration_all, test_gaussian_filter_random) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;

  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  std::vector<int> in;
  petrov_o_vertical_image_filtration_all::TaskAll test_task_all(task_data_all);

  in = GenerateRandomInput(kWidth, kHeight);

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(kWidth);
  task_data_all->inputs_count.emplace_back(kHeight);
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Check result
  if (test_task_all.GetRank() == 0) {
    const std::vector<float> gaussian_kernel = {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 4.0F / 16.0F,
                                                2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F};

    std::vector<int> out_ref = ComputeReference(in, kWidth, kHeight, gaussian_kernel);

    EXPECT_EQ(out.size(), out_ref.size());
    for (size_t idx = 0; idx < out.size(); ++idx) {
      EXPECT_EQ(out[idx], out_ref[idx]) << "Error on index " << idx;
    }
  }
}