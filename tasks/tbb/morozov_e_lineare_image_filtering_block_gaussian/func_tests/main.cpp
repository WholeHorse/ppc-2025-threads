#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/morozov_e_lineare_image_filtering_block_gaussian/include/ops_tbb.hpp"

namespace {
std::vector<double> GenerateRandomVector(int n, int m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0, 255);
  std::vector<double> vector(n * m);
  // Создание матрицы

  // Заполнение матрицы случайными числами
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      vector[(i * n) + j] = distrib(gen);  // Генерация случайного числа
    }
  }
  return vector;
}
}  // namespace

TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, empty_image_test) {
  int n = 0;
  int m = 0;
  std::vector<int> image(n * m, 1);
  std::vector<int> image_res(n * m, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, size_input_not_equal_size_output_test1) {
  int n = 0;
  int m = 0;
  std::vector<int> image(n * m, 1);
  std::vector<int> image_res((n + 1) * m, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, size_input_not_equal_size_output_test2) {
  int n = 0;
  int m = 0;
  std::vector<int> image(n * m, 0);
  std::vector<int> image_res(n * m, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test1) {
  int n = 5;
  int m = 5;
  // clang-format off
  std::vector<double> image = 
  {1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,
   1, 1, 1, 1, 1,
   1, 1, 1, 1, 1};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test2) {
  int n = 5;
  int m = 5;
  // clang-format off
  std::vector<double> image = 
  {2, 2, 3, 2, 2,
   2, 2, 3, 2, 2,
   2, 2, 3, 2, 2,
   2, 2, 3, 2, 2,
   2, 2, 3, 2, 2};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  // clang-format off
  std::vector<double> real_res = 
  {2, 2, 3, 2, 2,
   2, 2.25, 2.5, 2.25, 2,
   2, 2.25, 2.5, 2.25, 2,
   2, 2.25, 2.5, 2.25, 2,
   2, 2, 3, 2, 2};
  // clang-format on
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test3) {
  int n = 5;
  int m = 5;
  // clang-format off
  std::vector<double> image = 
  {1, 2, 3, 4, 5,
   6, 7, 8, 9, 10,
   1, 2, 3, 4, 5,
   1, 2, 3, 4, 5,
   1, 2, 3, 4, 5};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  // clang-format off
  std::vector<double> real_res = 
  {1, 2, 3, 4, 5,
   6, 4.5, 5.5, 6.5, 10,
   1, 3.25, 4.25, 5.25, 5,
   1, 2, 3, 4, 5,
   1, 2, 3, 4, 5};
  // clang-format on
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test4) {
  int n = 5;
  int m = 5;
  // clang-format off
  std::vector<double> image = 
  {5, 5, 5, 5, 5,
   5, 5, 5, 5, 5,
   5, 5, 5, 5, 5,
   5, 5, 5, 5, 5,
   5, 5, 5, 5, 5};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test5) {
  int n = 5;
  int m = 5;
  // clang-format off
  std::vector<double> image = 
  {1, 2, 3, 4, 5,
   6, 7, 8, 9, 10,
   10, 9, 8, 7, 6,
   5, 4, 3, 2, 1,
   1, 1, 1, 1, 1};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  // clang-format off
  std::vector<double> real_res = 
  {1, 2, 3, 4, 5,
   6, 6.25, 6.75, 7.25, 10,
   10, 7.25, 6.75, 6.25, 6,
   5, 4.5, 3.75, 3, 1,
   1, 1, 1, 1, 1};
  // clang-format on
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test6) {
  int n = 3;
  int m = 3;
  // clang-format off
  std::vector<double> image = 
   {1, 6, 7,
	8, 2, 1,
	8, 2, 4};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  // clang-format off
  std::vector<double> real_res = 
  {1, 6, 7,
   8, 3.875, 1,
   8, 2, 4};
  // clang-format on
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, main_test7_) {
  int n = 2;
  int m = 3;
  // clang-format off
  std::vector<double> image = 
   {1, 6, 7,
	8, 2, 1};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, random_test1) {
  int n = 3;
  int m = 3;
  std::vector image_res(n * m, 0.0);
  std::vector<double> image = GenerateRandomVector(n, m);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> res(n * m);
  // clang-format off
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16},
      {2.0 / 16, 4.0 / 16, 2.0 / 16},
      {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  // clang-format on
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
        res[(i * m) + j] = image[(i * m) + j];
      } else {
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += image[((i + ki) * m) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        res[(i * m) + j] = sum;
      }
    }
  }
  EXPECT_EQ(image_res, res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_tbb, random_test2) {
  int n = 17;
  int m = 23;
  std::vector image_res(n * m, 0.0);
  std::vector<double> image = GenerateRandomVector(n, m);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_tbb::TestTaskTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> res(n * m);
  // clang-format off
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16},
      {2.0 / 16, 4.0 / 16, 2.0 / 16},
      {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  // clang-format on
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
        res[(i * m) + j] = image[(i * m) + j];
      } else {
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += image[((i + ki) * m) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        res[(i * m) + j] = sum;
      }
    }
  }
  EXPECT_EQ(image_res, res);
}