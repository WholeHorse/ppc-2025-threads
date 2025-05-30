#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "all/morozov_e_lineare_image_filtering_block_gaussian/include/ops_all.hpp"
#include "core/task/include/task.hpp"

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
      vector[(i * m) + j] = distrib(gen);  // Генерация случайного числа
    }
  }
  return vector;
}
void ApplyGaussianFilter(const std::vector<double> &image, std::vector<double> &result, int n, int m) {
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
        result[(i * m) + j] = image[(i * m) + j];
        continue;
      }

      double sum = 0.0;
      for (int ki = -1; ki <= 1; ++ki) {
        for (int kj = -1; kj <= 1; ++kj) {
          sum += image[((i + ki) * m) + (j + kj)] * kernel[ki + 1][kj + 1];
        }
      }
      result[(i * m) + j] = sum;
    }
  }
}
std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<double> &input, std::vector<double> &output, int n,
                                                    int m) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  // Добавление входных данных
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(n);
  task_data->inputs_count.emplace_back(m);

  // Добавление выходных данных
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(n);
  task_data->outputs_count.emplace_back(m);

  return task_data;
}
}  // namespace

TEST(morozov_e_lineare_image_filtering_block_gaussian_all, empty_image_test) {
  boost::mpi::communicator world;
  int n = 0;
  int m = 0;
  std::vector<double> image(n * m, 1);
  std::vector<double> image_res(n * m, 1);

  // Create task_data
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_sequential.Validation(), false);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, size_input_not_equal_size_output_test1) {
  boost::mpi::communicator world;
  int n = 0;
  int m = 0;
  std::vector<double> image(n * m, 1);
  std::vector<double> image_res((n + 1) * m, 1);

  // Create task_data
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_sequential.Validation(), false);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, size_input_not_equal_size_output_test2) {
  boost::mpi::communicator world;
  int n = 0;
  int m = 0;
  std::vector<double> image(n * m, 0);
  std::vector<double> image_res(n * m, 0);

  // Create task_data
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_sequential.Validation(), false);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test1) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(image, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test2) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
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
  if (world.rank() == 0) {
    EXPECT_EQ(real_res, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test3) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
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
  if (world.rank() == 0) {
    EXPECT_EQ(real_res, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test4) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(image, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test5) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
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
  if (world.rank() == 0) {
    EXPECT_EQ(real_res, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test6) {
  boost::mpi::communicator world;
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
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
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
  if (world.rank() == 0) {
    EXPECT_EQ(real_res, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, main_test7_) {
  boost::mpi::communicator world;
  int n = 2;
  int m = 3;
  // clang-format off
  std::vector<double> image = 
   {1, 6, 7,
	8, 2, 1};
  // clang-format on
  std::vector image_res(n * m, 0.0);

  // Create task_data
  auto task_data_seq = CreateTaskData(image, image_res, n, m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(image, image_res);
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, random_test1) {
  boost::mpi::communicator world;
  int n = 1000;
  int m = 1000;
  std::vector image_res(n * m, 0.0);
  std::vector<double> image;
  if (world.rank() == 0) {
    image = GenerateRandomVector(n, m);
  }
  boost::mpi::broadcast(world, image, 0);  // NOLINT(misc-include-cleaner)
  auto task_data_seq = CreateTaskData(image, image_res, n, m);
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> res(n * m);
  ApplyGaussianFilter(image, res, n, m);
  if (world.rank() == 0) {
    ASSERT_EQ(image_res.size(), res.size());
    for (size_t i = 0; i < image_res.size(); ++i) {
      ASSERT_NEAR(image_res[i], res[i], 0.0000001);
    }
  }
}
TEST(morozov_e_lineare_image_filtering_block_gaussian_all, random_test2) {
  boost::mpi::communicator world;
  int n = 17;
  int m = 23;
  std::vector image_res(n * m, 0.0);
  std::vector<double> image;
  if (world.rank() == 0) {
    image = GenerateRandomVector(n, m);
  }
  boost::mpi::broadcast(world, image, 0);  // NOLINT(misc-include-cleaner)
  auto task_data_seq = CreateTaskData(image, image_res, n, m);
  morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> res(n * m);
  ApplyGaussianFilter(image, res, n, m);
  if (world.rank() == 0) {
    ASSERT_EQ(image_res.size(), res.size());
    for (size_t i = 0; i < image_res.size(); ++i) {
      ASSERT_NEAR(image_res[i], res[i], 0.0000001);
    }
  }
}