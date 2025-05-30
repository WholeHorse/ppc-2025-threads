#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace sozonov_i_image_filtering_block_partitioning_all {

std::vector<double> ZeroEdges(std::vector<double> img, int wdth, int hght) {
  for (int i = 0; i < wdth; ++i) {
    img[i] = 0;
    img[((hght - 1) * wdth) + i] = 0;
  }
  for (int i = 1; i < hght - 1; ++i) {
    img[i * wdth] = 0.75;
    img[(i * wdth) + wdth - 1] = 0.75;
  }

  return img;
}

}  // namespace sozonov_i_image_filtering_block_partitioning_all

TEST(sozonov_i_image_filtering_block_partitioning_all, test_empty_image) {
  boost::mpi::communicator world;

  const int width = 0;
  const int height = 0;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_image_less_than_3x3) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 2;

  // Create data
  std::vector<double> in = {4, 6, 8, 24};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_wrong_pixels) {
  boost::mpi::communicator world;

  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {143, 6, 853, -24, 31, -25, 1, 5, -7, 361, 28, 98, -45, 982, 461};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_13x3) {
  boost::mpi::communicator world;

  const int width = 13;
  const int height = 3;

  // Create data
  std::vector<double> in = {34,  24, 27, 67, 42, 48, 93, 26,  47, 2,  34, 13, 81, 24, 32, 12, 34, 72, 112, 16,
                            123, 54, 32, 45, 67, 23, 76, 154, 52, 78, 94, 10, 23, 54, 93, 56, 23, 56, 72};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,    0,  0,       0,       0,       0,       0,       0,       0,  0,
                             0,    0,  0,       34.875,  46.5625, 41.3125, 50.5625, 65.9375, 62, 56.5,
                             65.5, 59, 39.9375, 38.6875, 47,      37.5625, 0,       0,       0,  0,
                             0,    0,  0,       0,       0,       0,       0,       0,       0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_15x5) {
  boost::mpi::communicator world;

  const int width = 15;
  const int height = 5;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0,  0,  0,  0,  0,  0,     0,  0,  0,  0,    0,     0,  0,  0,  11.5, 16, 17, 18,
                             19, 20, 21, 22, 23, 24, 25,    26, 27, 28, 21.5, 22.75, 31, 32, 33, 34,   35, 36, 37,
                             38, 39, 40, 41, 42, 43, 32.75, 34, 46, 47, 48,   49,    50, 51, 52, 53,   54, 55, 56,
                             57, 58, 44, 0,  0,  0,  0,     0,  0,  0,  0,    0,     0,  0,  0,  0,    0,  0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_16x4) {
  boost::mpi::communicator world;

  const int width = 16;
  const int height = 4;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                             12.25, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 23,
                             24.25, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 35,
                             0,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_17x3) {
  boost::mpi::communicator world;

  const int width = 17;
  const int height = 3;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                             13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 24.5,
                             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_100x100) {
  boost::mpi::communicator world;

  const int width = 100;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_150x100) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_120x200) {
  boost::mpi::communicator world;

  const int width = 120;
  const int height = 200;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}