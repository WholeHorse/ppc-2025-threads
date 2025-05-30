#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif

#include "../include/chc.hpp"
#include "../include/chc_all.hpp"
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

using namespace voroshilov_v_convex_hull_components_all;

namespace {

void SortHulls(std::vector<Hull>& hulls) {
  std::ranges::sort(hulls, [](const Hull& a, const Hull& b) {
    const Pixel& left_top_a = *std::ranges::min_element(
        a, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });
    const Pixel& left_top_b = *std::ranges::min_element(
        b, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });

    return left_top_a.x < left_top_b.x || (left_top_a.x == left_top_b.x && left_top_a.y < left_top_b.y);
  });
}

bool ValidationTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
    task_data_all->inputs_count.emplace_back(pixels.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
    task_data_all->outputs_count.emplace_back(0);
  }
  ChcTaskALL chc_task_all(task_data_all);
  return chc_task_all.ValidationImpl();
}

std::vector<Hull> SimpleRunTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
    task_data_all->inputs_count.emplace_back(pixels.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
    task_data_all->outputs_count.emplace_back(0);
  }
  ChcTaskALL chc_task_all(task_data_all);
  chc_task_all.ValidationImpl();
  chc_task_all.PreProcessingImpl();
  chc_task_all.RunImpl();
  chc_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    int hulls_size = static_cast<int>(task_data_all->outputs_count[0]);
    std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

    return hulls;
  }
  return {};
}

#ifndef _WIN32

bool ImageRunTest(std::string& src_path, std::string& exp_path) {
  // Load source image:
  cv::Mat src_image = cv::imread(src_path);
  if (src_image.empty()) {
    return false;
  }

  // Convert to shades of gray:
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, cv::COLOR_BGR2GRAY);

  // Convert to black and white:
  cv::Mat bin_image;
  cv::threshold(gray_image, bin_image, 128, 1, cv::THRESH_BINARY);

  // Convert to std::vector<int>:
  std::vector<int> pixels(bin_image.rows * bin_image.cols);
  for (int i = 0; i < bin_image.rows; i++) {
    for (int j = 0; j < bin_image.cols; j++) {
      pixels[(i * bin_image.cols) + j] = bin_image.at<uchar>(i, j);
    }
  }

  int* p_height = &bin_image.rows;
  int height = bin_image.rows;
  int* p_width = &bin_image.cols;
  int width = bin_image.cols;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
    task_data_all->inputs_count.emplace_back(pixels.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
    task_data_all->outputs_count.emplace_back(0);
  }
  ChcTaskALL chc_task_all(task_data_all);
  chc_task_all.ValidationImpl();
  chc_task_all.PreProcessingImpl();
  chc_task_all.RunImpl();
  chc_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    int hulls_size = static_cast<int>(task_data_all->outputs_count[0]);
    std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

    // Draw hulls on source image:
    for (Hull hull : hulls) {
      for (size_t i = 0; i < hull.size() - 1; i++) {
        cv::circle(src_image, cv::Point(hull[i].x, hull[i].y), 2, cv::Scalar(0, 0, 255), cv::FILLED);

        cv::line(src_image, cv::Point(hull[i].x, hull[i].y), cv::Point(hull[i + 1].x, hull[i + 1].y),
                 cv::Scalar(0, 0, 255), 1);
      }
      cv::circle(src_image, cv::Point(hull[hull.size() - 1].x, hull[hull.size() - 1].y), 2, cv::Scalar(0, 0, 255),
                 cv::FILLED);

      cv::line(src_image, cv::Point(hull[hull.size() - 1].x, hull[hull.size() - 1].y), cv::Point(hull[0].x, hull[0].y),
               cv::Scalar(0, 0, 255), 1);
    }

    // Load expected image:
    cv::Mat exp_image = cv::imread(exp_path);
    if (exp_image.empty()) {
      src_image.release();
      gray_image.release();
      bin_image.release();
      return false;
    }

    // cv::imwrite(exp_path, src_image);

    // Compare edited source image with expected image:
    double difference = cv::norm(src_image, exp_image);

    src_image.release();
    gray_image.release();
    bin_image.release();
    exp_image.release();
    // They are same if difference == 0.0
    return difference <= 0.0;
  }
  return true;
}

#endif

}  // namespace
TEST(voroshilov_v_convex_hull_components_all, simpleValidationTest) {
  // clang-format off
  std::vector<int> pixels = 
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 0;
  int width = 3;

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_FALSE(ValidationTest(height, width, pixels));
  }
}

TEST(voroshilov_v_convex_hull_components_all, simpleTest0Components) {
  // clang-format off
  std::vector<int> pixels =
  {0, 0, 0,
   0, 0, 0,
   0, 0, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(result_hulls.empty());
  }
}

TEST(voroshilov_v_convex_hull_components_all, simpleTest1Component) {
  // clang-format off
  std::vector<int> pixels =
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;
  Hull hull = {{1, 0}, {0, 1}, {1, 2}, {2, 1}};  // First coordinate is Y, second is X!!!
  expect_hulls.push_back(hull);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_EQ(result_hulls.size(), expect_hulls.size());

    for (size_t i = 0; i < result_hulls.size(); i++) {
      for (size_t j = 0; j < result_hulls[i].size(); j++) {
        EXPECT_EQ(result_hulls[i][j], expect_hulls[i][j]);
      }
    }
  }
}

TEST(voroshilov_v_convex_hull_components_all, simpleTest3Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 1, 1,
   1, 0, 0, 0, 1, 1,
   0, 0, 1, 0, 0, 0,
   0, 1, 1, 1, 0, 0,
   1, 1, 0, 0, 0, 0};
  // clang-format on
  int height = 5;
  int width = 6;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  Hull hull1 = {{0, 0}, {0, 1}, {1, 0}};
  expect_hulls.push_back(hull1);

  Hull hull2 = {{0, 4}, {0, 5}, {1, 5}, {1, 4}};
  expect_hulls.push_back(hull2);

  Hull hull3 = {{4, 0}, {2, 2}, {3, 3}, {4, 1}};
  expect_hulls.push_back(hull3);

  SortHulls(result_hulls);
  SortHulls(expect_hulls);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_EQ(result_hulls.size(), expect_hulls.size());

    for (size_t i = 0; i < result_hulls.size(); i++) {
      for (size_t j = 0; j < result_hulls[i].size(); j++) {
        EXPECT_EQ(result_hulls[i][j], expect_hulls[i][j]);
      }
    }
  }
}

TEST(voroshilov_v_convex_hull_components_all, simpleTest5Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
   1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
   0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
   0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
   0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
   0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  // clang-format on
  int height = 11;
  int width = 10;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  Hull hull1 = {{0, 0}, {0, 1}, {1, 2}, {2, 2}, {2, 1}, {1, 0}};
  expect_hulls.push_back(hull1);

  Hull hull2 = {{1, 4}, {0, 5}, {0, 6}, {1, 7}, {2, 6}, {2, 5}};
  expect_hulls.push_back(hull2);

  Hull hull3 = {{8, 7}, {5, 8}, {6, 9}, {10, 9}};
  expect_hulls.push_back(hull3);

  Hull hull4 = {{7, 1}, {6, 2}, {7, 3}, {8, 2}};
  expect_hulls.push_back(hull4);

  Hull hull5 = {{9, 4}, {8, 5}, {10, 5}};
  expect_hulls.push_back(hull5);

  SortHulls(result_hulls);
  SortHulls(expect_hulls);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_EQ(result_hulls.size(), expect_hulls.size());

    for (size_t i = 0; i < result_hulls.size(); i++) {
      for (size_t j = 0; j < result_hulls[i].size(); j++) {
        EXPECT_EQ(result_hulls[i][j], expect_hulls[i][j]);
      }
    }
  }
}

#ifndef _WIN32

TEST(voroshilov_v_convex_hull_components_all, imageTest0) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/0_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/0_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest0Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/0_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/0_incorrect.png");

  bool res = ImageRunTest(src_path, inc_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_FALSE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest1) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/1_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/1_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest1Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/1_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/1_incorrect.png");

  bool res = ImageRunTest(src_path, inc_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_FALSE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest2) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/2_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/2_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest3) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/3_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/3_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest4) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/4_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/4_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest5) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/5_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/5_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest6) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/6_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/6_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

TEST(voroshilov_v_convex_hull_components_all, imageTest7) {
  std::string src_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/7_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("all/voroshilov_v_convex_hull_components/data/7_expected.png");

  bool res = ImageRunTest(src_path, exp_path);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ASSERT_TRUE(res);
  }
}

#endif
