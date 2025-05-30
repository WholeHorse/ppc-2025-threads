#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "stl/zaytsev_d_sobel/include/ops_stl.hpp"

#ifndef _WIN32
namespace {
std::vector<int> MatToVector(const cv::Mat &img) {
  std::vector<int> vec;

  vec.reserve(img.rows * img.cols);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      vec.push_back(static_cast<int>(img.at<uchar>(i, j)));
    }
  }
  return vec;
}
}  // namespace
#endif

TEST(zaytsev_d_sobel_stl, test_validation_fail_count_in_out) {
  std::vector<int> in(9, 0);
  std::vector<int> out(10, 0);
  std::vector<int> size = {3, 3};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.push_back(out.size());

  zaytsev_d_sobel_stl::TestTaskSTL task(task_data);
  ASSERT_EQ(task.Validation(), false);
}

TEST(zaytsev_d_sobel_stl, test_validation_fail_small_image) {
  std::vector<int> in(10, 0);
  std::vector<int> out(10, 0);
  std::vector<int> size = {5, 2};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.push_back(out.size());

  zaytsev_d_sobel_stl::TestTaskSTL task(task_data);
  ASSERT_EQ(task.Validation(), false);
}

TEST(zaytsev_d_sobel_stl, test_validation_wrong_size) {
  std::vector<int> in(10, 0);
  std::vector<int> out(10, 0);
  std::vector<int> size = {5, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.push_back(out.size());

  zaytsev_d_sobel_stl::TestTaskSTL task(task_data);
  ASSERT_EQ(task.Validation(), false);
}

TEST(zaytsev_d_sobel_stl, SobelEdgeDetection_5x5) {
  constexpr size_t kSize = 5;
  std::vector<int> input = {0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 10, 10, 10, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0, 0};
  std::vector<int> expected_output = {0,  0, 0, 0,  0,  0,  42, 40, 42, 0, 0, 40, 0,
                                      40, 0, 0, 42, 40, 42, 0,  0,  0,  0, 0, 0};
  std::vector<int> output(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(zaytsev_d_sobel_stl, Sobel_12x12) {
  constexpr size_t kSize = 12;
  std::vector<int> input = {0,   0,   0,   0,   0,   0,   50,  50,  50,  0,   0,   0,   0,   50,  100, 150, 200, 255,
                            255, 200, 150, 100, 50,  0,   0,   100, 150, 200, 255, 255, 255, 255, 200, 150, 100, 0,
                            0,   150, 200, 255, 255, 255, 255, 255, 255, 200, 150, 0,   0,   200, 255, 255, 255, 255,
                            255, 255, 255, 255, 200, 0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
                            50,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 50,  100, 200, 255, 255, 255, 255,
                            255, 255, 255, 255, 200, 100, 150, 150, 200, 255, 255, 255, 255, 255, 255, 200, 150, 150,
                            200, 100, 150, 200, 255, 255, 255, 200, 150, 100, 150, 200, 255, 50,  100, 150, 200, 255,
                            200, 150, 100, 50,  50,  255, 0,   0,   0,   50,  100, 150, 150, 100, 50,  0,   0,   0};
  std::vector<int> expected_output = {
      0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
      0, 255, 255, 255, 255, 77,  77,  255, 255, 255, 255, 0, 0, 255, 255, 255, 77,  0,   0,   77,  255, 255, 255, 0,
      0, 255, 255, 77,  0,   0,   0,   0,   77,  255, 255, 0, 0, 255, 77,  0,   0,   0,   0,   0,   0,   77,  255, 0,
      0, 255, 77,  0,   0,   0,   0,   0,   0,   77,  255, 0, 0, 255, 255, 77,  0,   0,   0,   0,   77,  255, 255, 0,
      0, 255, 255, 255, 77,  0,   77,  239, 255, 255, 219, 0, 0, 255, 255, 255, 255, 110, 255, 255, 255, 255, 255, 0,
      0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
  std::vector<int> output(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(zaytsev_d_sobel_stl, SobelEdgeDetection_UniformImage) {
  constexpr size_t kSize = 5;
  std::vector<int> input(kSize * kSize, 10);
  std::vector<int> expected_output(kSize * kSize, 0);
  std::vector<int> output(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

#ifndef _WIN32
TEST(zaytsev_d_sobel_stl, SobelEdgeDetection_OpenCVImage) {
  cv::Mat input_img =
      cv::imread(ppc::util::GetAbsolutePath("stl/zaytsev_d_sobel/data/inwhite.png"), cv::IMREAD_GRAYSCALE);
  cv::Mat expected_img =
      cv::imread(ppc::util::GetAbsolutePath("stl/zaytsev_d_sobel/data/outputwhite.png"), cv::IMREAD_GRAYSCALE);

  std::vector<int> input = MatToVector(input_img);
  std::vector<int> expected = MatToVector(expected_img);
  std::vector<int> output(input.size(), 0);
  std::vector<int> size = {input_img.cols, input_img.rows};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], expected[i]);
  }
}
#endif

TEST(zaytsev_d_sobel_stl, Sobel_Circle) {
  constexpr size_t kSize = 20;
  std::vector<int> input = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      10,  20,  30,  40,  40,  30,  20,  10,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   10,  30,  50,  70,  90,
      110, 110, 90,  70,  50,  30,  10,  0,   0,   0,   0,   0,   0,   0,   10,  40,  70,  100, 130, 150, 170, 170, 150,
      130, 100, 70,  40,  10,  0,   0,   0,   0,   0,   0,   30,  70,  110, 140, 170, 190, 210, 210, 190, 170, 140, 110,
      70,  30,  0,   0,   0,   0,   0,   10,  50,  100, 140, 170, 200, 220, 230, 230, 220, 200, 170, 140, 100, 50,  10,
      0,   0,   0,   0,   20,  70,  130, 170, 200, 220, 240, 245, 245, 240, 220, 200, 170, 130, 70,  20,  0,   0,   0,
      0,   30,  90,  150, 190, 220, 240, 250, 255, 255, 250, 240, 220, 190, 150, 90,  30,  0,   0,   0,   0,   40,  110,
      170, 210, 230, 245, 255, 255, 255, 255, 245, 230, 210, 170, 110, 40,  0,   0,   0,   0,   40,  110, 170, 210, 230,
      245, 255, 255, 255, 255, 245, 230, 210, 170, 110, 40,  0,   0,   0,   0,   30,  90,  150, 190, 220, 240, 250, 255,
      255, 250, 240, 220, 190, 150, 90,  30,  0,   0,   0,   0,   20,  70,  130, 170, 200, 220, 240, 245, 245, 240, 220,
      200, 170, 130, 70,  20,  0,   0,   0,   0,   10,  50,  100, 140, 170, 200, 220, 230, 230, 220, 200, 170, 140, 100,
      50,  10,  0,   0,   0,   0,   0,   30,  70,  110, 140, 170, 190, 210, 210, 190, 170, 140, 110, 70,  30,  0,   0,
      0,   0,   0,   0,   10,  40,  70,  100, 130, 150, 170, 170, 150, 130, 100, 70,  40,  10,  0,   0,   0,   0,   0,
      0,   0,   10,  30,  50,  70,  90,  110, 110, 90,  70,  50,  30,  10,  0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   10,  20,  30,  40,  40,  30,  20,  10,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0};
  std::vector<int> expected_output = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   14,  44,  82,  121, 150, 150, 121, 82,  44,  14,  0,   0,   0,   0,   0,   0,   0,   0,   14,  58,  134,
      215, 255, 255, 255, 255, 255, 255, 215, 134, 58,  14,  0,   0,   0,   0,   0,   14,  84,  200, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 200, 84,  14,  0,   0,   0,   0,   58,  200, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 200, 58,  0,   0,   0,   14,  134, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 134, 14,  0,   0,   44,  215, 255, 255, 255, 255, 255, 223, 161, 161, 223, 255, 255, 255, 255, 255, 215,
      44,  0,   0,   82,  255, 255, 255, 255, 255, 226, 157, 107, 107, 157, 226, 255, 255, 255, 255, 255, 82,  0,   0,
      121, 255, 255, 255, 255, 223, 157, 91,  47,  47,  91,  157, 223, 255, 255, 255, 255, 121, 0,   0,   150, 255, 255,
      255, 255, 161, 107, 47,  7,   7,   47,  107, 161, 255, 255, 255, 255, 150, 0,   0,   150, 255, 255, 255, 255, 161,
      107, 47,  7,   7,   47,  107, 161, 255, 255, 255, 255, 150, 0,   0,   121, 255, 255, 255, 255, 223, 157, 91,  47,
      47,  91,  157, 223, 255, 255, 255, 255, 121, 0,   0,   82,  255, 255, 255, 255, 255, 226, 157, 107, 107, 157, 226,
      255, 255, 255, 255, 255, 82,  0,   0,   44,  215, 255, 255, 255, 255, 255, 223, 161, 161, 223, 255, 255, 255, 255,
      255, 215, 44,  0,   0,   14,  134, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 134, 14,
      0,   0,   0,   58,  200, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 200, 58,  0,   0,   0,   0,
      14,  84,  200, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 200, 84,  14,  0,   0,   0,   0,   0,   14,  58,
      134, 215, 255, 255, 255, 255, 255, 255, 215, 134, 58,  14,  0,   0,   0,   0,   0,   0,   0,   0,   14,  44,  82,
      121, 150, 150, 121, 82,  44,  14,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0};
  std::vector<int> output(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(zaytsev_d_sobel_stl, RectangularImage_18x20) {
  constexpr int kWidth = 18;
  constexpr int kHeight = 20;

  std::vector<int> input = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
      0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,
      0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,
      0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,
      0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

  std::vector<int> expected_output = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
      0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,
      0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,
      0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,
      0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,
      0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255,
      255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,
      0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255,
      255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,
      0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,
      0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,
      0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
  std::vector<int> output(kWidth * kHeight, 0);
  std::vector<int> size = {kWidth, kHeight};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->inputs_count.push_back(size.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_stl::TestTaskSTL sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}