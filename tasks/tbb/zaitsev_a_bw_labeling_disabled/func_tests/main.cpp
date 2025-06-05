#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#ifndef _WIN32
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc.hpp"
#endif
#include "tbb/zaitsev_a_bw_labeling/include/ops_tbb.hpp"

using TestData = std::tuple<std::vector<std::uint8_t>, std::vector<std::uint16_t>, unsigned int, unsigned int>;

namespace {

// NOLINTNEXTLINE(readability-identifier-naming)
class zaitsev_a_labeling_test_tbb : public ::testing::TestWithParam<std::string> {
 protected:
#ifndef _WIN32
  static TestData GenerateImage() {
    int max_size = 500;
    int min_size = 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(min_size, max_size);

    int width = distr(gen);
    int height = distr(gen);

    cv::Mat img(height, width, CV_8UC1);
    cv::randu(img, 0, 2);

    std::vector<std::uint8_t> in(img.reshape(1, static_cast<int>(img.total()) * img.channels()));

    cv::Mat labels;
    cv::connectedComponents(img, labels, 8, CV_16U);

    std::vector<std::uint16_t> exp(labels.reshape(1, static_cast<int>(labels.total()) * labels.channels()));

    img.release();
    labels.release();

    return std::make_tuple(in, exp, width, height);
  }

  static TestData PrepareImages(const std::string& filename) {
    cv::Mat img_raw = cv::imread(ppc::util::GetAbsolutePath("tbb/zaitsev_a_bw_labeling/data/" + filename));
    unsigned int width = img_raw.cols;
    unsigned int height = img_raw.rows;

    cv::Mat img_gray;
    cv::cvtColor(img_raw, img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat img;
    cv::threshold(img_gray, img, 128, 1, cv::THRESH_BINARY_INV);

    std::vector<std::uint8_t> in(img.reshape(1, static_cast<int>(img.total()) * img.channels()));

    cv::Mat labels;
    cv::connectedComponents(img, labels, 8, CV_16U);

    std::vector<std::uint16_t> out(labels.reshape(1, static_cast<int>(labels.total()) * labels.channels()));

    return std::make_tuple(in, out, width, height);
  }

  static bool IsIsomorphic(const std::vector<std::uint16_t>& first, std::vector<std::uint16_t>& second) {
    std::map<std::uint16_t, std::uint16_t> concordance;
    std::set<std::uint16_t> already_been;

    if (first.size() != second.size()) {
      return false;
    }

    for (uint32_t i = 0; i < first.size(); i++) {
      if (first[i] != second[i]) {
        if (!concordance.contains(first[i]) && !already_been.contains(second[i])) {
          concordance[first[i]] = second[i];
          already_been.insert(second[i]);
        } else if (concordance.contains(first[i]) && already_been.contains(second[i])) {
          if (second[i] != concordance[first[i]]) {
            return false;
          }
        } else {
          return false;
        }
      }
    }
    return true;
  }
#endif
};

TEST_F(zaitsev_a_labeling_test_tbb, validation_fails_on_incorrect_input) {
  const int width = 16;
  std::vector<std::uint8_t> in(width, 0);
  std::vector<uint16_t> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<std::uint8_t*>(out.data()));

  zaitsev_a_labeling_tbb::Labeler intask(task_data_seq);
  EXPECT_FALSE(intask.ValidationImpl());
}
#ifndef _WIN32
TEST_P(zaitsev_a_labeling_test_tbb, returns_correct_label_map) {
  const auto testcase = GetParam();
  const auto& [in, exp, width, height] = (testcase == "rand") ? GenerateImage() : PrepareImages(testcase);

  std::vector<std::uint16_t> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<std::uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<std::uint8_t*>(out.data()));

  zaitsev_a_labeling_tbb::Labeler task(task_data_seq);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_TRUE(IsIsomorphic(exp, out));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(zaitsev_a_labeling_test_tbb, zaitsev_a_labeling_test_tbb, ::testing::Values(
    "help.jpg",
    "kittens.jpg",
    "rombs.jpg",
    "silly.jpg",
    "trig.jpg",
    "rand",
    "rand",
    "rand",
    "rand",
    "rand"
  )
);
// clang-format on

#endif
}  // namespace