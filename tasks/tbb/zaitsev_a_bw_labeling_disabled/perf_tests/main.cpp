#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>

#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <set>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/zaitsev_a_bw_labeling/include/ops_tbb.hpp"

#ifndef _WIN32
namespace {
void GenerateImage(std::vector<std::uint8_t>& in, std::vector<std::uint16_t>& exp, std::uint16_t width,
                   std::uint16_t height) {
  cv::Mat img;
  img.create(height, width, CV_8UC1);
  cv::randu(img, 0, 2);

  in = std::vector<uint8_t>(img.reshape(1, static_cast<int>(img.total()) * img.channels()));

  cv::Mat labels(height, width, CV_16U);
  cv::connectedComponents(img, labels, 8, CV_16U);
  exp = std::vector<uint16_t>(labels.reshape(1, static_cast<int>(labels.total()) * labels.channels()));
  img.release();
  labels.release();
}

bool IsIsomorphic(const std::vector<std::uint16_t>& first, std::vector<std::uint16_t>& second) {
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
}  // namespace
#endif

TEST(zaitsev_a_labeling_tbb, test_pipeline_run) {
#ifndef _WIN32
  const int width = 1000;
  const int height = 1000;
  std::vector<std::uint8_t> in(width * height);
  std::vector<std::uint16_t> out(width * height);
  std::vector<std::uint16_t> exp(width * height);
  GenerateImage(in, exp, width, height);

  // Create task_data

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<std::uint8_t*>(out.data()));

  // Create Task
  auto test_task_sequential = std::make_shared<zaitsev_a_labeling_tbb::Labeler>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_TRUE(IsIsomorphic(exp, out));
#endif
  EXPECT_TRUE(true);
}

TEST(zaitsev_a_labeling_tbb, test_task_run) {
#ifndef _WIN32
  const int width = 1000;
  const int height = 1000;
  std::vector<std::uint8_t> in(width * height);
  std::vector<std::uint16_t> out(width * height);
  std::vector<std::uint16_t> exp(width * height);
  GenerateImage(in, exp, width, height);

  // Create task_data

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<std::uint8_t*>(out.data()));

  // Create Task
  auto test_task_sequential = std::make_shared<zaitsev_a_labeling_tbb::Labeler>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_TRUE(IsIsomorphic(exp, out));
#endif
  EXPECT_TRUE(true);
}