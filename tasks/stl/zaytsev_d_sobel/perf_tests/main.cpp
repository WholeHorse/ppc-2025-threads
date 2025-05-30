#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/zaytsev_d_sobel/include/ops_stl.hpp"

namespace {
std::vector<int> GenerateRandomImage(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 255);

  std::vector<int> image(size);
  for (size_t i = 0; i < size; i++) {
    image[i] = dis(gen);
  }
  return image;
}

std::vector<int> SSobel(const std::vector<int> &input, int width, int height) {
  std::vector<int> output(width * height, 0);

  int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int sum_x = 0;
      int sum_y = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int pixel = input[((y + ky) * width) + (x + kx)];
          sum_x += gx[ky + 1][kx + 1] * pixel;
          sum_y += gy[ky + 1][kx + 1] * pixel;
        }
      }

      int magnitude = static_cast<int>(std::sqrt((sum_x * sum_x) + (sum_y * sum_y)));
      output[(y * width) + x] = std::min(magnitude, 255);
    }
  }

  return output;
}

}  // namespace

TEST(zaytsev_d_sobel_stl, test_pipeline_run) {
  constexpr int kSize = 4500;

  std::vector<int> in = GenerateRandomImage(kSize * kSize);
  std::vector<int> out(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.push_back(size.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<zaytsev_d_sobel_stl::TestTaskSTL>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected = SSobel(in, kSize, kSize);
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(zaytsev_d_sobel_stl, test_task_run) {
  constexpr int kSize = 4500;

  std::vector<int> in = GenerateRandomImage(kSize * kSize);
  std::vector<int> out(kSize * kSize, 0);
  std::vector<int> size = {kSize, kSize};

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(size.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(size.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<zaytsev_d_sobel_stl::TestTaskSTL>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected = SSobel(in, kSize, kSize);
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}
