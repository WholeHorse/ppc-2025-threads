#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/sidorina_p_gradient_method/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(sidorina_p_gradient_method_all, test_pipeline_run) {
  int size = static_cast<int>(std::pow(2, 12));
  std::vector<double> a(size * size);
  std::vector<double> a0(size * size);
  std::vector<double> b(size, 0);
  std::vector<double> solution(size, 0);
  std::vector<double> expected(size, 0);
  double tolerance = 1e-6;
  std::vector<double> result(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0F, 1000.0F);

  for (int i = 0; i < size; i++) {
    a0[i] = dist(gen);
    for (int j = i; j < size; j++) {
      double value = dist(gen);
      a[(i * size) + j] = value;
      a[(j * size) + i] = value;
    }
  }

  for (int i = 0; i < size; i++) {
    a[(i * size) + i] += static_cast<float>(size) * 10.0F;
  }

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task->outputs_count.emplace_back(result.size());

  auto gradient_method = std::make_shared<sidorina_p_gradient_method_all::GradientMethod>(task);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gradient_method);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < size; ++j) {
      sum += a[(i * size) + j] * solution[j];
    }
    EXPECT_NEAR(sum, b[i], tolerance);
  }
}

TEST(sidorina_p_gradient_method_all, test_task_run) {
  int size = static_cast<int>(std::pow(2, 12));
  std::vector<double> a(size * size);
  std::vector<double> a0(size * size);
  std::vector<double> b(size, 0);
  std::vector<double> solution(size, 0);
  std::vector<double> expected(size, 0);
  double tolerance = 1e-6;
  std::vector<double> result(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0F, 1000.0F);

  for (int i = 0; i < size; i++) {
    a0[i] = dist(gen);
    for (int j = i; j < size; j++) {
      double value = dist(gen);
      a[(i * size) + j] = value;
      a[(j * size) + i] = value;
    }
  }

  for (int i = 0; i < size; i++) {
    a[(i * size) + i] += static_cast<float>(size) * 10.0F;
  }

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task->outputs_count.emplace_back(result.size());

  auto gradient_method = std::make_shared<sidorina_p_gradient_method_all::GradientMethod>(task);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gradient_method);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int i = 0; i < size; i++) {
    double sum = 0.0;
    for (int j = 0; j < size; j++) {
      sum += a[(i * size) + j] * solution[j];
    }
    EXPECT_NEAR(sum, b[i], tolerance);
  }
}