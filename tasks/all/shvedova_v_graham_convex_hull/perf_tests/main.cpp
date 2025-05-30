#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "../include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi.h"

constexpr int kCount = 4500000;

namespace {
std::vector<double> GenSrc(int count) {
  std::vector<double> points;
  points.reserve(count * 2);
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(-4.0, 4.0);
  for (int i = 0; i < count; ++i) {
    points.push_back(dis(gen));
    points.push_back(dis(gen));
  }
  return points;
}

bool VerifyHullBasic(const std::vector<double>& points, const std::vector<double>& hull, int hull_size) {
  if (hull_size < 3) {
    return false;
  }

  std::set<std::pair<double, double>> original_points;
  for (size_t i = 0; i < points.size(); i += 2) {
    original_points.emplace(points[i], points[i + 1]);
  }

  for (int i = 0; i < hull_size * 2; i += 2) {
    if (!original_points.contains({hull[i], hull[i + 1]})) {
      return false;
    }
  }

  return true;
}
}  // namespace

TEST(shvedova_v_graham_convex_hull_all, test_pipeline_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> points = GenSrc(kCount);
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
    task_data_all->inputs_count.emplace_back(points.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
    task_data_all->outputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    task_data_all->outputs_count.emplace_back(hull.size());
  }
  auto test_task_alluential = std::make_shared<shvedova_v_graham_convex_hull_all::GrahamConvexHullALL>(task_data_all);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(VerifyHullBasic(points, hull, scan_size));
  }
}

TEST(shvedova_v_graham_convex_hull_all, test_task_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> points = GenSrc(kCount);
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
    task_data_all->inputs_count.emplace_back(points.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
    task_data_all->outputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
    task_data_all->outputs_count.emplace_back(hull.size());
  }
  auto test_task_alluential = std::make_shared<shvedova_v_graham_convex_hull_all::GrahamConvexHullALL>(task_data_all);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(VerifyHullBasic(points, hull, scan_size));
  }
}
