#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "all/alputov_i_graham_scan/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> PointsToDoublesPerf(const std::vector<alputov_i_graham_scan_all::Point>& points) {
  std::vector<double> doubles;
  doubles.reserve(points.size() * 2);
  for (const auto& p : points) {
    doubles.push_back(p.x);
    doubles.push_back(p.y);
  }
  return doubles;
}

constexpr int kPerfPointCount = 100000;

std::vector<alputov_i_graham_scan_all::Point> GeneratePerfData(int count) {
  std::vector<alputov_i_graham_scan_all::Point> points;
  points.reserve(count + 4);
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-1000.0, 1000.0);
  for (int i = 0; i < count; ++i) {
    points.emplace_back(dis(gen), dis(gen));
  }
  points.emplace_back(-1001.0, -1001.0);
  points.emplace_back(1001.0, -1001.0);
  points.emplace_back(1001.0, 1001.0);
  points.emplace_back(-1001.0, 1001.0);
  return points;
}

bool VerifyHullBasic(const std::vector<alputov_i_graham_scan_all::Point>& original_points_struct,
                     const std::vector<double>& hull_doubles, int hull_size) {
  if (hull_size < 3 && !original_points_struct.empty()) {
    if (original_points_struct.size() < 3) {
      return hull_size == static_cast<int>(original_points_struct.size());
    }
    return false;
  }
  if (hull_size == 0 && original_points_struct.empty()) {
    return true;
  }

  std::set<std::pair<double, double>> original_points_set;
  for (const auto& p : original_points_struct) {
    original_points_set.emplace(p.x, p.y);
  }

  for (int i = 0; i < hull_size; ++i) {
    if (original_points_set.find({hull_doubles[2 * i], hull_doubles[(2 * i) + 1]}) == original_points_set.end()) {
      return false;
    }
  }
  return true;
}

struct TaskALLDeleter {
  void operator()(alputov_i_graham_scan_all::TestTaskALL* p) const {
    if (p != nullptr) {
      p->CleanupMPIResources();
      delete p;
    }
  }
};

}  // namespace

TEST(alputov_i_graham_scan_all, test_pipeline_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_struct = GeneratePerfData(kPerfPointCount);
  std::vector<double> input_doubles = PointsToDoublesPerf(input_points_struct);

  int actual_hull_size = 0;
  std::vector<double> output_hull_doubles(input_doubles.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_hull_size));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task_obj(
      new alputov_i_graham_scan_all::TestTaskALL(task_data), TaskALLDeleter());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_obj);

  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(VerifyHullBasic(input_points_struct, output_hull_doubles, actual_hull_size));
    std::set<std::pair<double, double>> hull_set;
    for (int i = 0; i < actual_hull_size; ++i) {
      hull_set.insert({output_hull_doubles[2 * i], output_hull_doubles[(2 * i) + 1]});
    }
    ASSERT_TRUE(hull_set.count({-1001.0, -1001.0}));
    ASSERT_TRUE(hull_set.count({1001.0, 1001.0}));
  }
}

TEST(alputov_i_graham_scan_all, test_task_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_struct = GeneratePerfData(kPerfPointCount);
  std::vector<double> input_doubles = PointsToDoublesPerf(input_points_struct);

  int actual_hull_size = 0;
  std::vector<double> output_hull_doubles(input_doubles.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_hull_size));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task_obj(
      new alputov_i_graham_scan_all::TestTaskALL(task_data), TaskALLDeleter());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_obj);

  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(VerifyHullBasic(input_points_struct, output_hull_doubles, actual_hull_size));
    std::set<std::pair<double, double>> hull_set;
    for (int i = 0; i < actual_hull_size; ++i) {
      hull_set.insert({output_hull_doubles[2 * i], output_hull_doubles[(2 * i) + 1]});
    }
    ASSERT_TRUE(hull_set.count({-1001.0, -1001.0}));
    ASSERT_TRUE(hull_set.count({1001.0, 1001.0}));
  }
}