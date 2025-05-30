#pragma once

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_graham_scan_all {

struct Point {
  double x, y;
  Point(double x_in = 0, double y_in = 0) : x(x_in), y(y_in) {}
  bool operator<(const Point& other) const {
    if (y != other.y) {
      return y < other.y;
    }
    return x < other.x;
  }
  bool operator==(const Point& other) const {
    constexpr double kEpsilon = 1e-9;
    return std::abs(x - other.x) < kEpsilon && std::abs(y - other.y) < kEpsilon;
  }
  bool operator!=(const Point& other) const { return !(*this == other); }
};

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void CleanupMPIResources();

 private:
  int rank_{};
  int world_size_{};
  MPI_Datatype mpi_point_datatype_{MPI_DATATYPE_NULL};
  MPI_Comm active_comm_ = MPI_COMM_NULL;
  int active_procs_count_ = 0;
  bool mpi_resources_active_ = false;

  std::vector<Point> input_points_;
  std::vector<Point> convex_hull_;
  Point pivot_;
  std::vector<Point> local_points_;
  std::vector<Point> globally_sorted_points_;

  bool InitializeRun(size_t& current_total_num_points_ref, int& current_rank_in_active_comm_out);
  size_t DistributePointsAndBroadcastPivot(int current_rank_in_active_comm);
  int SortLocalAndGatherSortedPoints(int current_rank_in_active_comm);
  void ConstructFinalHullOnRoot(int current_rank_in_active_comm, int total_sorted_points_count);

  static Point FindPivot(const std::vector<Point>& points);
  static bool CompareAngles(const Point& p1, const Point& p2, const Point& pivot);
  static double CrossProduct(const Point& o, const Point& a, const Point& b);
  static void RemoveDuplicates(std::vector<Point>& points);
  static std::vector<Point> BuildHull(const std::vector<Point>& sorted_points, const Point& pivot);
  static void LocalParallelSort(std::vector<Point>& points, const Point& pivot_for_sort);
};

}  // namespace alputov_i_graham_scan_all