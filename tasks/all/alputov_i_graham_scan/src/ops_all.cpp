#include "all/alputov_i_graham_scan/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace alputov_i_graham_scan_all {

TestTaskALL::TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int blocklengths[2] = {1, 1};
  MPI_Aint displacements[2];
  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
  Point p_dummy;
  MPI_Get_address(&p_dummy.x, &displacements[0]);
  MPI_Get_address(&p_dummy.y, &displacements[1]);
  displacements[1] -= displacements[0];
  displacements[0] = 0;
  MPI_Type_create_struct(2, blocklengths, displacements, types, &mpi_point_datatype_);
  MPI_Type_commit(&mpi_point_datatype_);

  if (mpi_point_datatype_ != MPI_DATATYPE_NULL) {
    mpi_resources_active_ = true;
  }
}

void TestTaskALL::CleanupMPIResources() {
  if (!mpi_resources_active_) {
    return;
  }

  if (mpi_point_datatype_ != MPI_DATATYPE_NULL) {
    MPI_Type_free(&mpi_point_datatype_);
    mpi_point_datatype_ = MPI_DATATYPE_NULL;
  }
  if (active_comm_ != MPI_COMM_NULL && active_comm_ != MPI_COMM_WORLD) {
    MPI_Comm_free(&active_comm_);
    active_comm_ = MPI_COMM_NULL;
  }
  mpi_resources_active_ = false;
}

bool TestTaskALL::ValidationImpl() {
  if (rank_ != 0) {
    return true;
  }

  if (task_data->inputs.empty() || task_data->inputs_count.empty() || task_data->outputs.empty() ||
      task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->inputs_count.size() != 1 || task_data->outputs.size() != 2 ||
      task_data->outputs_count.size() != 2) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  if (task_data->inputs_count[0] % 2 != 0) {
    return false;
  }

  size_t num_points = task_data->inputs_count[0] / 2;
  if (num_points < 3) {
    return false;
  }

  if (task_data->outputs_count[0] != 1) {
    return false;
  }
  if (task_data->outputs_count[1] < num_points * 2) {
    return false;
  }

  return true;
}

bool TestTaskALL::PreProcessingImpl() {
  if (rank_ == 0) {
    size_t num_input_doubles = task_data->inputs_count[0];
    size_t num_points = num_input_doubles / 2;
    input_points_.resize(num_points);
    auto* input_doubles = reinterpret_cast<double*>(task_data->inputs[0]);
    for (size_t i = 0; i < num_points; ++i) {
      input_points_[i] = Point(input_doubles[2 * i], input_doubles[(2 * i) + 1]);
    }
  }
  return true;
}

bool TestTaskALL::InitializeRun(size_t& current_total_num_points_ref, int& current_rank_in_active_comm_out) {
  if (rank_ == 0) {
    current_total_num_points_ref = input_points_.size();
  }
  MPI_Bcast(&current_total_num_points_ref, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (current_total_num_points_ref < 3) {
    if (rank_ == 0) {
      if (current_total_num_points_ref == 0) {
        convex_hull_.clear();
      } else if (current_total_num_points_ref == 1) {
        convex_hull_ = {input_points_[0]};
      } else {
        if (input_points_[0] < input_points_[1]) {
          convex_hull_ = {input_points_[0], input_points_[1]};
        } else {
          convex_hull_ = {input_points_[1], input_points_[0]};
        }
      }
    }
    return false;
  }

  active_procs_count_ = std::min(world_size_, static_cast<int>(current_total_num_points_ref));
  if (active_procs_count_ == 0 && current_total_num_points_ref > 0) {
    active_procs_count_ = 1;
  }

  if (active_comm_ != MPI_COMM_NULL && active_comm_ != MPI_COMM_WORLD) {
    MPI_Comm_free(&active_comm_);
    active_comm_ = MPI_COMM_NULL;
  }

  int color = (rank_ < active_procs_count_ && active_procs_count_ > 0) ? 0 : 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank_, &active_comm_);

  if (color == 1) {
    return false;
  }

  MPI_Comm_rank(active_comm_, &current_rank_in_active_comm_out);
  return true;
}

size_t TestTaskALL::DistributePointsAndBroadcastPivot(int current_rank_in_active_comm) {
  if (current_rank_in_active_comm == 0) {
    pivot_ = FindPivot(input_points_);
  }
  MPI_Bcast(&pivot_, 1, mpi_point_datatype_, 0, active_comm_);

  std::vector<Point> points_to_sort_globally_on_root;
  if (current_rank_in_active_comm == 0) {
    for (const auto& p : input_points_) {
      if (p != pivot_) {
        points_to_sort_globally_on_root.push_back(p);
      }
    }
  }

  size_t num_points_to_scatter = 0;
  if (current_rank_in_active_comm == 0) {
    num_points_to_scatter = points_to_sort_globally_on_root.size();
  }
  MPI_Bcast(&num_points_to_scatter, 1, MPI_UNSIGNED_LONG, 0, active_comm_);

  if (num_points_to_scatter == 0) {
    return 0;
  }

  std::vector<int> send_counts_for_scatterv(active_procs_count_);
  std::vector<int> displs_for_scatterv(active_procs_count_);
  int local_recv_count = 0;

  if (current_rank_in_active_comm == 0) {
    int base_count = static_cast<int>(num_points_to_scatter / static_cast<size_t>(active_procs_count_));
    int remainder = static_cast<int>(num_points_to_scatter % static_cast<size_t>(active_procs_count_));
    displs_for_scatterv[0] = 0;
    for (int i = 0; i < active_procs_count_; ++i) {
      send_counts_for_scatterv[i] = base_count + (i < remainder ? 1 : 0);
      if (i > 0) {
        displs_for_scatterv[i] = displs_for_scatterv[i - 1] + send_counts_for_scatterv[i - 1];
      }
    }
  }

  MPI_Scatter((current_rank_in_active_comm == 0) ? send_counts_for_scatterv.data() : nullptr, 1, MPI_INT,
              &local_recv_count, 1, MPI_INT, 0, active_comm_);

  local_points_.resize(local_recv_count);
  MPI_Scatterv((current_rank_in_active_comm == 0) ? points_to_sort_globally_on_root.data() : nullptr,
               (current_rank_in_active_comm == 0) ? send_counts_for_scatterv.data() : nullptr,
               (current_rank_in_active_comm == 0) ? displs_for_scatterv.data() : nullptr, mpi_point_datatype_,
               local_points_.data(), local_recv_count, mpi_point_datatype_, 0, active_comm_);

  return num_points_to_scatter;
}

int TestTaskALL::SortLocalAndGatherSortedPoints(int current_rank_in_active_comm) {
  if (!local_points_.empty()) {
    LocalParallelSort(local_points_, pivot_);
  }

  std::vector<int> local_sizes_recv_on_root;
  if (current_rank_in_active_comm == 0) {
    local_sizes_recv_on_root.resize(active_procs_count_);
  }
  int my_local_size_after_sort = static_cast<int>(local_points_.size());

  MPI_Gather(&my_local_size_after_sort, 1, MPI_INT,
             (current_rank_in_active_comm == 0) ? local_sizes_recv_on_root.data() : nullptr, 1, MPI_INT, 0,
             active_comm_);

  int total_sorted_points_count = 0;
  std::vector<int> displs_for_gatherv;
  if (current_rank_in_active_comm == 0) {
    displs_for_gatherv.resize(active_procs_count_);

    globally_sorted_points_.clear();
    for (int i = 0; i < active_procs_count_; ++i) {
      total_sorted_points_count += local_sizes_recv_on_root[i];
      if (i > 0) {
        displs_for_gatherv[i] = displs_for_gatherv[i - 1] + local_sizes_recv_on_root[i - 1];
      }
    }
    globally_sorted_points_.resize(total_sorted_points_count);
  }

  MPI_Gatherv(local_points_.data(), my_local_size_after_sort, mpi_point_datatype_,
              (current_rank_in_active_comm == 0) ? globally_sorted_points_.data() : nullptr,
              (current_rank_in_active_comm == 0) ? local_sizes_recv_on_root.data() : nullptr,
              (current_rank_in_active_comm == 0) ? displs_for_gatherv.data() : nullptr, mpi_point_datatype_, 0,
              active_comm_);

  return total_sorted_points_count;
}

void TestTaskALL::ConstructFinalHullOnRoot(int current_rank_in_active_comm, int total_sorted_points_count) {
  if (current_rank_in_active_comm == 0) {
    if (total_sorted_points_count > 0) {
      LocalParallelSort(globally_sorted_points_, pivot_);
      RemoveDuplicates(globally_sorted_points_);
    }

    if (globally_sorted_points_.empty()) {
      convex_hull_ = {pivot_};
    } else {
      convex_hull_ = BuildHull(globally_sorted_points_, pivot_);
    }
  }
}

bool TestTaskALL::RunImpl() {
  size_t current_total_num_points = 0;
  int current_rank_in_active_comm = 0;

  if (!InitializeRun(current_total_num_points, current_rank_in_active_comm)) {
    return true;
  }

  size_t num_points_to_scatter = DistributePointsAndBroadcastPivot(current_rank_in_active_comm);

  if (num_points_to_scatter == 0) {
    if (current_rank_in_active_comm == 0) {
      convex_hull_ = {pivot_};
    }
    return true;
  }

  int total_sorted_points_count = SortLocalAndGatherSortedPoints(current_rank_in_active_comm);
  ConstructFinalHullOnRoot(current_rank_in_active_comm, total_sorted_points_count);

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  if (this->rank_ == 0) {
    int* hull_size_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    *hull_size_ptr = static_cast<int>(convex_hull_.size());

    auto* hull_data_ptr = reinterpret_cast<double*>(task_data->outputs[1]);
    for (size_t i = 0; i < convex_hull_.size(); ++i) {
      hull_data_ptr[2 * i] = convex_hull_[i].x;
      hull_data_ptr[(2 * i) + 1] = convex_hull_[i].y;
    }
  }
  return true;
}

Point TestTaskALL::FindPivot(const std::vector<Point>& points) {
  if (points.empty()) {
    throw std::runtime_error("Cannot find pivot in empty set of points. ");
  }
  return *std::ranges::min_element(points, [](const Point& a, const Point& b) { return a < b; });
}

double TestTaskALL::CrossProduct(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

bool TestTaskALL::CompareAngles(const Point& p1, const Point& p2, const Point& pivot) {
  double cross = CrossProduct(pivot, p1, p2);
  constexpr double kEpsilon = 1e-9;

  if (std::abs(cross) < kEpsilon) {
    double dist1_sq = ((p1.x - pivot.x) * (p1.x - pivot.x)) + ((p1.y - pivot.y) * (p1.y - pivot.y));
    double dist2_sq = ((p2.x - pivot.x) * (p2.x - pivot.x)) + ((p2.y - pivot.y) * (p2.y - pivot.y));
    return dist1_sq < dist2_sq;
  }
  return cross > 0;
}

void TestTaskALL::RemoveDuplicates(std::vector<Point>& points) {
  if (points.empty()) {
    return;
  }
  points.erase(std::ranges::unique(points).begin(), points.end());
}

std::vector<Point> TestTaskALL::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) {
  std::vector<Point> hull;
  if (sorted_points.empty()) {
    hull.push_back(pivot);
    return hull;
  }

  hull.push_back(pivot);
  hull.push_back(sorted_points[0]);

  if (sorted_points.size() == 1) {
    return hull;
  }

  for (size_t i = 1; i < sorted_points.size(); ++i) {
    const Point& p = sorted_points[i];
    while (hull.size() >= 2 && CrossProduct(hull[hull.size() - 2], hull.back(), p) < 1e-9) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  while (hull.size() >= 3 && CrossProduct(hull[hull.size() - 2], hull.back(), hull[0]) < 1e-9) {
    hull.pop_back();
  }
  if (hull.size() >= 3 && CrossProduct(hull.back(), hull[0], hull[1]) < 1e-9) {
    hull.erase(hull.begin());
  }

  return hull;
}

void TestTaskALL::LocalParallelSort(std::vector<Point>& points, const Point& pivot_for_sort) {
  if (points.size() <= 1) {
    return;
  }

  auto comparator = [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot_for_sort); };

  const auto n = points.size();
  const auto num_threads_hint = static_cast<size_t>(ppc::util::GetPPCNumThreads());
  const size_t num_threads =
      std::max(static_cast<size_t>(1),
               std::min({n / 500, num_threads_hint, static_cast<size_t>(std::thread::hardware_concurrency())}));

  if (num_threads <= 1 || n < 500) {
    std::ranges::sort(points, comparator);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<size_t> chunk_offsets(num_threads + 1);

  for (size_t i = 0; i <= num_threads; ++i) {
    chunk_offsets[i] = (i * n) / num_threads;
  }

  using DifferenceType = typename std::vector<Point>::difference_type;

  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&points, comparator, start_offset = chunk_offsets[i], end_offset = chunk_offsets[i + 1]]() {
      if (start_offset < end_offset) {
        auto start_it = std::next(points.begin(), static_cast<DifferenceType>(start_offset));
        auto end_it = std::next(points.begin(), static_cast<DifferenceType>(end_offset));
        std::ranges::sort(std::ranges::subrange(start_it, end_it), comparator);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (size_t i = 1; i < num_threads; ++i) {
    auto first_unsorted_chunk_start = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i]));
    auto first_unsorted_chunk_end = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i + 1]));
    if (points.begin() < first_unsorted_chunk_start && first_unsorted_chunk_start < first_unsorted_chunk_end) {
      std::inplace_merge(points.begin(), first_unsorted_chunk_start, first_unsorted_chunk_end, comparator);
    }
  }
}

}  // namespace alputov_i_graham_scan_all