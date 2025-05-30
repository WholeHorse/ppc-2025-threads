#include "../include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
bool CheckCollinearity(std::span<double> raw_points) {
  const auto points_count = raw_points.size() / 2;
  if (points_count < 3) {
    return true;
  }
  const auto dx = raw_points[2] - raw_points[0];
  const auto dy = raw_points[3] - raw_points[1];
  for (size_t i = 2; i < points_count; i++) {
    const auto dx_i = raw_points[(i * 2)] - raw_points[0];
    const auto dy_i = raw_points[(i * 2) + 1] - raw_points[1];
    if (std::fabs((dx * dy_i) - (dy * dx_i)) > 1e-9) {
      return false;
    }
  }
  return true;
}
bool ComparePoints(const Point &p0, const Point &p1, const Point &p2) {
  struct XYPoint {
    double x, y;
  };
  const XYPoint d1{.x = p1[0] - p0[0], .y = p1[1] - p0[1]};
  const XYPoint d2{.x = p2[0] - p0[0], .y = p2[1] - p0[1]};
  const auto cross = (d1.x * d2.y) - (d1.y * d2.x);
  if (std::abs(cross) < 1e-9) {
    return (d1.x * d1.x + d1.y * d1.y) < (d2.x * d2.x + d2.y * d2.y);
  }
  return cross > 0;
}
double CrossProduct(const Point &p0, const Point &p1, const Point &p2) {
  return ((p1[0] - p0[0]) * (p2[1] - p0[1])) - ((p1[1] - p0[1]) * (p2[0] - p0[0]));
}
}  // namespace

namespace shvedova_v_graham_convex_hull_all {

bool GrahamConvexHullALL::ValidationImpl() {
  return (rank_ != 0) ||
         ((task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 2 &&
           task_data->outputs_count.size() == 2 && (task_data->inputs_count[0] % 2 == 0) &&
           (task_data->inputs_count[0] / 2 > 2) && (task_data->outputs_count[0] == 1) &&
           (task_data->outputs_count[1] >= task_data->inputs_count[0])) &&
          !CheckCollinearity({reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]}));
}

bool GrahamConvexHullALL::PreProcessingImpl() {
  if (rank_ != 0) {
    return true;
  }

  points_count_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(points_count_, Point{});

  auto *p_src = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < points_count_ * 2; i += 2) {
    input_[i / 2][0] = p_src[i];
    input_[i / 2][1] = p_src[i + 1];
  }

  res_.clear();
  res_.reserve(points_count_);

  return true;
}

void GrahamConvexHullALL::PerformSort(const Point &pivot) {
  const int threadsnum = std::min(static_cast<int>(procinput_.size()), ppc::util::GetPPCNumThreads());
  const int perthread = static_cast<int>(procinput_.size()) / threadsnum;
  const int unfit = static_cast<int>(procinput_.size()) % threadsnum;

  std::vector<std::span<Point>> fragments(threadsnum);
  auto it = procinput_.begin();
  for (int i = 0; i < threadsnum; i++) {
    auto nit = std::next(it, perthread + ((i < unfit) ? 1 : 0));
    fragments[i] = std::span{it, nit};
    it = nit;
  }

  const auto comp = [&](const Point &p1, const Point &p2) { return ComparePoints(pivot, p1, p2); };
  const auto merge = [&](int i, int k) {
    auto &primary = fragments[k];
    auto &secondary = fragments[k + i];

    std::inplace_merge(primary.begin(), primary.end(), secondary.end(), comp);
    primary = std::span{primary.begin(), secondary.end()};
  };

  std::vector<std::thread> ts(threadsnum);
  for (std::size_t i = 0; i < ts.size(); i++) {
    ts[i] = std::thread([&, i] { std::ranges::sort(fragments[i], comp); });
  }
  std::ranges::for_each(ts, [](auto &t) { t.join(); });

  for (int i = 1; i < threadsnum; i *= 2) {
    const auto factor = threadsnum - i;
    if (fragments.front().size() < 32) {
      for (int j = 0; j < factor; j += 2 * i) {
        merge(i, j);
      }
    } else {
      ts.clear();
      for (int j = 0; j < factor; j += 2 * i) {
        ts.emplace_back([&](int k) { merge(i, k); }, j);
      }
      std::ranges::for_each(ts, [](auto &t) { t.join(); });
    }
  }
}

bool GrahamConvexHullALL::RunImpl() {
  auto size = input_.size();
  MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  int worldsize{};
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  int processes = std::min(static_cast<int>(size), worldsize);

  MPI_Comm group{};
  if (rank_ >= processes) {
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank_, &group);
    MPI_Comm_free(&group);
    return true;
  }
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank_, &group);

  Point pivot{};
  if (rank_ == 0) {
    pivot = *std::ranges::min_element(input_, [](auto &a, auto &b) { return a[1] < b[1]; });
  }
  MPI_Bcast(pivot.data(), int(pivot.size()), MPI_DOUBLE, 0, group);

  std::vector<int> sendcnts(processes);
  std::vector<int> displacements(processes);
  const int perprocess = static_cast<int>(size) / processes;
  const int unfit = static_cast<int>(size) % processes;

  for (int i = 0; i < processes; ++i) {
    sendcnts[i] = (perprocess + (i < unfit ? 1 : 0)) * int(Point{}.size());
  }
  std::partial_sum(sendcnts.begin(), sendcnts.end() - 1, displacements.begin() + 1);
  displacements[0] = 0;

  int local_count = sendcnts[rank_] / int(Point{}.size());
  procinput_.reserve(size);
  procinput_.resize(local_count);

  MPI_Scatterv(input_.data(), sendcnts.data(), displacements.data(), MPI_DOUBLE, procinput_.data(), sendcnts[rank_],
               MPI_DOUBLE, 0, group);

  PerformSort(pivot);

  const auto comp = [&](const Point &p1, const Point &p2) { return ComparePoints(pivot, p1, p2); };
  for (int i = 1; i < processes; i *= 2) {
    if (rank_ % (2 * i) == 0) {
      const int secondary = rank_ + i;
      if (secondary < processes) {
        int32_t sizecomp{};
        MPI_Recv(&sizecomp, 1, MPI_INT32_T, secondary, 0, group, MPI_STATUS_IGNORE);

        const auto div = procinput_.size();
        procinput_.resize(div + sizecomp);
        MPI_Recv(procinput_.data() + div, sizecomp * int(Point{}.size()), MPI_DOUBLE, secondary, 0, group,
                 MPI_STATUS_IGNORE);
        std::ranges::inplace_merge(procinput_, procinput_.begin() + static_cast<std::int64_t>(div), comp);
      }
    } else if ((rank_ % i) == 0) {
      const auto sizeproc = std::int32_t(procinput_.size());
      MPI_Send(&sizeproc, 1, MPI_INT32_T, rank_ - i, 0, group);
      MPI_Send(procinput_.data(), sizeproc * int(Point{}.size()), MPI_DOUBLE, rank_ - i, 0, group);
      break;
    }
  }

  MPI_Comm_free(&group);

  if (rank_ == 0) {
    res_.push_back(procinput_[0]);
    res_.push_back(procinput_[1]);
    for (int i = 2; i < points_count_; ++i) {
      while (res_.size() > 1 && CrossProduct(res_[res_.size() - 2], res_.back(), procinput_[i]) <= 0) {
        res_.pop_back();
      }
      res_.push_back(procinput_[i]);
    }
  }
  return true;
}

bool GrahamConvexHullALL::PostProcessingImpl() {
  if (rank_ == 0) {
    int res_points_count = static_cast<int>(res_.size());
    *reinterpret_cast<int *>(task_data->outputs[0]) = res_points_count;
    auto *p_out = reinterpret_cast<double *>(task_data->outputs[1]);
    for (int i = 0; i < res_points_count; i++) {
      p_out[2 * i] = res_[i][0];
      p_out[(2 * i) + 1] = res_[i][1];
    }
  }
  return true;
}

}  // namespace shvedova_v_graham_convex_hull_all

namespace shvedova_v_graham_convex_hull_seq {

bool GrahamConvexHullSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 2 &&
          task_data->outputs_count.size() == 2 && (task_data->inputs_count[0] % 2 == 0) &&
          (task_data->inputs_count[0] / 2 > 2) && (task_data->outputs_count[0] == 1) &&
          (task_data->outputs_count[1] >= task_data->inputs_count[0])) &&
         !CheckCollinearity({reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]});
}

bool GrahamConvexHullSequential::PreProcessingImpl() {
  points_count_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(points_count_, Point{});

  auto *p_src = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < points_count_ * 2; i += 2) {
    input_[i / 2][0] = p_src[i];
    input_[i / 2][1] = p_src[i + 1];
  }

  res_.clear();
  res_.reserve(points_count_);

  return true;
}

void GrahamConvexHullSequential::PerformSort() {
  const auto pivot = *std::ranges::min_element(input_, [](auto &a, auto &b) { return a[1] < b[1]; });
  for (int pt = 0; pt < points_count_; pt++) {
    const bool even_step = pt % 2 == 0;
    const int shift = even_step ? 0 : -1;
    const int revshift = even_step ? -1 : 0;
    for (int i = 1; i < points_count_ + shift; i += 2) {
      if (ComparePoints(pivot, input_[i - shift], input_[i + revshift])) {
        std::swap(input_[i], input_[i - (even_step ? 1 : -1)]);
      }
    }
  }
}

bool GrahamConvexHullSequential::RunImpl() {
  PerformSort();

  for (int i = 0; i < 3; i++) {
    res_.push_back(input_[i]);
  }

  for (int i = 3; i < points_count_; ++i) {
    while (res_.size() > 1) {
      const auto &pv = res_.back();
      const auto dx1 = res_.rbegin()[1][0] - pv[0];
      const auto dy1 = res_.rbegin()[1][1] - pv[1];
      const auto dx2 = input_[i][0] - pv[0];
      const auto dy2 = input_[i][1] - pv[1];
      if (dx1 * dy2 < dy1 * dx2) {
        break;
      }
      res_.pop_back();
    }
    res_.push_back(input_[i]);
  }

  return true;
}

bool GrahamConvexHullSequential::PostProcessingImpl() {
  int res_points_count = static_cast<int>(res_.size());
  *reinterpret_cast<int *>(task_data->outputs[0]) = res_points_count;
  auto *p_out = reinterpret_cast<double *>(task_data->outputs[1]);
  for (int i = 0; i < res_points_count; i++) {
    p_out[2 * i] = res_[i][0];
    p_out[(2 * i) + 1] = res_[i][1];
  }
  return true;
}

}  // namespace shvedova_v_graham_convex_hull_seq
