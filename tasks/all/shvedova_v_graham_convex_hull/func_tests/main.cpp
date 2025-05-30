#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> BuildTaskData(const std::vector<double>& src, std::vector<double>& dst,
                                                   int& hull_count) {
  auto data = std::make_shared<ppc::core::TaskData>();

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(src.data())));
    data->inputs_count.emplace_back(src.size());

    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_count));
    data->outputs_count.emplace_back(1);
    data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
    data->outputs_count.emplace_back(dst.size());
  }

  return data;
}

template <typename Task = shvedova_v_graham_convex_hull_all::GrahamConvexHullALL>
void ExecuteTask(const std::shared_ptr<ppc::core::TaskData>& data) {
  Task task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

std::vector<double> GenSrc(int count) {
  std::vector<double> points;
  points.reserve(count * 2);
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(-2.55, 4.05);
  for (int i = 0; i < count; ++i) {
    points.push_back(dis(gen));
    points.push_back(dis(gen));
  }
  return points;
}
}  // namespace

TEST(shvedova_v_graham_convex_hull_all, convex_triangle) {
  std::vector<double> src = {0.0, 0.0, 2.0, 2.0, 2.0, 0.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {0, 0, 2, 0, 2, 2};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_square_inner) {
  std::vector<double> src = {0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 1.0, 1.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    // Ожидается, что внутренняя точка исключена, поэтому оболочка состоит из 4 точек
    std::vector<double> exp = {0, 0, 2, 0, 2, 2, 0, 2};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_identical_points) {
  std::vector<double> src = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_FALSE(task.Validation());  // Коллинеарность
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_collinear_diagonal) {
  std::vector<double> src = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0};
  int hull_count = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = BuildTaskData(src, dst, hull_count);
  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_collinear_vertical) {
  std::vector<double> src = {0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_collinear_horizontal) {
  std::vector<double> src = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_invalid_too_few) {
  std::vector<double> src = {0.0, 0.0, 1.0, 1.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_FALSE(task.ValidationImpl());
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_invalid_odd_coords) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return;
  }

  std::vector<double> src = {0.0, 0.0, 1.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  shvedova_v_graham_convex_hull_all::GrahamConvexHullALL task(data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(shvedova_v_graham_convex_hull_all, convex_rhomb) {
  std::vector<double> src = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_square) {
  std::vector<double> src = {2.0, 2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {-2, -2, 2, -2, 2, 2, -2, 2};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_rhomb_inner) {
  std::vector<double> src = {0.3,  -0.25, 1.0,   0.0,  2.0,  0.0, 0.3,  0.25,  0.0,   -2.0, 0.0,  -1.0,
                             0.25, -0.3,  -0.25, -0.3, 0.0,  1.0, 0.0,  2.0,   -0.25, 0.3,  0.25, 0.3,
                             -0.3, 0.25,  -1.0,  0.0,  -2.0, 0.0, -0.3, -0.25, 0.1,   0.1};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {0, -2, 2, 0, 0, 2, -2, 0};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_square_inner_complex) {
  std::vector<double> src = {-2.0, -2.0, -1.0, -1.0, -0.5, -1.0, -1.0, -0.5, 2.0, -2.0, 0.5, -1.0,
                             1.0,  -1.0, 1.0,  -0.5, 2.0,  2.0,  1.0,  1.0,  0.5, 1.0,  1.0, 0.5,
                             -2.0, 2.0,  -0.5, 1.0,  -1.0, 1.0,  -1.0, 0.5,  0.1, 0.1};
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {-2, -2, 2, -2, 2, 2, -2, 2};
    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_random) {
  constexpr int kCount = 100;
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<> dist(-100.0, 100.0);
  std::vector<double> src;
  src.reserve(kCount * 2);
  for (int i = 0; i < kCount; ++i) {
    src.push_back(dist(rng));
    src.push_back(dist(rng));
  }
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;
  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<std::pair<double, double>> conv_hull;
    conv_hull.reserve(hull_count);
    for (int i = 0; i < hull_count; ++i) {
      conv_hull.emplace_back(dst[2 * i], dst[(2 * i) + 1]);
    }
    for (int i = 0; i < hull_count; ++i) {
      double dx = conv_hull[(i + 1) % hull_count].first - conv_hull[i].first;
      double dy = conv_hull[(i + 1) % hull_count].second - conv_hull[i].second;
      double dx2 = conv_hull[(i + 2) % hull_count].first - conv_hull[(i + 1) % hull_count].first;
      double dy2 = conv_hull[(i + 2) % hull_count].second - conv_hull[(i + 1) % hull_count].second;
      double cross = (dx * dy2) - (dy * dx2);
      EXPECT_GT(cross, 0.0);
    }
  }
}

TEST(shvedova_v_graham_convex_hull_all, PentagonWithPointsInside) {
  std::vector<double> p = {1.5, 3.5, 2.0, 5.0, 2.2, 3.2, 4.1, 4.2, 2.0, 2.4, 5.0, 2.0, 3.7, 3.2, 4.3,
                           5.0, 5.0, 5.0, 5.0, 2.9, 3.5, 4.8, 2.7, 2.5, 3.0, 4.6, 1.0, 4.0, 2.0, 2.0};

  std::vector<double> src = p;
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);

  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {
        p[10], p[11], p[16], p[17], p[2], p[3], p[26], p[27], p[28], p[29],
    };

    EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
  }
}

TEST(shvedova_v_graham_convex_hull_all, random_48) {
  std::vector<double> src = GenSrc(48);
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);
  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> seq_dst(src.size(), 0.0);
    int seq_hull_count = 0;
    auto seq_data = BuildTaskData(src, seq_dst, seq_hull_count);
    ExecuteTask<shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential>(seq_data);

    EXPECT_EQ(dst, seq_dst);
  }
}

TEST(shvedova_v_graham_convex_hull_all, random_64) {
  std::vector<double> src = GenSrc(64);
  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);
  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> seq_dst(src.size(), 0.0);
    int seq_hull_count = 0;
    auto seq_data = BuildTaskData(src, seq_dst, seq_hull_count);
    ExecuteTask<shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential>(seq_data);

    EXPECT_EQ(dst, seq_dst);
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_circle) {
  constexpr int kNumPoints = 20;
  constexpr double kRadius = 5.0;

  std::vector<double> src;
  src.reserve(kNumPoints * 2);

  for (int i = 0; i < kNumPoints; ++i) {
    double angle = (2 * M_PI * i) / kNumPoints;
    src.push_back(kRadius * std::cos(angle));
    src.push_back(kRadius * std::sin(angle));
  }

  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);
  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_EQ(hull_count, kNumPoints);
    for (int i = 0; i < hull_count; ++i) {
      double x = dst[2 * i];
      double y = dst[(2 * i) + 1];
      double dist_sq = (x * x) + (y * y);
      EXPECT_NEAR(dist_sq, kRadius * kRadius, 1e-6);
    }
  }
}

TEST(shvedova_v_graham_convex_hull_all, convex_star) {
  std::vector<double> src = {
      0.0, 3.0, -2.0, 1.0, -3.0, -2.0, 0.0, -1.0, 3.0, -2.0, 2.0, 1.0,
  };

  std::vector<double> dst(src.size(), 0.0);
  int hull_count = 0;

  auto data = BuildTaskData(src, dst, hull_count);
  ExecuteTask(data);

  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> exp = {
        -3.0, -2.0, 3.0, -2.0, 2.0, 1.0, 0.0, 3.0, -2.0, 1.0,
    };

    EXPECT_EQ(hull_count, static_cast<int>(exp.size() / 2));
    std::vector<double> result(dst.begin(), dst.begin() + static_cast<int>(exp.size()));
    EXPECT_EQ(result, exp);
  }
}
