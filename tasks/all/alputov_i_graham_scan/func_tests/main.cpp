#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <set>
#include <vector>

#include "all/alputov_i_graham_scan/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> PointsToDoubles(const std::vector<alputov_i_graham_scan_all::Point>& points) {
  std::vector<double> doubles;
  doubles.reserve(points.size() * 2);
  for (const auto& p : points) {
    doubles.push_back(p.x);
    doubles.push_back(p.y);
  }
  return doubles;
}

std::vector<alputov_i_graham_scan_all::Point> DoublesToPoints(const std::vector<double>& doubles, int hull_size) {
  std::vector<alputov_i_graham_scan_all::Point> points;
  if (hull_size == 0) {
    return points;
  }
  points.reserve(hull_size);
  for (int i = 0; i < hull_size; ++i) {
    points.emplace_back(doubles[2 * i], doubles[(2 * i) + 1]);
  }
  return points;
}

void GenerateRandomData(std::vector<alputov_i_graham_scan_all::Point>& data, size_t count, unsigned int seed = 42,
                        double range_min = -1000.0, double range_max = 1000.0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(range_min, range_max);

  data.clear();
  data.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    data.emplace_back(dist(gen), dist(gen));
  }
}

std::vector<alputov_i_graham_scan_all::Point> GenerateStarPoints(size_t num_points_star) {
  std::vector<alputov_i_graham_scan_all::Point> input;
  input.reserve(num_points_star * 2);
  for (size_t i = 0; i < num_points_star; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
    input.emplace_back(20.0 * std::cos(angle), 20.0 * std::sin(angle));
    input.emplace_back(5.0 * std::cos(angle + (std::numbers::pi / static_cast<double>(num_points_star))),
                       5.0 * std::sin(angle + (std::numbers::pi / static_cast<double>(num_points_star))));
  }
  return input;
}

std::vector<alputov_i_graham_scan_all::Point> GenerateConvexPolygon(size_t num_vertices, double radius = 100.0,
                                                                    double cx = 0.0, double cy = 0.0) {
  if (num_vertices < 3) {
    return {};
  }
  std::vector<alputov_i_graham_scan_all::Point> polygon;
  polygon.reserve(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_vertices);
    polygon.emplace_back(cx + (radius * std::cos(angle)), cy + (radius * std::sin(angle)));
  }
  return polygon;
}

void ExecuteAndValidateTask(alputov_i_graham_scan_all::TestTaskALL& task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

void AssertPointsEqual(const std::vector<alputov_i_graham_scan_all::Point>& actual,
                       const std::vector<alputov_i_graham_scan_all::Point>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  if (actual.empty() && expected.empty()) {
    return;
  }
  if (actual.empty() || expected.empty()) {
    FAIL() << "One hull is empty while other is not.";
    return;
  }

  std::set<alputov_i_graham_scan_all::Point> actual_set(actual.begin(), actual.end());
  std::set<alputov_i_graham_scan_all::Point> expected_set(expected.begin(), expected.end());
  ASSERT_EQ(actual_set.size(), actual.size()) << "Actual hull has duplicate points.";
  ASSERT_EQ(expected_set.size(), expected.size()) << "Expected hull has duplicate points.";

  ASSERT_EQ(actual_set, expected_set);
}

struct TaskALLDeleter {
  void operator()(alputov_i_graham_scan_all::TestTaskALL* p) const {
    if (p != nullptr) {
      p->CleanupMPIResources();
      delete p;
    }
  }
};

void ValidateHullWithSpecificCorners(int hull_size_actual, const std::vector<alputov_i_graham_scan_all::Point>&,
                                     const std::vector<alputov_i_graham_scan_all::Point>& actual_hull,
                                     const std::vector<alputov_i_graham_scan_all::Point>& specific_corners_to_check) {
  ASSERT_GE(hull_size_actual, static_cast<int>(specific_corners_to_check.size()));
  if (!specific_corners_to_check.empty()) {
    ASSERT_GE(hull_size_actual, 3);
  }

  std::set<alputov_i_graham_scan_all::Point> actual_set(actual_hull.begin(), actual_hull.end());
  for (const auto& corner : specific_corners_to_check) {
    ASSERT_TRUE(actual_set.count(corner)) << "Corner (" << corner.x << "," << corner.y << ") not found in actual hull.";
  }
}
}  // namespace

TEST(alputov_i_graham_scan_all, minimal_triangle_case) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {2, 0}, {1, 2}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 3);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, collinear_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 2);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    std::vector<alputov_i_graham_scan_all::Point> expected_hull_option1 = {{0, 0}, {3, 3}};
    std::vector<alputov_i_graham_scan_all::Point> expected_hull_option2 = {{3, 3}, {0, 0}};

    bool match = (actual_hull.size() == 2 &&
                  ((actual_hull[0] == expected_hull_option1[0] && actual_hull[1] == expected_hull_option1[1]) ||
                   (actual_hull[0] == expected_hull_option2[0] && actual_hull[1] == expected_hull_option2[1])));
    if (!match) {
      AssertPointsEqual(actual_hull, expected_hull_option1);
    } else {
      ASSERT_TRUE(match);
    }
  }
}

TEST(alputov_i_graham_scan_all, perfect_square_case) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, random_100_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 100);
  std::vector<alputov_i_graham_scan_all::Point> corners = {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, duplicate_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_duplicates(10, {2.5, 3.5});
  std::vector<alputov_i_graham_scan_all::Point> expected_square = {{0, 0}, {5, 0}, {0, 5}, {5, 5}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_duplicates;
  input_points.insert(input_points.end(), expected_square.begin(), expected_square.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_square);
  }
}

TEST(alputov_i_graham_scan_all, star_figure) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t num_points_star = 10;
  std::vector<alputov_i_graham_scan_all::Point> input_points = GenerateStarPoints(num_points_star);

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, static_cast<int>(num_points_star));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);

    std::vector<alputov_i_graham_scan_all::Point> expected_outer_points;
    expected_outer_points.reserve(num_points_star);
    for (size_t i = 0; i < num_points_star; ++i) {
      double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
      expected_outer_points.emplace_back(20.0 * std::cos(angle), 20.0 * std::sin(angle));
    }
    AssertPointsEqual(actual_hull, expected_outer_points);
  }
}

TEST(alputov_i_graham_scan_all, zero_points_invalid) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(6);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(0);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  if (rank == 0) {
    ASSERT_FALSE(task->ValidationImpl());
  } else {
    ASSERT_TRUE(task->ValidationImpl());
  }
}

TEST(alputov_i_graham_scan_all, single_point_invalid) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  if (rank == 0) {
    ASSERT_FALSE(task->ValidationImpl());
  } else {
    ASSERT_TRUE(task->ValidationImpl());
  }
}

TEST(alputov_i_graham_scan_all, two_points_invalid) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 1}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  if (rank == 0) {
    ASSERT_FALSE(task->ValidationImpl());
  } else {
    ASSERT_TRUE(task->ValidationImpl());
  }
}

TEST(alputov_i_graham_scan_all, random_2500_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 2500, 123);
  std::vector<alputov_i_graham_scan_all::Point> corners = {{-2000, -2000}, {2000, -2000}, {2000, 2000}, {-2000, 2000}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, all_points_identical_covers_scatter_zero) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  alputov_i_graham_scan_all::Point expected_pivot = {1, 1};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 1);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    std::vector<alputov_i_graham_scan_all::Point> expected_hull = {expected_pivot};
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, two_distinct_points_one_duplicate_of_pivot) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 1}, {0, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 2);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {1, 1}};
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, invalid_output_buffer_size) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 0}, {0, 1}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles((input_points.size() * 2) - 1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  if (rank == 0) {
    ASSERT_FALSE(task->ValidationImpl());
  } else {
    ASSERT_TRUE(task->ValidationImpl());
  }
}

TEST(alputov_i_graham_scan_all, three_points_collinear_horizontal) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 0}, {2, 0}};
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {2, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 2);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, four_points_square_with_interior_point) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, five_points_convex_pentagon) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {3, 0}, {4, 2}, {1, 4}, {-1, 2}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 5);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, random_50_points_with_corners) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 50, 5050);
  std::vector<alputov_i_graham_scan_all::Point> corners = {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, random_200_points_with_corners) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 200, 200200);
  std::vector<alputov_i_graham_scan_all::Point> corners = {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, large_random_1000_points_with_corners) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 1000, 10001000, -5000, 5000);
  std::vector<alputov_i_graham_scan_all::Point> corners = {{-6000, -6000}, {6000, -6000}, {6000, 6000}, {-6000, 6000}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, very_large_random_5000_points_with_corners) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points_random_part;
  GenerateRandomData(input_points_random_part, 5000, 50005000, -10000, 10000);
  std::vector<alputov_i_graham_scan_all::Point> corners = {
      {-12000, -12000}, {12000, -12000}, {12000, 12000}, {-12000, 12000}};

  std::vector<alputov_i_graham_scan_all::Point> input_points = input_points_random_part;
  input_points.insert(input_points.end(), corners.begin(), corners.end());

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    ValidateHullWithSpecificCorners(hull_size_actual, input_points, actual_hull, corners);
  }
}

TEST(alputov_i_graham_scan_all, points_match_active_procs_convex_polygon) {
  int rank{};
  int world_size_val{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_val);

  size_t num_polygon_vertices = static_cast<size_t>(std::max(3, world_size_val));
  std::vector<alputov_i_graham_scan_all::Point> input_points = GenerateConvexPolygon(num_polygon_vertices);
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, static_cast<int>(num_polygon_vertices));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, points_one_more_than_active_procs_convex_polygon) {
  int rank{};
  int world_size_val{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_val);

  size_t num_polygon_vertices = static_cast<size_t>(std::max(3, world_size_val + 1));
  std::vector<alputov_i_graham_scan_all::Point> input_points = GenerateConvexPolygon(num_polygon_vertices, 50.0);
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, static_cast<int>(num_polygon_vertices));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, points_forming_circle_hull) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t num_circle_points = 12;
  std::vector<alputov_i_graham_scan_all::Point> input_points = GenerateConvexPolygon(num_circle_points, 10.0);
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, static_cast<int>(num_circle_points));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, points_thin_rectangle_hull) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {100, 0}, {100, 0.1}, {0, 0.1}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);
  }
}

TEST(alputov_i_graham_scan_all, collinear_points_vertical) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {0, 1}, {0, 2}, {0, 3}};
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {0, 3}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 2);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, triangle_with_many_duplicates) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points;
  input_points.reserve(15);
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {2, 0}, {1, 2}};
  for (int i = 0; i < 5; ++i) {
    input_points.emplace_back(0, 0);
  }
  for (int i = 0; i < 5; ++i) {
    input_points.emplace_back(2, 0);
  }
  for (int i = 0; i < 5; ++i) {
    input_points.emplace_back(1, 2);
  }

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 3);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, collinear_duplicates_on_edge_plus_one_off_line) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 1}, {1, 1}, {2, 2},
                                                                {3, 3}, {3, 3}, {1, -1}};
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {3, 3}, {1, -1}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 3);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}

TEST(alputov_i_graham_scan_all, many_points_on_line_segment_plus_one_peak) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points;
  input_points.reserve(12);
  for (int i = 0; i <= 10; ++i) {
    input_points.emplace_back(static_cast<double>(i) * 0.1, 0.0);
  }
  input_points.emplace_back(0.5, 1.0);
  std::vector<alputov_i_graham_scan_all::Point> expected_hull = {{0, 0}, {1, 0}, {0.5, 1.0}};

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }
  std::shared_ptr<alputov_i_graham_scan_all::TestTaskALL> task(new alputov_i_graham_scan_all::TestTaskALL(task_data),
                                                               TaskALLDeleter());
  ExecuteAndValidateTask(*task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 3);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, expected_hull);
  }
}