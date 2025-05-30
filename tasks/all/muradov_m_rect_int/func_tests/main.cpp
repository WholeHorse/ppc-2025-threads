#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#include "all/muradov_m_rect_int/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

constexpr double kAbsErr = 0.5;

namespace {
void MuradovMRectIntTest(int iterations, std::vector<std::pair<double, double>> bounds, double ref,
                         const muradov_m_rect_int_all::Matfun &fun) {
  boost::mpi::communicator world;

  double out = 0.0;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&iterations));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data->inputs_count.emplace_back(1);
    task_data->inputs_count.emplace_back(bounds.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    task_data->outputs_count.emplace_back(1);
  }
  muradov_m_rect_int_all::RectIntTaskMpiStlPar task(task_data, fun);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(out, ref, std::max(0.05 * ref, kAbsErr));
  }
}
}  // namespace

TEST(muradov_m_rect_int_all, onedim_zerobounds) {
  MuradovMRectIntTest(100, {std::make_pair(0., 0.)}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, twodim_zerobounds) {
  MuradovMRectIntTest(100, {{0., 0.}, {0., 0.}}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, threedim_zerobounds) {
  MuradovMRectIntTest(100, {{0., 0.}, {0., 0.}, {0., 0.}}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, onedim_samebounds) {
  MuradovMRectIntTest(100, {std::make_pair(5., 5.)}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, twodim_samebounds) {
  MuradovMRectIntTest(100, {{5., 5.}, {10., 10.}}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, threedim_samebounds) {
  MuradovMRectIntTest(100, {{5., 5.}, {10., 10.}, {20., 20.}}, 0, [](const auto &args) { return 200; });
}

TEST(muradov_m_rect_int_all, sin_mul_cos_1) {
  MuradovMRectIntTest(100, {std::make_pair(0, std::numbers::pi)}, 0,
                      [](const auto &args) { return std::sin(args[0]) * std::cos(args[0]); });
}

TEST(muradov_m_rect_int_all, sin_plus_cos_1) {
  MuradovMRectIntTest(100, {std::make_pair(0, std::numbers::pi)}, 2,
                      [](const auto &args) { return std::sin(args[0]) + std::cos(args[0]); });
}

TEST(muradov_m_rect_int_all, sin_mul_cos_2) {
  MuradovMRectIntTest(100, {{0, std::numbers::pi}, {0, std::numbers::pi}}, 0,
                      [](const auto &args) { return std::sin(args[0]) * std::cos(args[1]); });
}

TEST(muradov_m_rect_int_all, sin_plus_cos_2) {
  MuradovMRectIntTest(100, {{0, std::numbers::pi}, {0, std::numbers::pi}}, 2 * std::numbers::pi,
                      [](const auto &args) { return std::sin(args[0]) + std::cos(args[1]); });
}

TEST(muradov_m_rect_int_all, sin_mul_cos_3) {
  MuradovMRectIntTest(100, {{0, std::numbers::pi}, {0, std::numbers::pi}, {0, std::numbers::pi}}, 0,
                      [](const auto &args) {
                        return (std::sin(args[0]) * std::cos(args[1])) + (std::sin(args[1]) * std::cos(args[2]));
                      });
}

TEST(muradov_m_rect_int_all, sin_plus_cos_3) {
  MuradovMRectIntTest(100, {{0, std::numbers::pi}, {0, std::numbers::pi}, {0, std::numbers::pi}}, 4 * std::numbers::pi,
                      [](const auto &args) {
                        return (std::sin(args[0]) + std::cos(args[1])) * (std::sin(args[1]) + std::cos(args[2]));
                      });
}

TEST(muradov_m_rect_int_all, polynomial_sum_1) {
  MuradovMRectIntTest(100, {{0, 3}, {0, 3}}, 189. / 4.,
                      [](const auto &args) { return (args[0] * args[1]) + std::pow(args[1], 2); });
}

TEST(muradov_m_rect_int_all, polynomial_sum_2) {
  MuradovMRectIntTest(100, {{0, 2}, {0, 3}}, 27,
                      [](const auto &args) { return (args[0] * args[1]) + std::pow(args[1], 2); });
}

TEST(muradov_m_rect_int_all, neg_bounds) {
  MuradovMRectIntTest(100, {{-2, 2}, {-3, 3}}, 72.028,
                      [](const auto &args) { return (args[0] * args[1]) + std::pow(args[1], 2); });
}