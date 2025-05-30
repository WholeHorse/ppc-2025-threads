#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {
void RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound> bounds, std::size_t discretization, double ref,
                 const polikanov_v_rectangles::FunctionExecutor &function) {
  boost::mpi::communicator world;

  double out = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&discretization));
    task_data->inputs_count.emplace_back(bounds.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  polikanov_v_rectangles::TaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(out, ref, 0.2);
  }
}
}  // namespace

TEST(polikanov_v_rectangles_all, exp) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 1.71,
              [](const auto &p) { return std::exp(p[0]); });
}

TEST(polikanov_v_rectangles_all, exp_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 3.15,
              [](const auto &p) { return std::pow(std::exp(p[0]), 2); });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 1.45,
              [](const auto &p) { return std::exp(std::pow(p[0], 2)); });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 0.2,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2); });
}

TEST(polikanov_v_rectangles_all, exp__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 2.9,
              [](const auto &p) { return std::exp(p[0]) * std::exp(p[1]); });
}

TEST(polikanov_v_rectangles_all, exp_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 9.93,
              [](const auto &p) { return std::pow(std::exp(p[0]), 2) * std::pow(std::exp(p[1]), 2); });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 2.10,
              [](const auto &p) { return std::exp(std::pow(p[0], 2)) * std::exp(std::pow(p[1], 2)); });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 0.03,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2); });
}

TEST(polikanov_v_rectangles_all, exp__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 4.97,
              [](const auto &p) { return std::exp(p[0]) * std::exp(p[1]) * std::exp(p[2]); });
}

TEST(polikanov_v_rectangles_all, exp_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 31.31, [](const auto &p) {
    return std::pow(std::exp(p[0]), 2) * std::pow(std::exp(p[1]), 2) * std::pow(std::exp(p[2]), 2);
  });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 3.05, [](const auto &p) {
    return std::exp(std::pow(p[0], 2)) * std::exp(std::pow(p[1], 2)) * std::exp(std::pow(p[2], 2));
  });
}

TEST(polikanov_v_rectangles_all, exp_arg_pow_2_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.007, [](const auto &p) {
    return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2) * std::pow(std::pow(p[2], 2), 2);
  });
}

TEST(polikanov_v_rectangles_all, sin) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 0.454,
              [](const auto &p) { return std::sin(p[0]); });
}

TEST(polikanov_v_rectangles_all, sin_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 0.267,
              [](const auto &p) { return std::pow(std::sin(p[0]), 2); });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 0.304,
              [](const auto &p) { return std::sin(std::pow(p[0], 2)); });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 150, 0.0,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2); });
}

TEST(polikanov_v_rectangles_all, sin__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 0.206,
              [](const auto &p) { return std::sin(p[0]) * std::sin(p[1]); });
}

TEST(polikanov_v_rectangles_all, sin_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 0.071,
              [](const auto &p) { return std::pow(std::sin(p[0]), 2) * std::pow(std::sin(p[1]), 2); });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 0.092,
              [](const auto &p) { return std::sin(std::pow(p[0], 2)) * std::sin(std::pow(p[1], 2)); });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 150, 0.0,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2); });
}

TEST(polikanov_v_rectangles_all, sin__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.0936,
              [](const auto &p) { return std::sin(p[0]) * std::sin(p[1]) * std::sin(p[2]); });
}

TEST(polikanov_v_rectangles_all, sin_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.0192, [](const auto &p) {
    return std::pow(std::sin(p[0]), 2) * std::pow(std::sin(p[1]), 2) * std::pow(std::sin(p[2]), 2);
  });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.0282, [](const auto &p) {
    return std::sin(std::pow(p[0], 2)) * std::sin(std::pow(p[1], 2)) * std::sin(std::pow(p[2], 2));
  });
}

TEST(polikanov_v_rectangles_all, sin_arg_pow_2_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.0, [](const auto &p) {
    return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2) * std::pow(std::pow(p[2], 2), 2);
  });
}

TEST(polikanov_v_rectangles_all, exp_sin_cos) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 80, 0.66,
              [](const auto &p) { return std::exp(p[0]) * std::sin(p[0]) * std::cos(p[0]); });
}

TEST(polikanov_v_rectangles_all, linear_func) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 80, 1,
              [](const auto &p) { return p[0] + p[1]; });
}

TEST(polikanov_v_rectangles_all, log) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {3, 4}), 80, 2.495,
              [](const auto &p) { return std::log(p[0]) + std::log(p[1]); });
}
