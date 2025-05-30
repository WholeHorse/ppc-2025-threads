#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_tbb.hpp"

const double kPi = std::numbers::pi;

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_integral_1d_quadratic) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {1.0};
  std::vector<int> steps = {100000};
  double expected = 1.0 / 3.0;
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  test_task_tbb.SetFunction([](const std::vector<double>& point) { return point[0] * point[0]; });

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_integral_2d_linear) {
  std::vector<double> lower = {0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {1000, 1000};
  double expected = 1.0;
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  test_task_tbb.SetFunction([](const std::vector<double>& point) { return point[0] + point[1]; });

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_integral_3d_cubic) {
  std::vector<double> lower = {0.0, 0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {100, 100, 100};
  double expected = 0.125;
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  test_task_tbb.SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; });

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-4);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb, test_integral_2d_circle_area) {
  std::vector<double> lower = {-1.0, -1.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {1000, 1000};
  double expected = kPi;
  double result = 0.0;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_tbb->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_tbb->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_tbb->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_tbb->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  test_task_tbb.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0] + point[1] * point[1] <= 1.0) ? 1.0 : 0.0; });

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-2);
}