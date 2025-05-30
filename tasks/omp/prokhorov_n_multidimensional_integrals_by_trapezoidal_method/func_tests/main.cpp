#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_omp.hpp"

const double kPi = std::numbers::pi;

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp, test_integral_1d_quadratic) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {1.0};
  std::vector<int> steps = {100000};
  double expected = 1.0 / 3.0;
  double result = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_omp->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_omp->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_omp->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_omp->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  test_task_omp.SetFunction([](const std::vector<double>& point) { return point[0] * point[0]; });

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp, test_integral_2d_linear) {
  std::vector<double> lower = {0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {1000, 1000};
  double expected = 1.0;
  double result = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_omp->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_omp->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_omp->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_omp->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  test_task_omp.SetFunction([](const std::vector<double>& point) { return point[0] + point[1]; });

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp, test_integral_3d_cubic) {
  std::vector<double> lower = {0.0, 0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {100, 100, 100};
  double expected = 0.125;
  double result = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_omp->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_omp->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_omp->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_omp->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  test_task_omp.SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; });

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-4);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp, test_integral_2d_circle_area) {
  std::vector<double> lower = {-1.0, -1.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {1000, 1000};
  double expected = kPi;
  double result = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_omp->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_omp->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_omp->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_omp->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  test_task_omp.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0] + point[1] * point[1] <= 1.0) ? 1.0 : 0.0; });

  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-2);
}