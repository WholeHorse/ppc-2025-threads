#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "all/chizhov_m_trapezoid_method/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {
void RunTests(int div, int dimm, std::vector<double> &limits, std::function<double(const std::vector<double> &)> f,
              double expected_result) {
  boost::mpi::communicator world;
  std::vector<double> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
    task_data_mpi->inputs_count.emplace_back(sizeof(div));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimm));
    task_data_mpi->inputs_count.emplace_back(sizeof(dimm));

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_mpi->inputs_count.emplace_back(limits.size());

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  chizhov_m_trapezoid_method_all::TestTaskMPI test_task_mpi(task_data_mpi);
  test_task_mpi.SetFunc(std::move(f));

  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], expected_result, 0.1);
  }
}

TEST(chizhov_m_trapezoid_method_all, one_variable_squared) {
  int div = 20;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0]; };
  RunTests(div, dim, limits, f, 41.66);
}

TEST(chizhov_m_trapezoid_method_all, one_variable_cube) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0] * f_val[0]; };
  RunTests(div, dim, limits, f, 156.25);
}

TEST(chizhov_m_trapezoid_method_all, mul_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[1]; };
  RunTests(div, dim, limits, f, 56.25);
}

TEST(chizhov_m_trapezoid_method_all, sum_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] + f_val[1]; };
  RunTests(div, dim, limits, f, 60);
}

TEST(chizhov_m_trapezoid_method_all, dif_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[1] - f_val[0]; };
  RunTests(div, dim, limits, f, -15);
}

TEST(chizhov_m_trapezoid_method_all, cos_one_variable) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return std::cos(f_val[0]); };
  RunTests(div, dim, limits, f, -0.95);
}

TEST(chizhov_m_trapezoid_method_all, sin_two_variables) {
  int div = 45;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return std::sin(f_val[0] + f_val[1]); };
  RunTests(div, dim, limits, f, -1.37);
}

TEST(chizhov_m_trapezoid_method_all, exp_two_variables) {
  int div = 80;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return std::exp(f_val[0] + f_val[1]); };
  RunTests(div, dim, limits, f, 364.25);
}

TEST(chizhov_m_trapezoid_method_all, combine_exp_sin_cos) {
  int div = 90;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) {
    return std::exp(-f_val[0]) * std::sin(f_val[0]) * std::cos(f_val[1]);
  };
  RunTests(div, dim, limits, f, 0.073);
}

}  // namespace

TEST(chizhov_m_trapezoid_method_all, invalid_value_dim) {
  int div = 10;
  int dim = -2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_mpi->inputs_count.emplace_back(sizeof(div));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_mpi->inputs_count.emplace_back(sizeof(dim));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_mpi->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_all::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_FALSE(test_task_mpi.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_all, invalid_value_div) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_mpi->inputs_count.emplace_back(sizeof(div));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_mpi->inputs_count.emplace_back(sizeof(dim));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_mpi->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_all::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_FALSE(test_task_mpi.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_all, invalid_limit_size) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0};

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_mpi->inputs_count.emplace_back(sizeof(div));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_mpi->inputs_count.emplace_back(sizeof(dim));

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_mpi->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_all::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_FALSE(test_task_mpi.ValidationImpl());
}