#include <gtest/gtest.h>
#define OMPI_SKIP_MPICXX
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

#include "all/anufriev_d_integrals_simpson/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
const double kPi = std::numbers::pi;

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& elements,
                                                  std::vector<double>& out_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* input_ptr = reinterpret_cast<uint8_t*>(const_cast<double*>(elements.data()));
  auto* output_ptr = reinterpret_cast<uint8_t*>(out_buffer.data());
  task_data->inputs.push_back(input_ptr);
  task_data->inputs_count.push_back(static_cast<uint32_t>(elements.size() * sizeof(double)));
  task_data->outputs.push_back(output_ptr);
  task_data->outputs_count.push_back(static_cast<uint32_t>(out_buffer.size() * sizeof(double)));
  return task_data;
}
}  // namespace

TEST(anufriev_d_integrals_simpson_all, test_1D_sin) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {1, 0.0, kPi / 2.0, 100, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    double result = out_buffer[0];
    EXPECT_NEAR(result, 1.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_2D_sum_of_squares) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  if (rank == 0) {
    double result = out_buffer[0];
    EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_2D_sin_cos) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {2, 0.0, kPi / 2.0, 200, 0.0, kPi / 2.0, 200, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  if (rank == 0) {
    double result = out_buffer[0];
    EXPECT_NEAR(result, 1.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_unknown_func) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  if (rank == 0) {
    double result = out_buffer[0];
    EXPECT_DOUBLE_EQ(result, 0.0);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_empty_input_ptr) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(nullptr);
  task_data->inputs_count.push_back(0);
  std::vector<double> out_buffer(1, 0.0);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_buffer.data()));
  task_data->outputs_count.push_back(sizeof(double));

  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(task_data);
  ASSERT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_output_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  task_data->inputs_count.push_back(in.size() * sizeof(double));
  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(task_data);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_dimension_zero) {
  std::vector<double> in = {0, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_dimension_negative) {
  std::vector<double> in = {-1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_not_enough_data) {
  std::vector<double> in = {2, 0.0, 1.0, 200};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_odd_n) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_negative_n) {
  std::vector<double> in = {1, 0.0, 1.0, -2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_no_output_buffer_in_taskdata) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_n_not_integer) {
  std::vector<double> in = {1, 0.0, 1.0, 100.5, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_n_too_large) {
  std::vector<double> in = {1, 0.0, 1.0, static_cast<double>(std::numeric_limits<int>::max()) + 10.0, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_func_code_not_integer) {
  std::vector<double> in = {1, 0.0, 1.0, 100, 1.5};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_func_code_too_large) {
  std::vector<double> in = {1, 0.0, 1.0, 100, static_cast<double>(std::numeric_limits<int>::max()) + 10.0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_func_code_too_small) {
  std::vector<double> in = {1, 0.0, 1.0, 100, static_cast<double>(std::numeric_limits<int>::min()) - 10.0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_a_greater_than_b) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {1, 1.0, 0.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  if (rank == 0) {
    double result = out_buffer[0];
    EXPECT_NEAR(result, -1.0 / 3.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_empty_output_count) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out_buf(1);
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_invalid_small_output_count) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out_buf(1);
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
  td->outputs_count.push_back(sizeof(double) - 1);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_all, test_dimension_zero_valid_case) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {0, 0};
  std::vector<double> out_buffer(1, 999.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(out_buffer[0], 0.0);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_calculate_run_params_n_is_zero_inside_dim) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {1, 0.0, 1.0, 0, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_all, test_run_total_points_zero_after_calc) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> in = {0, 0};
  std::vector<double> out_buffer(1, 999.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(out_buffer[0], 0.0);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_distribute_few_points_many_procs) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_NEAR(out_buffer[0], 1.0 / 3.0, 1e-3);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_distribute_one_point_one_proc) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size != 1) {
    GTEST_SKIP() << "Skipping single point test for world_size > 1";
    return;
  }

  std::vector<double> in = {1, 0.0, 0.1, 2, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_NEAR(out_buffer[0], 1.0 - std::cos(0.1), 1e-5);
  }
}

TEST(anufriev_d_integrals_simpson_all, test_preprocessing_invalid_n_on_non_root) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size < 2 && rank != 0) {
    GTEST_SKIP() << "Skipping non-root n validation test for world_size < 2 or if this is root";
  }

  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_all, test_run_result_zero_on_non_root) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<double> in = {1, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_all::IntegralsSimpsonAll task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank != 0) {
  } else {
    EXPECT_NEAR(out_buffer[0], 1.0 / 3.0, 1e-3);
  }
}