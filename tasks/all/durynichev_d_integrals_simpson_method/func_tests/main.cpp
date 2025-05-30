#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/durynichev_d_integrals_simpson_method/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(durynichev_d_integrals_simpson_method_all, test_integral_1D_x_squared) {
  boost::mpi::communicator world;

  std::vector<double> in = {0.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], 1.0 / 3.0, 1e-5);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_1D_x_squared_reverse) {
  boost::mpi::communicator world;

  std::vector<double> in = {1.0, 0.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], -(1.0 / 3.0), 1e-5);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_1D_x_squared_wider_range) {
  boost::mpi::communicator world;

  std::vector<double> in = {0.0, 2.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], 8.0 / 3.0, 1e-5);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_2D_x2_plus_y2) {
  boost::mpi::communicator world;

  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], 2.0 / 3.0, 1e-4);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_2D_x2_plus_y2_reverse1) {
  boost::mpi::communicator world;

  std::vector<double> in = {1.0, 0.0, 1.0, 0.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], 2.0 / 3.0, 1e-4);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_2D_x2_plus_y2_reverse2) {
  boost::mpi::communicator world;

  std::vector<double> in = {1.0, 0.0, 0.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], -(2.0 / 3.0), 1e-4);
  }
}

TEST(durynichev_d_integrals_simpson_method_all, test_integral_2D_x2_plus_y2_neg_bounds) {
  boost::mpi::communicator world;

  std::vector<double> in = {-1.0, 1.0, -1.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  durynichev_d_integrals_simpson_method_all::SimpsonIntegralSTLMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_NEAR(out[0], 2 + (2.0 / 3.0), 1e-4);
  }
}