
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/vladimirova_j_m_monte_karlo/include/ops_all.hpp"
#include "core/task/include/task.hpp"
namespace {
bool PiVal314(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((x * x) + (y * y)) <= 1);
}  // namespace

bool SphereR3(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return (((x * x) + (y * y) + (z * z)) <= 9);
};

bool Rectangle34(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return ((x >= 0) && (x <= 3) && (y >= 0) && (y <= 4));
}

bool Parallelogram333(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return ((x >= 0) && (x <= 3) && (y >= 0) && (y <= 3) && (z >= 0) && (z <= 3));
}

bool BigParallelogram100100100(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return ((x >= 0) && (x <= 100) && (y >= 0) && (y <= 100) && (z >= 0) && (z <= 100));
}

bool Parabola(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((3 * x * x) + (12 * x) - 10) <= y);
}

bool HypercubeX4Pr4433(std::vector<double> arr, size_t size = 4) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  double w = arr[3];

  return ((x >= 0) && (x <= 4) && (y >= 0) && (y <= 4) && (z >= 0) && (z <= 3) && (w >= 0) && (w <= 3));
}

bool TriangleModuleMinus5(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  if (x > 0) {
    x *= -1;
  }
  return x <= (y - 5);
}

bool SomeRandomFunc(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((x * x) - (1.2 * x * x * x) - y - 5) <= 0);
}

}  // namespace

TEST(vladimirova_j_m_monte_karlo_all, test_zero_var) {
  boost::mpi::communicator world;
  // Create data

  std::vector<double> val_b = {};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_one_var) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_no_limit_var) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-1, 1, 0};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_zero_size_limit_var) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-1, 1, 0, 0};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_one_point) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-1, 1, 0, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(1));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((2 == (int)out[0]) || ((int)out[0] == 0));
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_PiVal_314) {
  boost::mpi::communicator world;
  //  Create data
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(9000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(3, (int)out[0]);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_PiVal_314_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(3, (int)out[0]);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_PiVal_314_1_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 1, -1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((3.14 / 2) - out[0]) < 0.5);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_SphereR3_1) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-3, 3, -3, 3, -3, 3};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((113 - out[0]) < 3);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_SphereR3_1_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 3, -3, 3, -3, 3};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((113 / 2) - (int)out[0]) < 5);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_SphereR3_1_9) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 3, 0, 3, 0, 3};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(9000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((113 / 9) - (int)out[0]) < 3);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_Rectangle34) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 3, 0, 4};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(Rectangle34));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(30000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((3 * 4) - out[0]) == 0);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_Rectangle34_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 4, 0, 4};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(Rectangle34));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(30000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((3 * 4) - out[0]) < 0.5);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_Parallelogram333) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 3, 0, 3, 0, 3};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(Parallelogram333));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(900));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((3 * 3 * 3) - out[0]) == 0);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_BigParallelogram100100100) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 100, 0, 100, 0, 100};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(BigParallelogram100100100));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(((100 * 100 * 100) - out[0]) == 0);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_Parabola_1_5_2_9) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-4, 0, -24, 0};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(Parabola));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(65000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((72 - out[0]) < 1);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_HypercubeX4Pr4433) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {0, 4, 0, 4, 0, 4, 0, 4};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(HypercubeX4Pr4433));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(90000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((3 * 3 * 4 * 4 - out[0]) < 5);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_TriangleModuleMinus5) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-5, 5, 0, 5};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(TriangleModuleMinus5));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(65000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((25 - out[0]) < 1);
  }
}

TEST(vladimirova_j_m_monte_karlo_all, test_SomeRandomFunc) {
  boost::mpi::communicator world;
  // Create data
  std::vector<double> val_b = {-2, 0, -5.5, -2};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(SomeRandomFunc));
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(9000));
  task_data_all->inputs_count.emplace_back(val_b.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());
  // Create Task
  vladimirova_j_m_monte_karlo_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE((2.533 - out[0]) < 0.5);
  }
}
