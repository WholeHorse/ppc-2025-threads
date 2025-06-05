#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/fomin_v_conjugate_gradient/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(FominVConjugateGradientAll, test_small_system) {
  constexpr size_t kCount = 3;
  std::vector<double> a = {4, 1, 1, 1, 3, 0, 1, 0, 2};
  std::vector<double> b = {6, 5, 3};
  std::vector<double> expected_x = {17.0 / 19.0, 26.0 / 19.0, 20.0 / 19.0};

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  std::vector<double> out(kCount, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::FominVConjugateGradientAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(FominVConjugateGradientAll, test_large_system) {
  constexpr size_t kCount = 5;
  std::vector<double> a = {5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5};
  std::vector<double> b = {6, 7, 7, 7, 6};
  std::vector<double> expected_x = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(kCount, 0.0);

  // Correct input setup
  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::FominVConjugateGradientAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(FominVConjugateGradientAll, DotProduct) {
  boost::mpi::communicator world;
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  double expected = 32.0;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientAll task(task_data);

  double result = fomin_v_conjugate_gradient::FominVConjugateGradientAll::DotProduct(world, a, b);
  if (world.rank() == 0) {
    EXPECT_DOUBLE_EQ(result, expected * world.size());
  }
}

TEST(FominVConjugateGradientAll, MatrixVectorMultiply) {
  boost::mpi::communicator world;

  constexpr size_t kCount = 2;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {5.0, 6.0};
  std::vector<double> expected = {17.0, 39.0};

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  std::vector<double> out(kCount, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::FominVConjugateGradientAll task(task_data);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();

  auto result = task.MatrixVectorMultiply(b);

  if (world.rank() == 0) {
    EXPECT_EQ(result, expected);
  }
}

TEST(FominVConjugateGradientAll, VectorAdd) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {5.0, 7.0, 9.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientAll task(task_data);
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorAdd(a, b);

  EXPECT_EQ(result, expected);
}

TEST(FominVConjugateGradientAll, VectorSub) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {-3.0, -3.0, -3.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientAll task(task_data);
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorSub(a, b);

  EXPECT_EQ(result, expected);
}

TEST(FominVConjugateGradientAll, VectorScalarMultiply) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  double scalar = 2.0;
  std::vector<double> expected = {2.0, 4.0, 6.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientAll task(task_data);
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorScalarMultiply(v, scalar);

  EXPECT_EQ(result, expected);
}
