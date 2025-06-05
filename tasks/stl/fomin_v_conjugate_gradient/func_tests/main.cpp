#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

TEST(FominVConjugateGradientStl, DotProduct) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  double expected = (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientStl task(task_data);

  EXPECT_DOUBLE_EQ(task.DotProduct(a, b), expected);
}

TEST(FominVConjugateGradientStl, MatrixVectorMultiply) {
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};  // 2x2 матрица
  std::vector<double> x = {5.0, 6.0};
  std::vector<double> expected = {(1 * 5) + (2 * 6), (3 * 5) + (4 * 6)};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientStl task(task_data);

  task.n = 2;

  auto result = task.MatrixVectorMultiply(a, x);
  EXPECT_EQ(result, expected);
}

TEST(FominVConjugateGradientStl, VectorAdd) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {5.0, 7.0, 9.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorAdd(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(FominVConjugateGradientStl, VectorSub) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {-3.0, -3.0, -3.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorSub(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(FominVConjugateGradientStl, VectorScalarMultiply) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  double scalar = 2.0;
  std::vector<double> expected = {2.0, 4.0, 6.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorScalarMultiply(v, scalar);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}
