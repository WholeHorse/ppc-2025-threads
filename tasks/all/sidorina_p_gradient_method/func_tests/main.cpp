#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "all/sidorina_p_gradient_method/include/ops_all.hpp"
#include "core/task/include/task.hpp"

using Params =
    std::tuple<int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double>;

using ParamsVal =
    std::tuple<int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double>;

namespace {
class SidorinaPGradientMethodAllTest : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(SidorinaPGradientMethodAllTest, Test_matrix) {
  const auto &[size, a, b, solution, expected, tolerance] = GetParam();
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task->outputs_count.emplace_back(result.size());

  sidorina_p_gradient_method_all::GradientMethod gradient_method(task);

  ASSERT_TRUE(gradient_method.ValidationImpl());
  gradient_method.PreProcessingImpl();
  gradient_method.RunImpl();
  gradient_method.PostProcessingImpl();
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_NEAR(result[i], expected[i], tolerance);
  }
}

class SidorinaPGradientMethodAllTestVal : public ::testing::TestWithParam<ParamsVal> {
 protected:
};

TEST_P(SidorinaPGradientMethodAllTestVal, Test_validation) {
  const auto &[size, a, b, solution, expected, tolerance] = GetParam();
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task->outputs_count.emplace_back(result.size());

  sidorina_p_gradient_method_all::GradientMethod gradient_method(task);

  ASSERT_FALSE(gradient_method.ValidationImpl());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(SidorinaPGradientMethodAllTest, SidorinaPGradientMethodAllTest,
                         ::testing::Values(Params(1, {2}, {4}, {0}, {2}, 1e-6), 
                                           Params(2, {3, 2, 2, 7}, {1, 8}, {0, 0}, {-0.529411764, 1.294117647}, 1e-7),
                                           Params(2, {3, 5, 5, 20}, {19, 55}, {0, 0}, {3, 2}, 1e-6),
                                           Params(2, {6, 2, 2, 10}, {5, 11}, {0, -2}, {0.5, 1}, 1e-6),
                                           Params(2, {8, -3, -3, 6}, {15, -30}, {-5, 0}, {0, -5}, 1e-6),
                                           Params(2, {70, 12, 12, 8}, {100, 7}, {0, 0}, {1.72, -1.7}, 1e-2),
                                           Params(3, {4, -1, 2, -1, 6, -2, 2, -2, 5}, {-1, 9, -10}, {-3, 5, 0}, {1, 1, -2}, 1e-3)));


INSTANTIATE_TEST_SUITE_P(SidorinaPGradientMethodAllTestVal, SidorinaPGradientMethodAllTestVal,
                         ::testing::Values(Params(0, {2}, {4}, {0}, {2}, 1e-6),
                                           Params(1, {}, {4}, {0}, {2}, 1e-6),
                                           Params(-1, {2}, {4}, {0}, {2}, 1e-6),
                                           Params(1, {2}, {}, {0}, {2}, 1e-6),
                                           Params(1, {2}, {4}, {}, {2}, 1e-6),
                                           Params(1, {2}, {4}, {0}, {}, 1e-6),
                                           Params(2, {2}, {4}, {0}, {2}, 1e-6),
                                           Params(1, {2, 3, 4, 5}, {4, 2}, {0, 0}, {2, 0}, 1e-6),
                                           Params(3, {2, 3, 4, 5}, {4, 2, 3}, {0, 0, 0}, {2, 0, 0}, 1e-6),
                                           Params(2, {2, 3, 4, 5}, {4, 2, 4}, {0, 0}, {2, 0}, 1e-6)));
//clang-format on

}  // namespace