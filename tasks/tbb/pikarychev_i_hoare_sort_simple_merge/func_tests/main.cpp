#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/pikarychev_i_hoare_sort_simple_merge/include/ops_tbb.hpp"

namespace {
void PerformTest(int size, bool reverse) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-5000, 5000);

  std::vector<int> in(size);
  std::ranges::generate(in, [&] { return dist(gen); });

  std::vector<int> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&reverse));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = pikarychev_i_hoare_sort_simple_merge::HoareThreadBB<int>(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out, [&](const auto& a, const auto& b) { return reverse ? (a < b) : (a > b); }));
}
}  // namespace

TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_0) { PerformTest(0, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_1) { PerformTest(1, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_2) { PerformTest(2, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_3) { PerformTest(3, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_4) { PerformTest(4, false); }
TEST(pikarychev_i_hoare_sort_simple_merge_tbb, standard_5) { PerformTest(5, false); }