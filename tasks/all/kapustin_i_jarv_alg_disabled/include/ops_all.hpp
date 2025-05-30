#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_jarv_alg_all {

class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::pair<int, int>> input_;
  std::vector<std::pair<int, int>> output_;
  std::pair<int, int> current_point_;
  std::pair<int, int> next_point_;
  size_t leftmost_index_;
  std::pair<int, int> FindLocalBestOMP(size_t start, size_t end, size_t current_index,
                                       const std::pair<int, int>& init_best);

  [[nodiscard]] static int Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                       const std::pair<int, int>& r);
  [[nodiscard]] static int CalculateDistance(const std::pair<int, int>& p1, const std::pair<int, int>& p2);
};

}  // namespace kapustin_i_jarv_alg_all