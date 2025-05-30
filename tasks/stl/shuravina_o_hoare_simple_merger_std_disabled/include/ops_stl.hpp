#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(std::shared_ptr<ppc::core::TaskData> task_data);

  bool Validation() override;
  bool PreProcessing() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;

  static void QuickSort(std::vector<int>& arr, int left, int right);
  static void MergeHelper(std::vector<int>& arr, int left, int mid, int right);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shuravina_o_hoare_simple_merger_stl