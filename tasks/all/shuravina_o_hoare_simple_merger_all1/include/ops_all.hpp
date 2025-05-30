#ifndef OPS_ALL_HPP
#define OPS_ALL_HPP

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(std::shared_ptr<ppc::core::TaskData> task_data);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<int> local_data_;

  void QuickSort(std::vector<int>& arr, int low, int high);
  static void Merge(std::vector<int>& arr, int low, int mid, int high);
  void ParallelQuickSort(std::vector<int>& arr);
  void DistributeData();
  void GatherAndMergeResults();
};

}  // namespace shuravina_o_hoare_simple_merger

#endif