#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_tbb {

class HoareSortTBB : public ppc::core::Task {
 public:
  explicit HoareSortTBB(std::shared_ptr<ppc::core::TaskData> task_data);
  bool Validation() override;
  bool PreProcessing() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<int> data_;
  static constexpr std::size_t kThreshold = 10000;

  void SequentialQuickSort(int* arr, std::size_t left, std::size_t right);
  static std::size_t Partition(int* arr, std::size_t left, std::size_t right);
  void ParallelQuickSort(int* arr, std::size_t left, std::size_t right);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shuravina_o_hoare_simple_merger_tbb