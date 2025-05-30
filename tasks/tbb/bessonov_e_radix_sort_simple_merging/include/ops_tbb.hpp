#pragma once

#include <cstdint>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_tbb {

class TestTaskTbb : public ppc::core::Task {
 public:
  explicit TestTaskTbb(ppc::core::TaskDataPtr task_data);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;

  static void ConvertToSortableBits(const std::vector<double>& in, std::vector<uint64_t>& out);
  static void ConvertToDoubles(const std::vector<uint64_t>& in, std::vector<double>& out);
  static void RadixSort(std::vector<uint64_t>& data);
};

}  // namespace bessonov_e_radix_sort_simple_merging_tbb