#pragma once

#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_all {

class RadixALL : public ppc::core::Task {
 public:
  explicit RadixALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::array<int, 256> ComputeFrequency(const std::vector<int>& a, int shift);
  static std::array<int, 256> ComputeIndices(const std::array<int, 256>& count);
  static void DistributeElements(const std::vector<int>& a, std::vector<int>& b, std::array<int, 256> index,
                                 const int shift);
  std::vector<int> PerformKWayMerge(const std::vector<int>& all_data, const std::vector<int>& sizes,
                                    const std::vector<int>& displs);
  void Merge(boost::mpi::communicator& group);

 private:
  int original_size_;
  std::vector<int> input_, output_;
  std::vector<int> procchunk_;
  boost::mpi::communicator world_;
};

}  // namespace burykin_m_radix_all