#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_all {

class RadixALL : public ppc::core::Task {
 public:
  explicit RadixALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  void Squash(boost::mpi::communicator& group);
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  std::vector<int> procchunk_;
  boost::mpi::communicator world_;
};

}  // namespace burykin_m_radix_all