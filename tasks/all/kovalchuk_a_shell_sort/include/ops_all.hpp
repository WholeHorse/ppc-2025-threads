#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort_all {

class ShellSortAll : public ppc::core::Task {
 public:
  explicit ShellSortAll(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void SimpleMerge(int num_procs, const std::vector<int>& gathered, const std::vector<int>& displs, int total_size);
  std::vector<int> input_, counts_, result_;
  boost::mpi::communicator world_, group_;
};

}  // namespace kovalchuk_a_shell_sort_all
