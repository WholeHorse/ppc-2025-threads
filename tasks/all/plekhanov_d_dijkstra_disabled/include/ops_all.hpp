#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_dijkstra_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  [[nodiscard]] int GetRank() const { return world_.rank(); }

 private:
  boost::mpi::communicator world_;
  std::vector<int> graph_data_;
  std::vector<int> distances_;
  size_t start_vertex_;
  size_t num_vertices_;
  static const int kEndOfVertexList;
};

}  // namespace plekhanov_d_dijkstra_all