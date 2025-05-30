#pragma once
#include <tbb/tbb.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "oneapi/tbb/concurrent_unordered_map.h"

namespace konkov_i_sparse_matmul_ccs {

class SparseMatmulTask : public ppc::core::Task {
 public:
  explicit SparseMatmulTask(ppc::core::TaskDataPtr task_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  void MultiplySparseMatrices(std::vector<tbb::concurrent_unordered_map<int, double>>& column_map);
  void BuildResultMatrix(const std::vector<tbb::concurrent_unordered_map<int, double>>& column_map);
  bool PostProcessingImpl() override;

  std::vector<double> A_values, B_values, C_values;
  std::vector<int> A_row_indices, B_row_indices, C_row_indices;
  std::vector<int> A_col_ptr, B_col_ptr, C_col_ptr;
  int rowsA, colsA, rowsB, colsB;
};

}  // namespace konkov_i_sparse_matmul_ccs