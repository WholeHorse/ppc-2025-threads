#pragma once
#include <omp.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_omp {

class SparseMatmulTask : public ppc::core::Task {
 public:
  explicit SparseMatmulTask(ppc::core::TaskDataPtr task_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> A_values, B_values, C_values;
  std::vector<int> A_row_indices, B_row_indices, C_row_indices;
  std::vector<int> A_col_ptr, B_col_ptr, C_col_ptr;
  int rowsA, colsA, rowsB, colsB;
};

}  // namespace konkov_i_sparse_matmul_ccs_omp