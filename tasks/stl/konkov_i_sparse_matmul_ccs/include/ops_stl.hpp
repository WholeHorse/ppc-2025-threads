#pragma once
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_stl {

class SparseMatmulTask : public ppc::core::Task {
 public:
  explicit SparseMatmulTask(ppc::core::TaskDataPtr task_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  void ProcessColumn(int thread_id, int col_b, std::vector<double>& thread_values, std::vector<int>& thread_row_indices,
                     std::vector<int>& thread_col_ptr);
  void MergeThreadResults(int num_threads, const std::vector<std::vector<double>>& thread_c_values,
                          const std::vector<std::vector<int>>& thread_c_row_indices,
                          const std::vector<std::vector<int>>& thread_c_col_ptr);
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> A_values, B_values, C_values;
  std::vector<int> A_row_indices, B_row_indices, C_row_indices;
  std::vector<int> A_col_ptr, B_col_ptr, C_col_ptr;
  int rowsA, colsA, rowsB, colsB;
};

}  // namespace konkov_i_sparse_matmul_ccs_stl
