#include "omp/konkov_i_sparse_matmul_ccs/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_omp {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    return false;
  }
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  C_col_ptr.resize(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTask::RunImpl() {
  std::vector<std::unordered_map<int, double>> column_map(colsB);

#pragma omp parallel for
  for (int col_b = 0; col_b < colsB; ++col_b) {
    for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
      int row_b = B_row_indices[j];
      double val_b = B_values[j];

      if (row_b >= colsA) {
        continue;
      }

      for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
        if (static_cast<size_t>(k) >= A_row_indices.size()) {
          continue;
        }

        int row_a = A_row_indices[k];
        double val_a = A_values[k];

#pragma omp critical
        column_map[col_b][row_a] += val_a * val_b;
      }
    }
  }

  C_col_ptr.resize(colsB + 1, 0);
  int count = 0;

  for (int col = 0; col < colsB; ++col) {
    std::vector<int> rows;
    for (const auto& pair : column_map[col]) {
      if (pair.second != 0) {
        rows.push_back(pair.first);
      }
    }
    std::ranges::sort(rows);

    for (int row : rows) {
      C_values.push_back(column_map[col][row]);
      C_row_indices.push_back(row);
      count++;
    }
    C_col_ptr[col + 1] = count;
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_omp
