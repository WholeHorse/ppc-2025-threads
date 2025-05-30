#include "tbb/konkov_i_sparse_matmul_ccs/include/ops_tbb.hpp"

#include <tbb/global_control.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "oneapi/tbb/concurrent_unordered_map.h"
#include "oneapi/tbb/parallel_for.h"

namespace konkov_i_sparse_matmul_ccs {

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
  std::vector<tbb::concurrent_unordered_map<int, double>> column_map(colsB);

  MultiplySparseMatrices(column_map);
  BuildResultMatrix(column_map);

  return true;
}

void SparseMatmulTask::MultiplySparseMatrices(std::vector<tbb::concurrent_unordered_map<int, double>>& column_map) {
  tbb::parallel_for(tbb::blocked_range<int>(0, colsB), [&](const tbb::blocked_range<int>& range) {
    for (int col_b = range.begin(); col_b < range.end(); ++col_b) {
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
          column_map[col_b][row_a] += val_a * val_b;
        }
      }
    }
  });
}

void SparseMatmulTask::BuildResultMatrix(const std::vector<tbb::concurrent_unordered_map<int, double>>& column_map) {
  C_col_ptr.resize(colsB + 1, 0);
  int count = 0;

  for (int col = 0; col < colsB; ++col) {
    std::vector<std::pair<int, double>> sorted_rows;

    for (const auto& pair : column_map[col]) {
      if (pair.second != 0) {
        sorted_rows.emplace_back(pair.first, pair.second);
      }
    }

    std::ranges::sort(sorted_rows);

    for (const auto& pair : sorted_rows) {
      C_values.push_back(pair.second);
      C_row_indices.push_back(pair.first);
      count++;
    }

    C_col_ptr[col + 1] = count;
  }
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs
