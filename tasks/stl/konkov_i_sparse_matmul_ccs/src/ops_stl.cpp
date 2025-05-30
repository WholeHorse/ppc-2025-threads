#include "stl/konkov_i_sparse_matmul_ccs/include/ops_stl.hpp"

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_stl {

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

void SparseMatmulTask::ProcessColumn(int thread_id, int col_b, std::vector<double>& thread_values,
                                     std::vector<int>& thread_row_indices, std::vector<int>& thread_col_ptr) {
  std::unordered_map<int, double> column_result;

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
      column_result[row_a] += val_a * val_b;
    }
  }

  std::vector<int> rows;
  for (const auto& pair : column_result) {
    if (pair.second != 0.0) {
      rows.push_back(pair.first);
    }
  }

  std::ranges::sort(rows);

  for (int row : rows) {
    thread_values.push_back(column_result[row]);
    thread_row_indices.push_back(row);
    thread_col_ptr[col_b + 1]++;
  }
}

void SparseMatmulTask::MergeThreadResults(int num_threads, const std::vector<std::vector<double>>& thread_c_values,
                                          const std::vector<std::vector<int>>& thread_c_row_indices,
                                          const std::vector<std::vector<int>>& thread_c_col_ptr) {
  for (int col = 0; col < colsB; ++col) {
    for (int t = 0; t < num_threads; ++t) {
      int start = (col == 0) ? 0 : thread_c_col_ptr[t][col];
      int end = thread_c_col_ptr[t][col + 1];

      C_col_ptr[col + 1] += end - start;
      C_values.insert(C_values.end(), thread_c_values[t].begin() + start, thread_c_values[t].begin() + end);
      C_row_indices.insert(C_row_indices.end(), thread_c_row_indices[t].begin() + start,
                           thread_c_row_indices[t].begin() + end);
    }
  }

  // prefix sum for final C_col_ptr
  for (int col = 1; col <= colsB; ++col) {
    C_col_ptr[col] += C_col_ptr[col - 1];
  }
}

bool SparseMatmulTask::RunImpl() {
  auto num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 4;  // fallback
  }

  std::vector<std::vector<double>> thread_c_values(num_threads);
  std::vector<std::vector<int>> thread_c_row_indices(num_threads);
  std::vector<std::vector<int>> thread_c_col_ptr(num_threads, std::vector<int>(colsB + 1, 0));

  auto worker = [&](int thread_id) {
    for (int col_b = thread_id; col_b < colsB; col_b += num_threads) {
      ProcessColumn(thread_id, col_b, thread_c_values[thread_id], thread_c_row_indices[thread_id],
                    thread_c_col_ptr[thread_id]);
    }

    // prefix sum within this thread's portion
    for (int col = 1; col <= colsB; ++col) {
      thread_c_col_ptr[thread_id][col] += thread_c_col_ptr[thread_id][col - 1];
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Merge thread results
  C_values.clear();
  C_row_indices.clear();
  C_col_ptr.assign(colsB + 1, 0);

  MergeThreadResults(num_threads, thread_c_values, thread_c_row_indices, thread_c_col_ptr);

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_stl
