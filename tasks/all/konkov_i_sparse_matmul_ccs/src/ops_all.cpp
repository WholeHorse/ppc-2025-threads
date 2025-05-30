#include "all/konkov_i_sparse_matmul_ccs/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <core/task/include/task.hpp>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// NOLINTNEXTLINE
BOOST_CLASS_TRACKING(std::vector<double>, boost::serialization::track_never)
// NOLINTNEXTLINE
BOOST_CLASS_TRACKING(std::vector<int>, boost::serialization::track_never)

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
      valid = false;
    }
    if (A_col_ptr.empty() || B_col_ptr.empty()) {
      valid = false;
    }
  }
  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

bool SparseMatmulTask::PreProcessingImpl() {
  C_col_ptr.clear();
  C_row_indices.clear();
  C_values.clear();

  boost::mpi::broadcast(world_, A_values, 0);
  boost::mpi::broadcast(world_, A_row_indices, 0);
  boost::mpi::broadcast(world_, A_col_ptr, 0);
  boost::mpi::broadcast(world_, rowsA, 0);
  boost::mpi::broadcast(world_, colsA, 0);

  boost::mpi::broadcast(world_, B_values, 0);
  boost::mpi::broadcast(world_, B_row_indices, 0);
  boost::mpi::broadcast(world_, B_col_ptr, 0);
  boost::mpi::broadcast(world_, rowsB, 0);
  boost::mpi::broadcast(world_, colsB, 0);

  return true;
}

void SparseMatmulTask::ProcessColumn(int col_b, int start_col, std::vector<double>& local_values,
                                     std::vector<int>& local_rows, std::vector<int>& local_col_ptr) {
  std::unordered_map<int, double> column_result;

  for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
    int row_b = B_row_indices[j];
    double val_b = B_values[j];

    for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
      int row_a = A_row_indices[k];
      double val_a = A_values[k];
      column_result[row_a] += val_a * val_b;
    }
  }

  std::vector<std::pair<int, double>> sorted_entries;
  for (const auto& [row, val] : column_result) {
    if (val != 0.0) {
      sorted_entries.emplace_back(row, val);
    }
  }
  std::ranges::sort(sorted_entries, {}, &std::pair<int, double>::first);

  int local_col = col_b - start_col;
  local_col_ptr[local_col + 1] = static_cast<int>(sorted_entries.size());

  for (const auto& [row, val] : sorted_entries) {
    local_rows.push_back(row);
    local_values.push_back(val);
  }
}

void SparseMatmulTask::MergeThreadResults(const std::vector<std::vector<double>>& thread_values,
                                          const std::vector<std::vector<int>>& thread_rows,
                                          const std::vector<std::vector<int>>& thread_col_ptrs,
                                          std::vector<double>& local_values, std::vector<int>& local_rows,
                                          std::vector<int>& local_col_ptr, int num_local_cols, int num_threads) {
  for (int local_col = 0; local_col < num_local_cols; ++local_col) {
    for (int t = 0; t < num_threads; ++t) {
      int start = thread_col_ptrs[t][local_col];
      int end = thread_col_ptrs[t][local_col + 1];
      for (int i = start; i < end; ++i) {
        local_values.push_back(thread_values[t][i]);
        local_rows.push_back(thread_rows[t][i]);
      }
    }
    local_col_ptr[local_col + 1] = static_cast<int>(local_values.size());
  }
}

bool SparseMatmulTask::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int base_cols = colsB / size;
  int extra_cols = colsB % size;
  int start_col = (rank < extra_cols) ? ((base_cols + 1) * rank)
                                      : (((base_cols + 1) * extra_cols) + (base_cols * (rank - extra_cols)));
  int end_col = start_col + ((rank < extra_cols) ? (base_cols + 1) : base_cols);
  int num_local_cols = end_col - start_col;

  std::vector<double> local_values;
  std::vector<int> local_rows;
  std::vector<int> local_col_ptr(num_local_cols + 1, 0);

  int num_threads = std::max(ppc::util::GetPPCNumThreads(), 1);

  std::vector<std::vector<double>> thread_values(num_threads);
  std::vector<std::vector<int>> thread_rows(num_threads);
  std::vector<std::vector<int>> thread_col_ptrs(num_threads, std::vector<int>(num_local_cols + 1, 0));

  auto worker = [&](int thread_id) {
    for (int col = start_col + thread_id; col < end_col; col += num_threads) {
      int local_col = col - start_col;
      size_t initial_size = thread_values[thread_id].size();
      ProcessColumn(col, start_col, thread_values[thread_id], thread_rows[thread_id], thread_col_ptrs[thread_id]);
      thread_col_ptrs[thread_id][local_col + 1] = static_cast<int>(thread_values[thread_id].size() - initial_size);
    }
    for (int lc = 1; lc <= num_local_cols; ++lc) {
      thread_col_ptrs[thread_id][lc] += thread_col_ptrs[thread_id][lc - 1];
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) {
    t.join();
  }

  MergeThreadResults(thread_values, thread_rows, thread_col_ptrs, local_values, local_rows, local_col_ptr,
                     num_local_cols, num_threads);

  std::vector<int> proc_start_cols(size);
  std::vector<int> proc_end_cols(size);
  proc_start_cols[rank] = start_col;
  proc_end_cols[rank] = end_col;

  boost::mpi::gather(world_, start_col, proc_start_cols, 0);
  boost::mpi::gather(world_, end_col, proc_end_cols, 0);

  std::vector<std::vector<double>> all_values;
  std::vector<std::vector<int>> all_rows;
  std::vector<std::vector<int>> all_col_ptrs;

  boost::mpi::gather(world_, local_values, all_values, 0);
  boost::mpi::gather(world_, local_rows, all_rows, 0);
  boost::mpi::gather(world_, local_col_ptr, all_col_ptrs, 0);

  if (rank == 0) {
    C_col_ptr.resize(colsB + 1, 0);
    for (int global_col = 0; global_col < colsB; ++global_col) {
      for (int proc = 0; proc < size; ++proc) {
        if (global_col >= proc_start_cols[proc] && global_col < proc_end_cols[proc]) {
          int local_col = global_col - proc_start_cols[proc];
          int start = all_col_ptrs[proc][local_col];
          int end = all_col_ptrs[proc][local_col + 1];
          for (int i = start; i < end; ++i) {
            C_values.push_back(all_values[proc][i]);
            C_row_indices.push_back(all_rows[proc][i]);
          }
          C_col_ptr[global_col + 1] = C_col_ptr[global_col] + (end - start);
          break;
        }
      }
    }
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all
