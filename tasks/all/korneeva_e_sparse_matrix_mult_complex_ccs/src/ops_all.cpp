#include "all/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_gatherv.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace korneeva_e_sparse_matrix_mult_complex_ccs_all {

namespace {
void DistributeColumns(int rank, int size, int total_cols, int& start_col, int& end_col) {
  if (size <= 0) {
    start_col = total_cols;
    end_col = total_cols;
    return;
  }
  int cols_per_process = total_cols / size;
  int remaining_cols = total_cols % size;
  start_col = (rank * cols_per_process) + std::min(rank, remaining_cols);
  int extra_col = rank < remaining_cols ? 1 : 0;
  end_col = start_col + cols_per_process + extra_col;
  if (start_col >= total_cols || end_col > total_cols) {
    start_col = total_cols;
    end_col = total_cols;
  }
}
}  // namespace

bool SparseMatrixMultComplexCCS::PreProcessingImpl() {
  if (world_.rank() == 0) {
    matrix1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
    matrix2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
    result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);
  } else {
    matrix1_ = new SparseMatrixCCS();
    matrix2_ = new SparseMatrixCCS();
    result_ = SparseMatrixCCS(0, 0, 0);
  }
  return true;
}

bool SparseMatrixMultComplexCCS::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1) {
      return false;
    }

    auto* m1 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
    auto* m2 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);

    return m1 != nullptr && m2 != nullptr && m1->cols == m2->rows && m1->rows > 0 && m1->cols > 0 && m2->rows > 0 &&
           m2->cols > 0;
  }
  return true;
}

void SparseMatrixMultComplexCCS::ProcessColumnRange(int start_col, int end_col,
                                                    std::vector<std::vector<std::pair<Complex, int>>>& column_results,
                                                    const std::vector<int>& col_indices) {
  int num_threads = std::max(1, std::min(ppc::util::GetPPCNumThreads(), end_col - start_col));
  if ((end_col - start_col) < num_threads * 2) {
    for (int j = start_col; j < end_col; ++j) {
      ComputeColumn(col_indices[j - start_col], column_results[j - start_col]);
    }
  } else {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int cols_per_thread = (end_col - start_col) / num_threads;
    int remaining_thread_cols = (end_col - start_col) % num_threads;
    int thread_start = start_col;

    auto compute_range = [&](int thread_start, int thread_end) {
      for (int j = thread_start; j < thread_end; ++j) {
        ComputeColumn(col_indices[j - start_col], column_results[j - start_col]);
      }
    };

    for (int i = 0; i < num_threads; ++i) {
      int cols = cols_per_thread + (i < remaining_thread_cols ? 1 : 0);
      int thread_end = thread_start + cols;
      threads.emplace_back(compute_range, thread_start, thread_end);
      thread_start = thread_end;
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
}

void SparseMatrixMultComplexCCS::CollectLocalResults(
    const std::vector<std::vector<std::pair<Complex, int>>>& column_results, int start_col, int end_col,
    std::vector<Complex>& local_values, std::vector<int>& local_row_indices, std::vector<int>& local_col_offsets,
    int& local_nnz) {
  local_nnz = 0;
  local_values.clear();
  local_row_indices.clear();
  local_col_offsets.assign(end_col - start_col + 1, 0);
  for (int j = 0; j < end_col - start_col; ++j) {
    const auto& col_data = column_results[j];
    for (const auto& [value, row_idx] : col_data) {
      local_values.push_back(value);
      local_row_indices.push_back(row_idx);
    }
    local_nnz += static_cast<int>(col_data.size());
    local_col_offsets[j + 1] = local_nnz;
  }
}

void SparseMatrixMultComplexCCS::GatherGlobalResults(
    int rank, int size, int total_cols, int local_nnz, const std::vector<Complex>& local_values,
    const std::vector<int>& local_row_indices, const std::vector<std::vector<std::pair<Complex, int>>>& column_results,
    int start_col, int end_col) {
  std::vector<int> all_nnz;
  boost::mpi::all_gather(world_, local_nnz, all_nnz);
  int total_nnz = std::accumulate(all_nnz.begin(), all_nnz.end(), 0);

  std::vector<int> displacements(size);
  if (!displacements.empty()) {
    displacements[0] = 0;
    for (size_t i = 1; i < displacements.size(); ++i) {
      displacements[i] = displacements[i - 1] + all_nnz[i - 1];
    }
  }

  result_.values.resize(total_nnz);
  result_.row_indices.resize(total_nnz);
  result_.col_offsets.resize(total_cols + 1, 0);
  result_.nnz = total_nnz;

  if (total_nnz > 0) {
    boost::mpi::all_gatherv(world_, local_values, result_.values, all_nnz, displacements);
    boost::mpi::all_gatherv(world_, local_row_indices, result_.row_indices, all_nnz, displacements);
  }

  std::vector<int> local_col_counts(total_cols, 0);
  for (int j = 0; j < end_col - start_col; ++j) {
    local_col_counts[start_col + j] = static_cast<int>(column_results[j].size());
  }

  std::vector<int> global_col_counts(total_cols);
  boost::mpi::reduce(world_, local_col_counts, global_col_counts, std::plus<>{}, 0);

  if (rank == 0) {
    result_.col_offsets[0] = 0;
    for (int j = 0; j < total_cols; ++j) {
      result_.col_offsets[j + 1] = result_.col_offsets[j] + global_col_counts[j];
    }
  }
  boost::mpi::broadcast(world_, result_.col_offsets, 0);
}

bool SparseMatrixMultComplexCCS::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, *matrix1_, 0);
  boost::mpi::broadcast(world_, *matrix2_, 0);

  result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);

  int total_cols = matrix2_->cols;
  int start_col = 0;
  int end_col = 0;
  DistributeColumns(rank, size, total_cols, start_col, end_col);

  std::vector<std::vector<std::pair<Complex, int>>> column_results(end_col - start_col);
  std::vector<int> col_indices(end_col - start_col);
  std::iota(col_indices.begin(), col_indices.end(), start_col);

  if (end_col > start_col) {
    ProcessColumnRange(start_col, end_col, column_results, col_indices);
  }

  std::vector<Complex> local_values;
  std::vector<int> local_row_indices;
  std::vector<int> local_col_offsets;
  int local_nnz = 0;
  CollectLocalResults(column_results, start_col, end_col, local_values, local_row_indices, local_col_offsets,
                      local_nnz);

  GatherGlobalResults(rank, size, total_cols, local_nnz, local_values, local_row_indices, column_results, start_col,
                      end_col);

  boost::mpi::broadcast(world_, result_, 0);

  return true;
}

Complex SparseMatrixMultComplexCCS::ComputeElement(int row_idx, int col_start2, int col_end2) {
  Complex sum(0.0, 0.0);
  for (int k = 0; k < matrix1_->cols; ++k) {
    int col_start1 = matrix1_->col_offsets[k];
    int col_end1 = matrix1_->col_offsets[k + 1];
    sum += ComputeContribution(row_idx, k, col_start1, col_end1, col_start2, col_end2);
  }
  return sum;
}

Complex SparseMatrixMultComplexCCS::ComputeContribution(int row_idx, int k, int col_start1, int col_end1,
                                                        int col_start2, int col_end2) {
  Complex contribution(0.0, 0.0);
  for (int p = col_start1; p < col_end1; ++p) {
    if (matrix1_->row_indices[p] != row_idx) {
      continue;
    }
    for (int q = col_start2; q < col_end2; ++q) {
      if (matrix2_->row_indices[q] != k) {
        continue;
      }
      contribution += matrix1_->values[p] * matrix2_->values[q];
    }
  }
  return contribution;
}

void SparseMatrixMultComplexCCS::ComputeColumn(int col_idx, std::vector<std::pair<Complex, int>>& column_data) {
  if (col_idx >= matrix2_->cols) {
    return;
  }
  int col_start2 = matrix2_->col_offsets[col_idx];
  int col_end2 = matrix2_->col_offsets[col_idx + 1];
  if (col_start2 < 0 || col_end2 > matrix2_->nnz || col_start2 > col_end2) {
    return;
  }

  column_data.resize(0);
  column_data.reserve(std::min(matrix1_->rows, matrix2_->nnz));
  for (int i = 0; i < matrix1_->rows; ++i) {
    Complex sum = ComputeElement(i, col_start2, col_end2);
    if (std::abs(sum.real()) > 1e-10 || std::abs(sum.imag()) > 1e-10) {
      column_data.emplace_back(sum, i);
    }
  }
}

bool SparseMatrixMultComplexCCS::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<SparseMatrixCCS*>(task_data->outputs[0]) = result_;
  }
  return true;
}

}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_all