#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/complex.hpp>  // NOLINT(misc-include-cleaner)
#include <boost/serialization/vector.hpp>   // NOLINT(misc-include-cleaner)
#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_sparse_matrix_mult_complex_ccs_all {

using Complex = std::complex<double>;

struct SparseMatrixCCS {
  std::vector<Complex> values;
  std::vector<int> row_indices;
  std::vector<int> col_offsets;
  int rows;
  int cols;
  int nnz;

  SparseMatrixCCS(int r = 0, int c = 0, int n = 0) : rows(r), cols(c), nnz(n) {
    values.resize(nnz);
    row_indices.resize(nnz);
    col_offsets.resize(cols + 1, 0);
  }

  friend class boost::serialization::access;

  // clang-format off
  // NOLINTBEGIN(*)
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar & rows;
    ar & cols;
    ar & nnz;
    ar & values;
    ar & row_indices;
    ar & col_offsets;
  }
  // NOLINTEND(*)
  // clang-format on
};

class SparseMatrixMultComplexCCS : public ppc::core::Task {
 public:
  explicit SparseMatrixMultComplexCCS(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixCCS* matrix1_;
  SparseMatrixCCS* matrix2_;
  SparseMatrixCCS result_;
  boost::mpi::communicator world_;

  void ComputeColumn(int col_idx, std::vector<std::pair<Complex, int>>& column_data);
  Complex ComputeElement(int row_idx, int col_start2, int col_end2);
  Complex ComputeContribution(int row_idx, int k, int col_start1, int col_end1, int col_start2, int col_end2);
  void ProcessColumnRange(int start_col, int end_col, std::vector<std::vector<std::pair<Complex, int>>>& column_results,
                          const std::vector<int>& col_indices);
  static void CollectLocalResults(const std::vector<std::vector<std::pair<Complex, int>>>& column_results,
                                  int start_col, int end_col, std::vector<Complex>& local_values,
                                  std::vector<int>& local_row_indices, std::vector<int>& local_col_offsets,
                                  int& local_nnz);
  void GatherGlobalResults(int rank, int size, int total_cols, int local_nnz, const std::vector<Complex>& local_values,
                           const std::vector<int>& local_row_indices,
                           const std::vector<std::vector<std::pair<Complex, int>>>& column_results, int start_col,
                           int end_col);
};

}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_all