#pragma once

#include <omp.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using ComplexNum = std::complex<double>;

namespace yasakova_t_sparse_matrix_mult_all {

struct ElementPosition {
  int row_idx;
  int col_idx;
  ComplexNum val;
};

void AppendElement(std::vector<ElementPosition>& data, int row_idx, int col_idx, const ComplexNum& val);

struct SparseMatrixCRS {
  std::vector<ComplexNum> non_zero_elems;
  std::vector<int> column_idxs;
  std::vector<int> row_ptrs;
  int total_rows;
  int total_cols;
  SparseMatrixCRS() : non_zero_elems({}), column_idxs({}), row_ptrs({}), total_rows(0), total_cols(0) {};
  SparseMatrixCRS(int rows, int cols) : total_rows(rows), total_cols(cols) { row_ptrs.resize(rows + 1, 0); }
  void InsertElement(int row_idx, ComplexNum val, int col_idx);
  SparseMatrixCRS(const SparseMatrixCRS& other) = default;
  SparseMatrixCRS& operator=(const SparseMatrixCRS& other) = default;
  static void DisplayMatrix(const SparseMatrixCRS& matrix);
};

std::vector<ComplexNum> ConvertToDense(const SparseMatrixCRS& sparse_mat);
SparseMatrixCRS ConvertToSparse(std::vector<ComplexNum>& vec);
SparseMatrixCRS ConstructResultMatrix(
    const std::vector<yasakova_t_sparse_matrix_mult_all::ElementPosition>& all_results, int a_num_rows, int b_num_cols);
bool CompareMatrices(const SparseMatrixCRS& a, const SparseMatrixCRS& b);
bool AreClose(const ComplexNum& a, const ComplexNum& b, double epsilon);

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<ComplexNum> input_data_, output_data_;
  SparseMatrixCRS matrix_a_, matrix_b_;
  boost::mpi::communicator world_;
  void ProcessRowsRange(int start_row, int end_row, std::vector<ElementPosition>& local_results);
};

}  // namespace yasakova_t_sparse_matrix_mult_all