#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

void yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS::InsertElement(int row_idx, ComplexNum val, int col_idx) {
  for (int j = row_ptrs[row_idx]; j < row_ptrs[row_idx + 1]; ++j) {
    if (column_idxs[j] == col_idx) {
      non_zero_elems[j] += val;
      return;
    }
  }
  column_idxs.emplace_back(col_idx);
  non_zero_elems.emplace_back(val);
  for (int i = row_idx + 1; i <= total_rows; ++i) {
    row_ptrs[i]++;
  }
}

void yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS::DisplayMatrix(
    const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& matrix) {
  for (int i = 0; i < matrix.total_rows; ++i) {
    for (int j = matrix.row_ptrs[i]; j < matrix.row_ptrs[i + 1]; ++j) {
      std::cout << "Element at (" << i << ", " << matrix.column_idxs[j] << ") = " << matrix.non_zero_elems[j] << '\n';
    }
  }
}

bool yasakova_t_sparse_matrix_mult_all::AreClose(const ComplexNum& a, const ComplexNum& b, double epsilon) {
  return std::abs(a.real() - b.real()) < epsilon && std::abs(a.imag() - b.imag()) < epsilon;
}

void yasakova_t_sparse_matrix_mult_all::AppendElement(std::vector<ElementPosition>& data, int row_idx, int col_idx,
                                                      const ComplexNum& val) {
  data.push_back({row_idx, col_idx, val});
}

std::vector<ComplexNum> yasakova_t_sparse_matrix_mult_all::ConvertToDense(const SparseMatrixCRS& sparse_mat) {
  std::vector<ComplexNum> res = {};
  res.reserve(5 + sparse_mat.non_zero_elems.size() + sparse_mat.column_idxs.size() + sparse_mat.row_ptrs.size());
  res.emplace_back((double)sparse_mat.total_rows);
  res.emplace_back((double)sparse_mat.total_cols);
  res.emplace_back((double)sparse_mat.non_zero_elems.size());
  res.emplace_back((double)sparse_mat.column_idxs.size());
  res.emplace_back((double)sparse_mat.row_ptrs.size());
  for (unsigned int i = 0; i < (unsigned int)sparse_mat.non_zero_elems.size(); i++) {
    res.emplace_back(sparse_mat.non_zero_elems[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)sparse_mat.column_idxs.size(); i++) {
    res.emplace_back(sparse_mat.column_idxs[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)sparse_mat.row_ptrs.size(); i++) {
    res.emplace_back(sparse_mat.row_ptrs[i]);
  }
  return res;
}

bool yasakova_t_sparse_matrix_mult_all::CompareMatrices(const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& a,
                                                        const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& b) {
  if (a.total_cols != b.total_cols || a.total_rows != b.total_rows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)a.total_rows; ++i) {
    unsigned int this_row_start = a.row_ptrs[i];
    unsigned int this_row_end = a.row_ptrs[i + 1];
    unsigned int other_row_start = b.row_ptrs[i];
    unsigned int other_row_end = b.row_ptrs[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (a.column_idxs[j] == b.column_idxs[k] && AreClose(a.non_zero_elems[j], b.non_zero_elems[k], 0.000001)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS yasakova_t_sparse_matrix_mult_all::ConvertToSparse(
    std::vector<ComplexNum>& vec) {
  SparseMatrixCRS res;
  res.total_rows = (int)vec[0].real();
  res.total_cols = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  res.non_zero_elems.reserve(values_size);
  res.column_idxs.reserve(col_indices_size);
  res.row_ptrs.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    res.non_zero_elems.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    res.column_idxs.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    res.row_ptrs.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return res;
}

yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS yasakova_t_sparse_matrix_mult_all::ConstructResultMatrix(
    const std::vector<yasakova_t_sparse_matrix_mult_all::ElementPosition>& all_results, int a_num_rows,
    int b_num_cols) {
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(a_num_rows, b_num_cols);
  std::map<std::pair<int, int>, ComplexNum> result_map;

  for (const auto& rv : all_results) {
    result_map[{rv.row_idx, rv.col_idx}] += rv.val;
  }

  c.row_ptrs.resize(a_num_rows + 1);
  c.non_zero_elems.reserve(result_map.size());
  c.column_idxs.reserve(result_map.size());

  c.row_ptrs[0] = 0;

  for (int i = 0; i < a_num_rows; ++i) {
    for (auto& kv : result_map) {
      if (kv.first.first == i && kv.second != ComplexNum(0)) {
        c.column_idxs.push_back(kv.first.second);
        c.non_zero_elems.push_back(kv.second);
      }
    }
    c.row_ptrs[i + 1] = static_cast<int>(c.column_idxs.size());

    for (auto it = result_map.begin(); it != result_map.end();) {
      if (it->first.first == i) {
        it = result_map.erase(it);
      } else {
        ++it;
      }
    }
  }
  return c;
}

void yasakova_t_sparse_matrix_mult_all::TestTaskALL::ProcessRowsRange(int start_row, int end_row,
                                                                      std::vector<ElementPosition>& local_results) {
  for (int i = start_row; i < end_row; ++i) {
    for (int j_idx = matrix_a_.row_ptrs[i]; j_idx < matrix_a_.row_ptrs[i + 1]; ++j_idx) {
      int col_a = matrix_a_.column_idxs[j_idx];
      ComplexNum value_a = matrix_a_.non_zero_elems[j_idx];

      for (int k_idx = matrix_b_.row_ptrs[col_a]; k_idx < matrix_b_.row_ptrs[col_a + 1]; ++k_idx) {
        int col_b = matrix_b_.column_idxs[k_idx];
        ComplexNum value_b = matrix_b_.non_zero_elems[k_idx];

#pragma omp critical
        AppendElement(local_results, i, col_b, value_a * value_b);
      }
    }
  }
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<ComplexNum*>(task_data->inputs[0]);
  input_data_ = std::vector<ComplexNum>(in_ptr, in_ptr + input_size);
  std::vector<ComplexNum> matrix_a = {};
  std::vector<ComplexNum> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(input_data_[2].real() + input_data_[3].real() + input_data_[4].real()));
  matrix_b.reserve(input_data_.size() -
                   (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real()));
  for (unsigned int i = 0;
       i < (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real()); i++) {
    matrix_a.emplace_back(input_data_[i]);
  }
  for (auto i = (unsigned int)(5 + input_data_[2].real() + input_data_[3].real() + input_data_[4].real());
       i < (unsigned int)input_data_.size(); i++) {
    matrix_b.emplace_back(input_data_[i]);
  }
  matrix_a_ = ConvertToSparse(matrix_a);
  matrix_b_ = ConvertToSparse(matrix_b);
  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<ComplexNum*>(task_data->inputs[0]);
    std::vector<ComplexNum> vec = std::vector<ComplexNum>(in_ptr, in_ptr + input_size);
    return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
  }
  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int a_num_rows = matrix_a_.total_rows;
  int a_num_cols = matrix_a_.total_cols;
  int b_num_rows = matrix_b_.total_rows;
  int b_num_cols = matrix_b_.total_cols;

  MPI_Bcast(&a_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int row_per_proc = a_num_rows / size;
  int remainder = a_num_rows % size;

  int start_row = (rank * row_per_proc) + std::min(rank, remainder);
  int end_row = start_row + row_per_proc + (rank < remainder ? 1 : 0);

  std::vector<ElementPosition> local_results;
  local_results.reserve((end_row - start_row) * b_num_cols);

  // Установка числа потоков OpenMP
  int num_threads = ppc::util::GetPPCNumThreads();
  omp_set_num_threads(num_threads);

#pragma omp parallel
  {
    std::vector<ElementPosition> thread_local_results;
    thread_local_results.reserve((end_row - start_row) * b_num_cols / num_threads);

#pragma omp for schedule(static)
    for (int i = start_row; i < end_row; ++i) {
      for (int j_idx = matrix_a_.row_ptrs[i]; j_idx < matrix_a_.row_ptrs[i + 1]; ++j_idx) {
        int col_a = matrix_a_.column_idxs[j_idx];
        ComplexNum value_a = matrix_a_.non_zero_elems[j_idx];

        for (int k_idx = matrix_b_.row_ptrs[col_a]; k_idx < matrix_b_.row_ptrs[col_a + 1]; ++k_idx) {
          int col_b = matrix_b_.column_idxs[k_idx];
          ComplexNum value_b = matrix_b_.non_zero_elems[k_idx];

          AppendElement(thread_local_results, i, col_b, value_a * value_b);
        }
      }
    }

#pragma omp critical
    local_results.insert(local_results.end(), thread_local_results.begin(), thread_local_results.end());
  }

  // Сбор результатов с помощью MPI
  int local_size_bytes = static_cast<int>(local_results.size() * sizeof(ElementPosition));
  std::vector<int> recv_counts(size);
  MPI_Gather(&local_size_bytes, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> displs(size);
    if (!displs.empty()) {
      displs[0] = 0;
      for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
      }
    }
    int total_bytes = displs[size - 1] + recv_counts[size - 1];

    std::vector<char> recv_buffer(total_bytes);
    std::vector<ElementPosition> send_buffer(local_results.begin(), local_results.end());

    MPI_Gatherv(send_buffer.data(), static_cast<int>(send_buffer.size() * sizeof(ElementPosition)), MPI_BYTE,
                recv_buffer.data(), recv_counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<ElementPosition> all_results;
    all_results.reserve(total_bytes / sizeof(ElementPosition));
    auto* ptr = reinterpret_cast<ElementPosition*>(recv_buffer.data());
    size_t count_coords = total_bytes / sizeof(ElementPosition);
    for (size_t i = 0; i < count_coords; ++i) {
      all_results.push_back(ptr[i]);
    }

    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(a_num_rows, b_num_cols);
    c = ConstructResultMatrix(all_results, a_num_rows, b_num_cols);
    output_data_ = ConvertToDense(c);
  } else {
    MPI_Gatherv(local_results.data(), static_cast<int>(local_results.size() * sizeof(ElementPosition)), MPI_BYTE,
                nullptr, nullptr, nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_data_.size(); i++) {
      reinterpret_cast<ComplexNum*>(task_data->outputs[0])[i] = output_data_[i];
    }
  }
  return true;
}