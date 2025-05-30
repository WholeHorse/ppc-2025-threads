#include "all/lavrentiev_A_CCS/include/ops_stl.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<std::pair<double, int>>> new_elements_and_rows(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements_and_rows[sparse.rows[counter]].emplace_back(sparse.elements[counter], i);
      counter++;
    }
  }
  elements.reserve(counter);
  rows.reserve(counter);
  for (int i = 0; i < static_cast<int>(new_elements_and_rows.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements_and_rows[i].size()); ++j) {
      elements.emplace_back(new_elements_and_rows[i][j].first);
      rows.emplace_back(new_elements_and_rows[i][j].second);
    }
    i > 0 ? columns_sum.emplace_back(new_elements_and_rows[i].size() + columns_sum[i - 1])
          : columns_sum.emplace_back(new_elements_and_rows[i].size());
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_all::CCSALL::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  return index == 0 ? 0 : columns_sum[index] - GetElementsCount(index, columns_sum);
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::MatMul(const Sparse &matrix1, const Sparse &matrix2,
                                                                  int interval_begin, int interval_end) {
  Sparse temporary_matrix;
  int resize_data = static_cast<int>(matrix2.columnsSum.size() * matrix1.columnsSum.size());
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  temporary_matrix.columnsSum.resize(matrix2.size.second);
  temporary_matrix.elements.resize(resize_data);
  temporary_matrix.rows.resize(resize_data);
  auto accumulate = [&](int i_index, int j_index) {
    double sum = 0.0;
    for (int x = 0; x < GetElementsCount(j_index, matrix1.columnsSum); x++) {
      for (int y = 0; y < GetElementsCount(i_index, matrix2.columnsSum); y++) {
        if (matrix1.rows[CalculateStartIndex(j_index, matrix1.columnsSum) + x] ==
            matrix2.rows[CalculateStartIndex(i_index, matrix2.columnsSum) + y]) {
          sum += matrix1.elements[x + CalculateStartIndex(j_index, matrix1.columnsSum)] *
                 matrix2.elements[y + CalculateStartIndex(i_index, matrix2.columnsSum)];
        }
      }
    }
    return sum;
  };
  auto matrix_multiplicator = [&](int begin, int end) {
    for (int i = begin; i != end; ++i) {
      if (temporary_matrix.columnsSum[i] == 0) {
        temporary_matrix.columnsSum[i]++;
      }
      for (int j = 0; j < static_cast<int>(matrix1.columnsSum.size()); ++j) {
        double s = accumulate(i, j);
        if (s != 0) {
          temporary_matrix.elements[(i * matrix2.size.second) + j] = s;
          temporary_matrix.rows[(i * matrix2.size.second) + j] = j + 1;
          temporary_matrix.columnsSum[i]++;
        }
      }
    }
  };
  int thread_data_amount = (interval_end - interval_begin) / ppc::util::GetPPCNumThreads();
  for (size_t i = 0; i < threads.size(); ++i) {
    if (i != threads.size() - 1) {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount));
    } else {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount) +
                                   ((interval_end - interval_begin) % ppc::util::GetPPCNumThreads()));
    }
  }
  std::ranges::for_each(threads, [&](std::thread &thread) { thread.join(); });
  return {.size = temporary_matrix.size,
          .elements = temporary_matrix.elements,
          .rows = temporary_matrix.rows,
          .columnsSum = temporary_matrix.columnsSum};
}

int lavrentiev_a_ccs_all::CCSALL::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_all::CCSALL::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

void lavrentiev_a_ccs_all::CCSALL::GetDisplacements() {
  displ_.resize(world_.size());
  int amount = static_cast<int>(B_.columnsSum.size()) % world_.size();
  int count = static_cast<int>(B_.columnsSum.size()) / world_.size();
  if (static_cast<int>(B_.columnsSum.size()) == 0) {
    return;
  }
  for (int i = 0; i < world_.size(); ++i) {
    if (i != 0) {
      displ_[i] = count + displ_[i - 1];
      if (amount != 0) {
        displ_[i]++;
        amount--;
      }
    }
  }
  displ_.emplace_back(static_cast<int>(B_.columnsSum.size()));
}

bool lavrentiev_a_ccs_all::CCSALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
    B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
    if (IsEmpty()) {
      return true;
    }
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    auto am = std::vector<double>(in_ptr, in_ptr + (A_.size.first * A_.size.second));
    A_ = ConvertToSparse(A_.size, am);
    A_ = Transpose(A_);
    auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
    auto bm = std::vector<double>(in_ptr2, in_ptr2 + (B_.size.first * B_.size.second));
    B_ = ConvertToSparse(B_.size, bm);
    GetDisplacements();
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

void lavrentiev_a_ccs_all::CCSALL::CollectSizes() {
  if (world_.rank() != 0) {
    world_.send(0, 1, static_cast<int>(Process_data_.columnsSum.size()));
  } else {
    sum_sizes_.resize(world_.size(), static_cast<int>(Process_data_.columnsSum.size()));
    for (int i = 1; i < world_.size(); ++i) {
      world_.recv(i, 1, sum_sizes_[i]);
    }
  }
}
bool lavrentiev_a_ccs_all::CCSALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
           task_data->inputs_count[0] == task_data->inputs_count[3] &&
           task_data->inputs_count[1] == task_data->inputs_count[2];
  }
  return true;
}

void lavrentiev_a_ccs_all::CCSALL::CollectData() {
  sending_data_ = std::move(Process_data_.elements);
  sending_data_.resize(Process_data_.columnsSum.size() + (2 * resize_data_));
  for (int i = 0; i < resize_data_; i++) {
    sending_data_[i + resize_data_] = static_cast<double>(Process_data_.rows[i]);
    if (i < static_cast<int>(Process_data_.columnsSum.size())) {
      sending_data_[i + (2 * resize_data_)] = static_cast<double>(Process_data_.columnsSum[i]);
    }
  }
}

bool lavrentiev_a_ccs_all::CCSALL::RunImpl() {
  boost::mpi::broadcast(world_, displ_, 0);
  boost::mpi::broadcast(world_, A_, 0);
  boost::mpi::broadcast(world_, B_, 0);
  if (displ_.empty()) {
    return true;
  }
  resize_data_ = static_cast<int>(B_.columnsSum.size() * A_.columnsSum.size());
  Process_data_ = MatMul(A_, B_, displ_[world_.rank()], displ_[world_.rank() + 1]);
  CollectSizes();
  CollectData();
  if (world_.rank() == 0) {
    Answer_.columnsSum.clear();
    Answer_.rows.clear();
    Answer_.elements.clear();
    Answer_.elements.reserve(resize_data_);
    Answer_.rows.reserve(resize_data_);
    Answer_.columnsSum.reserve(resize_data_);
    std::vector<double> data_reader((resize_data_ * world_.size() * 2) +
                                    std::accumulate(sum_sizes_.begin(), sum_sizes_.end(), 0));
    std::vector<int> size(world_.size(), resize_data_ * 2);
    for (int i = 0; i < world_.size(); ++i) {
      size[i] += sum_sizes_[i];
    }
    boost::mpi::gatherv(world_, sending_data_, data_reader.data(), size, 0);
    int process_data_num = 0;
    int sum_by_elements_count = sum_sizes_.front();
    int past_data = 0;
    for (int i = 0; i < static_cast<int>(data_reader.size()); ++i) {
      if (i == (resize_data_ * 2 * (process_data_num + 1)) + sum_by_elements_count) {
        past_data += (resize_data_ * 2) + sum_sizes_[process_data_num];
        process_data_num++;
        sum_by_elements_count += sum_sizes_[process_data_num];
      }
      AddData(data_reader, past_data, i);
    }
    for (size_t i = 1; i < Answer_.columnsSum.size(); ++i) {
      Answer_.columnsSum[i] = Answer_.columnsSum[i] + Answer_.columnsSum[i - 1];
    }
    Answer_.size.first = B_.size.second;
    Answer_.size.second = B_.size.second;
  } else {
    boost::mpi::gatherv(world_, sending_data_, 0);
  }
  return true;
}

void lavrentiev_a_ccs_all::CCSALL::AddData(const std::vector<double> &data, int past_data, int index) {
  if (index - ((resize_data_ * 2) + past_data) < 0) {
    if (index - (resize_data_ + past_data) < 0 && data[index] != 0.0) {
      Answer_.elements.emplace_back(data[index]);
    } else {
      if (data[index] != 0.0) {
        Answer_.rows.emplace_back(data[index] - 1);
      }
    }
  } else {
    if (data[index] != 0.0) {
      Answer_.columnsSum.emplace_back(data[index] - 1);
    }
  }
}

bool lavrentiev_a_ccs_all::CCSALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto answer = ConvertFromSparse(Answer_);
    if (!answer.empty()) {
      std::ranges::copy(answer, reinterpret_cast<double *>(task_data->outputs[0]));
    }
  }
  return true;
}