#include "all/karaseva_e_congrad/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace karaseva_e_congrad_mpi;

bool TestTaskMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  if (rank_ == 0) {
    global_size_ = static_cast<uint64_t>(task_data->inputs_count[1]);
  }
  MPI_Bcast(&global_size_, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  b_.resize(global_size_);
  x_.assign(global_size_, 0.0);

  std::vector<int> counts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);

  if (rank_ == 0) {
    // copy right-hand side vector b
    auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(b_ptr, b_ptr + global_size_, b_.begin());

    // determine how many rows of A each process gets
    const int rows_per_proc = static_cast<int>(global_size_) / world_size_;
    const int remainder = static_cast<int>(global_size_) % world_size_;
    int offset = 0;

    for (int i = 0; i < world_size_; ++i) {
      const int local_rows = rows_per_proc + (i < remainder ? 1 : 0);
      counts[i] = local_rows * static_cast<int>(global_size_);
      displs[i] = offset * static_cast<int>(global_size_);
      offset += local_rows;
    }
  }

  // scatter counts so each rank knows its local chunk size
  int local_chunk_size = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  a_local_.resize(local_chunk_size);

  double* a_ptr = nullptr;
  if (rank_ == 0) {
    a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  }
  // distribute rows of A to all processes
  MPI_Scatterv(a_ptr, counts.data(), displs.data(), MPI_DOUBLE, a_local_.data(), local_chunk_size, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  // broadcast b to all ranks
  MPI_Bcast(b_.data(), static_cast<int>(global_size_), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

bool TestTaskMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  bool validation_result = true;
  if (rank_ == 0) {
    global_size_ = static_cast<uint64_t>(task_data->inputs_count[1]);
    validation_result =
        (task_data->inputs_count[0] == global_size_ * global_size_) && (task_data->outputs_count[0] == global_size_);
  }
  int validation_int = validation_result ? 1 : 0;
  MPI_Bcast(&validation_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return validation_int != 0;
}

namespace {

void ParallelInit(std::vector<double>& r, std::vector<double>& p, const std::vector<double>& b) {
  const int n = static_cast<int>(b.size());
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    r[i] = b[i];
    p[i] = r[i];
  }
}

double ParallelDotProduct(const std::vector<double>& a, const std::vector<double>& b) {
  double local_sum = 0.0;
  const int n = static_cast<int>(a.size());
#pragma omp parallel for reduction(+ : local_sum)
  for (int i = 0; i < n; ++i) {
    local_sum += a[i] * b[i];
  }
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_sum;
}

void MatrixVectorMultiply(const std::vector<double>& a_local, std::vector<double>& ap, const std::vector<double>& p,
                          int local_rows, int n) {
  std::vector<double> local_ap(local_rows, 0.0);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#pragma omp parallel for
  for (int i = 0; i < local_rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a_local[(i * n) + j] * p[j];
    }
    local_ap[i] = sum;
  }

  std::vector<int> recv_counts;
  std::vector<int> displs;
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    recv_counts.resize(world_size);
    displs.resize(world_size, 0);
  }

  MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  MPI_Gatherv(local_ap.data(), local_rows, MPI_DOUBLE, ap.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  MPI_Bcast(ap.data(), static_cast<int>(ap.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

}  // namespace

bool TestTaskMPI::RunImpl() {
  std::vector<double> r(global_size_);
  std::vector<double> p(global_size_);
  std::vector<double> ap(global_size_);

  ParallelInit(r, p, b_);
  double rs_old = ParallelDotProduct(r, r);
  const double tolerance = 1e-10;
  const int max_iter = static_cast<int>(global_size_);

  for (int it = 0; it < max_iter; ++it) {
    const int local_rows = static_cast<int>(a_local_.size()) / static_cast<int>(global_size_);
    MatrixVectorMultiply(a_local_, ap, p, local_rows, static_cast<int>(global_size_));

    const double p_ap = ParallelDotProduct(p, ap);
    if (std::abs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(global_size_); ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    const double rs_new = ParallelDotProduct(r, r);
    if (rs_new < tolerance) {
      break;
    }
    const double beta = rs_new / rs_old;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(global_size_); ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rs_old = rs_new;
  }
  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  MPI_Bcast(x_.data(), static_cast<int>(global_size_), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (task_data->outputs[0] != nullptr) {
    auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    for (size_t i = 0; i < x_.size(); ++i) {
      x_ptr[i] = x_[i];
    }
  }
  return true;
}