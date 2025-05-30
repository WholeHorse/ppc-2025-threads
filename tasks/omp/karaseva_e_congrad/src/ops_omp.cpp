#include "omp/karaseva_e_congrad/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool karaseva_e_congrad_omp::TestTaskOpenMP::PreProcessingImpl() {
  // Read input dimensions and copy data from task_data to internal buffers
  size_ = task_data->inputs_count[1];
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Initialize matrix A (size_ x size_) and vectors b/x
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);  // Initial solution guess

  return true;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::ValidationImpl() {
  // Verify that input matrix is square (N x N) and output has correct size (N)
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::RunImpl() {
  // Conjugate gradient working vectors
  std::vector<double> r(size_);   // Residual
  std::vector<double> p(size_);   // Search direction
  std::vector<double> ap(size_);  // Matrix-vector product A*p

  // Initialize residual r = b - A*x (x is initially zero)
  // and initial search direction p = r
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size_); ++i) {
    r[i] = b_[i];  // Since x is zero, r = b - 0
    p[i] = r[i];
  }

  // Calculate initial residual squared norm
  double rs_old = 0.0;
#pragma omp parallel for reduction(+ : rs_old)
  for (int i = 0; i < static_cast<int>(size_); ++i) {
    rs_old += r[i] * r[i];
  }

  const double tolerance = 1e-10;       // Convergence threshold
  const size_t max_iterations = size_;  // Worst-case iterations

  // Main conjugate gradient loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Compute matrix-vector product: ap = A * p
    // Using nested parallelism for inner loop
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      double temp = 0.0;
#pragma omp parallel for reduction(+ : temp)
      for (int j = 0; j < static_cast<int>(size_); ++j) {
        temp += A_[(i * size_) + j] * p[j];
      }
      ap[i] = temp;
    }

    // Compute p^T * A * p for alpha calculation
    double p_ap = 0.0;
#pragma omp parallel for reduction(+ : p_ap)
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      p_ap += p[i] * ap[i];
    }

    // Early exit if denominator becomes unstable
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

    // Update solution and residual vectors
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    // Compute new residual norm
    double rs_new = 0.0;
#pragma omp parallel for reduction(+ : rs_new)
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      rs_new += r[i] * r[i];
    }

    // Check convergence condition
    if (rs_new < tolerance * tolerance) {
      break;
    }

    // Update search direction using Polak-Ribiere formula
    const double beta = rs_new / rs_old;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;  // Update residual norm for next iteration
  }

  return true;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::PostProcessingImpl() {
  // Write results back to task_data output buffer
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(x_.size()); ++i) {
    x_ptr[i] = x_[i];
  }
  return true;
}