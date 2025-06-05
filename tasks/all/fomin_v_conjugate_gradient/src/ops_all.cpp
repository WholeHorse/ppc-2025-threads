#include "all/fomin_v_conjugate_gradient/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double fomin_v_conjugate_gradient::FominVConjugateGradientAll::DotProduct(const boost::mpi::communicator& world,
                                                                          const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  double local_sum = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, a.size()), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double sum) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          sum += a[i] * b[i];
        }
        return sum;
      },
      std::plus<>());

  double global_sum = 0.0;
  all_reduce(world, local_sum, global_sum, std::plus<>());
  return global_sum;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::MatrixVectorMultiply(
    const std::vector<double>& x) const {
  std::vector<double> local_result(rows_per_proc_, 0.0);

  tbb::parallel_for(tbb::blocked_range<int>(0, rows_per_proc_), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += local_a_[(i * n) + j] * x[j];
      }
      local_result[i] = sum;
    }
  });

  std::vector<double> global_result;
  if (world_.rank() == 0) {
    global_result.resize(n);
  }

  std::vector<int> sizes(world_.size());
  std::vector<int> displs(world_.size(), 0);
  int remainder = n % world_.size();
  for (int i = 0; i < world_.size(); ++i) {
    sizes[i] = (n / world_.size()) + (i < remainder ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }
  gatherv(world_, local_result.data(), rows_per_proc_, global_result.data(), sizes, displs, 0);
  broadcast(world_, global_result, 0);
  return global_result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::vector<double> input(in_ptr, in_ptr + input_size);

    n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
    a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
    b_ = std::vector<double>(input.begin() + (n * n), input.end());
  }

  broadcast(world_, n, 0);
  broadcast(world_, b_, 0);

  int remainder = n % world_.size();
  rows_per_proc_ = n / world_.size();
  if (world_.rank() < remainder) {
    rows_per_proc_++;
  }

  std::vector<int> counts_a(world_.size());
  std::vector<int> counts_b(world_.size());
  std::vector<int> displs_a(world_.size(), 0);
  std::vector<int> displs_b(world_.size(), 0);

  for (int i = 0; i < world_.size(); ++i) {
    int rows_i = (n / world_.size()) + (i < remainder ? 1 : 0);
    counts_a[i] = rows_i * n;
    counts_b[i] = rows_i;

    if (i > 0) {
      displs_a[i] = displs_a[i - 1] + counts_a[i - 1];
      displs_b[i] = displs_b[i - 1] + counts_b[i - 1];
    }
  }

  local_a_.resize(rows_per_proc_ * n);
  local_b_.resize(rows_per_proc_);

  if (world_.rank() == 0) {
    boost::mpi::scatterv(world_, a_, counts_a, displs_a, local_a_.data(), (rows_per_proc_ * n), 0);
    boost::mpi::scatterv(world_, b_, counts_b, displs_b, local_b_.data(), rows_per_proc_, 0);
  } else {
    boost::mpi::scatterv(world_, local_a_.data(), (rows_per_proc_ * n), 0);
    boost::mpi::scatterv(world_, local_b_.data(), rows_per_proc_, 0);
  }

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::RunImpl() {
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;

  double rs_old = DotProduct(world_, r, r);

  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> ap = MatrixVectorMultiply(p);
    double p_ap = DotProduct(world_, p, ap);

    broadcast(world_, p_ap, 0);
    if (std::abs(p_ap) < 1e-12) {
      break;
    }
    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    r = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(world_, r, r);
    broadcast(world_, rs_new, 0);
    if (std::sqrt(rs_new) < epsilon) {
      break;
    }
    double beta = rs_new / rs_old;
    p = VectorAdd(r, VectorScalarMultiply(p, beta));
    rs_old = rs_new;
  }

  output_ = x;

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
