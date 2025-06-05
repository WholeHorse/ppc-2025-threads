#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

double fomin_v_conjugate_gradient::FominVConjugateGradientStl::DotProduct(const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<double> partial_results(num_threads, 0.0);

  const size_t chunk_size = a.size() / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    const size_t start = i * chunk_size;
    const size_t end = (i == num_threads - 1) ? a.size() : start + chunk_size;
    threads[i] = std::thread([&, start, end, i]() {
      for (size_t j = start; j < end; ++j) {
        partial_results[i] += a[j] * b[j];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  double result = 0.0;
  for (const auto& val : partial_results) {
    result += val;
  }

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> result(n, 0.0);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  const int chunk_size = n / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const int start = t * chunk_size;
    const int end = (t == num_threads - 1) ? n : start + chunk_size;
    threads[t] = std::thread([&, start, end]() {
      for (int i = start; i < end; ++i) {
        for (int j = 0; j < n; ++j) {
          result[i] += a[(i * n) + j] * x[j];
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  const size_t chunk_size = a.size() / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = (t == num_threads - 1) ? a.size() : start + chunk_size;
    threads[t] = std::thread([&, start, end]() {
      for (size_t i = start; i < end; ++i) {
        result[i] = a[i] + b[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  const size_t chunk_size = a.size() / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = (t == num_threads - 1) ? a.size() : start + chunk_size;
    threads[t] = std::thread([&, start, end]() {
      for (size_t i = start; i < end; ++i) {
        result[i] = a[i] - b[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  const size_t chunk_size = v.size() / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = (t == num_threads - 1) ? v.size() : start + chunk_size;
    threads[t] = std::thread([&, start, end]() {
      for (size_t i = start; i < end; ++i) {
        result[i] = v[i] * scalar;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
  b_ = std::vector<double>(input.begin() + (n * n), input.end());
  output_.resize(n, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::RunImpl() {
  const double epsilon = (n < 100) ? 1e-4 : 1e-6;
  const int max_iter = (n < 100) ? 200 : 1000;
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r);

  for (int i = 0; i < max_iter; ++i) {
    std::vector<double> ap = MatrixVectorMultiply(a_, p);
    double p_ap = DotProduct(p, ap);

    if (std::abs(p_ap) < 1e-12) {
      break;
    }

    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    std::vector<double> r_new = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(r_new, r_new);
    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    double beta = rs_new / rs_old;
    p = VectorAdd(r_new, VectorScalarMultiply(p, beta));
    r = r_new;
    rs_old = rs_new;
  }

  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
