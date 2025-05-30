#include "stl/durynichev_d_integrals_simpson_method/include/ops_stl.hpp"

#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace durynichev_d_integrals_simpson_method_stl {

bool SimpsonIntegralSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);
  n_ = static_cast<int>(boundaries_.back());
  boundaries_.pop_back();
  dim_ = boundaries_.size() / 2;

  results_ = std::vector<double>(ppc::util::GetPPCNumThreads(), 0.0);
  return true;
}

bool SimpsonIntegralSTL::ValidationImpl() {
  return task_data->inputs_count[0] >= 3 && task_data->outputs_count[0] == 1 && (n_ % 2 == 0);
}

bool SimpsonIntegralSTL::RunImpl() {
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  if (dim_ == 1) {
    double a = boundaries_[0];
    double b = boundaries_[1];
    double interval_size = (b - a) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
      double sub_a = a + (i * interval_size);
      double sub_b = (i == num_threads - 1) ? b : sub_a + interval_size;
      threads[i] = std::thread(&SimpsonIntegralSTL::Simpson1D, this, sub_a, sub_b, std::ref(results_[i]));
    }
  } else if (dim_ == 2) {
    double x0 = boundaries_[0];
    double x1 = boundaries_[1];
    double y0 = boundaries_[2];
    double y1 = boundaries_[3];
    double interval_size_x = (x1 - x0) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
      double sub_x0 = x0 + (i * interval_size_x);
      double sub_x1 = (i == num_threads - 1) ? x1 : sub_x0 + interval_size_x;
      threads[i] = std::thread(&SimpsonIntegralSTL::Simpson2D, this, sub_x0, sub_x1, y0, y1, std::ref(results_[i]));
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return true;
}

bool SimpsonIntegralSTL::PostProcessingImpl() {
  double final_result = 0.0;
  for (double res : results_) {
    final_result += res;
  }
  reinterpret_cast<double*>(task_data->outputs[0])[0] = final_result;
  return true;
}

double SimpsonIntegralSTL::Func1D(double x) { return x * x; }

double SimpsonIntegralSTL::Func2D(double x, double y) { return (x * x) + (y * y); }

void SimpsonIntegralSTL::Simpson1D(double a, double b, double& result) const {
  double h = (b - a) / n_;
  double sum = Func1D(a) + Func1D(b);

  for (int i = 1; i < n_; i += 2) {
    sum += 4 * Func1D(a + (i * h));
  }
  for (int i = 2; i < n_ - 1; i += 2) {
    sum += 2 * Func1D(a + (i * h));
  }

  result = sum * h / 3.0;
}

void SimpsonIntegralSTL::Simpson2D(double x0, double x1, double y0, double y1, double& result) const {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

  for (int i = 0; i <= n_; i++) {
    double x = x0 + (i * hx);
    double coef_x = 0.0;
    if (i == 0 || i == n_) {
      coef_x = 1;
    } else if (i % 2 != 0) {
      coef_x = 4;
    } else {
      coef_x = 2;
    }

    for (int j = 0; j <= n_; j++) {
      double y = y0 + (j * hy);
      double coef_y = 0.0;
      if (j == 0 || j == n_) {
        coef_y = 1;
      } else if (j % 2 != 0) {
        coef_y = 4;
      } else {
        coef_y = 2;
      }
      sum += coef_x * coef_y * Func2D(x, y);
    }
  }

  result = sum * hx * hy / 9.0;
}

}  // namespace durynichev_d_integrals_simpson_method_stl