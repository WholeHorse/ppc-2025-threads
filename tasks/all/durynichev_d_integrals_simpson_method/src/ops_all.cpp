#include "all/durynichev_d_integrals_simpson_method/include/ops_all.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/reduce.hpp"
#include "core/util/include/util.hpp"

namespace durynichev_d_integrals_simpson_method_all {

bool SimpsonIntegralSTLMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);
    n_ = static_cast<int>(boundaries_.back());
    boundaries_.pop_back();
    dim_ = boundaries_.size() / 2;
  }

  results_ = std::vector<double>(ppc::util::GetPPCNumThreads(), 0.0);
  return true;
}

bool SimpsonIntegralSTLMPI::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] >= 3 && task_data->outputs_count[0] == 1 && (n_ % 2 == 0));
}

bool SimpsonIntegralSTLMPI::RunImpl() {
  boost::mpi::broadcast(world_, boundaries_, 0);
  boost::mpi::broadcast(world_, dim_, 0);
  boost::mpi::broadcast(world_, n_, 0);

  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);

  const int overall_rank_offset = num_threads * world_.rank();
  const int total_workers = num_threads * world_.size();

  double preres{};
  if (dim_ == 1) {
    const double a = boundaries_[0];
    const double b = boundaries_[1];

    const double h = (b - a) / n_;

    rescoeff_ = h / 3.0;
    preres = world_.rank() == 0 ? (Func1D(a) + Func1D(b)) : 0;

    for (int i = 0; i < num_threads; ++i) {
      threads[i] = std::thread(&SimpsonIntegralSTLMPI::Simpson1D, this, h, a, b, std::ref(results_[i]),
                               overall_rank_offset + i, total_workers);
    }
  } else if (dim_ == 2) {
    const double x0 = boundaries_[0];
    const double x1 = boundaries_[1];
    const double y0 = boundaries_[2];
    const double y1 = boundaries_[3];

    const double hx = (x1 - x0) / n_;
    const double hy = (y1 - y0) / n_;

    rescoeff_ = hx * hy / 9.0;
    preres = 0.0;

    for (int i = 0; i < num_threads; ++i) {
      threads[i] = std::thread(&SimpsonIntegralSTLMPI::Simpson2D, this, hx, hy, x0, x1, y0, y1, std::ref(results_[i]),
                               overall_rank_offset + i, total_workers);
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (double res : results_) {
    preres += res;
  }
  boost::mpi::reduce(world_, preres, total_res_, std::plus{}, 0);

  return true;
}

bool SimpsonIntegralSTLMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    total_res_ *= rescoeff_;
    reinterpret_cast<double*>(task_data->outputs[0])[0] = total_res_;
  }
  return true;
}

double SimpsonIntegralSTLMPI::Func1D(double x) { return x * x; }

double SimpsonIntegralSTLMPI::Func2D(double x, double y) { return (x * x) + (y * y); }

namespace {
std::pair<int, int> CalcRange(int space, int overall_rank, int total_workers) {
  const int base = space / total_workers;

  const int start = base * overall_rank;
  const int end = (overall_rank == (total_workers - 1)) ? space : (base * (overall_rank + 1));

  return std::make_pair(start, end);
}
}  // namespace

void SimpsonIntegralSTLMPI::Simpson1D(double h, double a, double b, double& result, int overall_rank,
                                      int total_workers) const {
  const auto r = CalcRange(n_ - 1, overall_rank, total_workers);
  double sum = 0.0;
  for (int i = 1 + r.first; i <= r.second; ++i) {
    double coef = (i % 2 == 0) ? 2.0 : 4.0;
    sum += coef * Func1D(a + (i * h));
  }
  result = sum;
}

void SimpsonIntegralSTLMPI::Simpson2D(double hx, double hy, double x0, double x1, double y0, double y1, double& result,
                                      int overall_rank, int total_workers) const {
  const auto r = CalcRange(n_ + 1, overall_rank, total_workers);

  double sum = 0.0;
  for (int i = r.first; i < r.second; ++i) {
    const double x = x0 + (i * hx);
    double coef_x = 0.0;
    if (i == 0 || i == n_) {
      coef_x = 1;
    } else if (i % 2 != 0) {
      coef_x = 4;
    } else {
      coef_x = 2;
    }

    for (int j = 0; j <= n_; j++) {
      const double y = y0 + (j * hy);
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

  result = sum;
}

}  // namespace durynichev_d_integrals_simpson_method_all