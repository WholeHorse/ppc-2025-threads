#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_integrals_simpson_method_all {

class SimpsonIntegralSTLMPI : public ppc::core::Task {
 public:
  explicit SimpsonIntegralSTLMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;

  std::vector<double> boundaries_;
  std::vector<double> results_;
  int n_{};
  size_t dim_{};
  double rescoeff_{};
  double total_res_{};

  static double Func1D(double x);
  static double Func2D(double x, double y);
  void Simpson1D(double h, double a, double b, double& result, int overall_rank, int total_workers) const;
  void Simpson2D(double hx, double hy, double x0, double x1, double y0, double y1, double& result, int overall_rank,
                 int total_workers) const;
};

}  // namespace durynichev_d_integrals_simpson_method_all