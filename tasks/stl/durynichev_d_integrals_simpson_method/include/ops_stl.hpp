#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_integrals_simpson_method_stl {

class SimpsonIntegralSTL : public ppc::core::Task {
 public:
  explicit SimpsonIntegralSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> boundaries_;
  std::vector<double> results_;
  int n_{};
  size_t dim_{};

  static double Func1D(double x);
  static double Func2D(double x, double y);
  void Simpson1D(double a, double b, double& result) const;
  void Simpson2D(double x0, double x1, double y0, double y1, double& result) const;
};

}  // namespace durynichev_d_integrals_simpson_method_stl