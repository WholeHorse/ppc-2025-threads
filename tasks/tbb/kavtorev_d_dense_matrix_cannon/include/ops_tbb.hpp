// Copyright 2025 Kavtorev Dmitry
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace kavtorev_d_dense_matrix_cannon_tbb {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> res_;
  int n_ = 0, m_ = 0;
};

std::vector<double> MultiplyMatrix(const std::vector<double>& a, const std::vector<double>& b, int rows_a, int col_b);
std::vector<double> CannonMatrixMultiplication(const std::vector<double>& a, const std::vector<double>& b, int n,
                                               int m);

}  // namespace kavtorev_d_dense_matrix_cannon_tbb