#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_vertical_image_filtration_all {

class TaskAll : public ppc::core::Task {
 public:
  explicit TaskAll(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  [[nodiscard]] int GetRank() const { return world_.rank(); }

 private:
  void ApplyGaussianFilterTbb(std::vector<int>& local_output_ref, size_t my_start_output_row_val,
                              size_t my_num_rows_val, size_t output_cols_val);
  std::vector<int> input_;
  std::vector<int> output_;
  size_t width_{}, height_{};
  std::vector<float> gaussian_kernel_ = {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 4.0F / 16.0F,
                                         2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F};

  boost::mpi::environment env_;
  boost::mpi::communicator world_;
};

}  // namespace petrov_o_vertical_image_filtration_all
