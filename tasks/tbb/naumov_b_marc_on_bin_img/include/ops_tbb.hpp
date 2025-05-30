#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/parallel_for.h"

namespace naumov_b_marc_on_bin_img_tbb {

std::vector<int> GenerateRandomBinaryMatrix(int rows, int cols, double probability = 0.5);
std::vector<int> GenerateSparseBinaryMatrix(int rows, int cols, double probability = 0.1);
std::vector<int> GenerateDenseBinaryMatrix(int rows, int cols, double probability = 0.9);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void FirstPass();
  void ResolveLabels();
  void SecondPass();

  void ProcessPixel(int i, int j, int& next_label);
  [[nodiscard]] int GetLeftLabel(int i, int j) const;
  [[nodiscard]] int GetTopLabel(int i, int j) const;
  void AssignNewLabel(size_t idx, int& next_label);
  void HandleLabelCollision(size_t idx, int left_label, int top_label);
  void HandleBothLabels(int a, int b);

  int FindRoot(int x);
  void UnionLabels(int a, int b);

  int rows_{};
  int cols_{};
  std::vector<int> input_image_;
  std::vector<int> output_image_;
  std::vector<int> label_parent_;
};

}  // namespace naumov_b_marc_on_bin_img_tbb