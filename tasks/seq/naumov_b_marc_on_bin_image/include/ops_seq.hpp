#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_marc_test_utils {
void VerifyBinaryOutput(const std::vector<int> &in, const std::vector<int> &out);
void CheckTopNeighbor(const std::vector<int> &in, const std::vector<int> &out, int i, int j, int n);
void CheckLeftNeighbor(const std::vector<int> &in, const std::vector<int> &out, int i, int j, int n);
void VerifyNeighborConsistency(const std::vector<int> &in, const std::vector<int> &out, int m, int n);
}  // namespace naumov_b_marc_test_utils

namespace naumov_b_marc_on_bin_image_seq {

std::vector<int> GenerateRandomBinaryMatrix(int rows, int cols, double probability = 0.5);
std::vector<int> GenerateSparseBinaryMatrix(int rows, int cols, double probability = 0.1);
std::vector<int> GenerateDenseBinaryMatrix(int rows, int cols, double probability = 0.9);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProcessBlock(int start_row, int start_col, int block_rows, int block_cols);
  void ProcessPixel(int row, int col);
  void AssignNewLabel(int row, int col);
  void AssignMinLabel(int row, int col, const std::vector<int> &neighbors);
  void MergeLabels();

  std::vector<int> FindAdjacentLabels(int row, int col);
  void AssignLabel(int row, int col, int &current_label);
  int FindRoot(int label);
  void UnionLabels(int label1, int label2);
  void CalculateBlockSize();

  int rows_{};
  int cols_{};
  std::vector<int> input_image_;
  std::vector<int> output_image_;
  std::vector<int> label_parent_;
  int block_size_ = 64;
  int current_label_ = 0;
};

}  // namespace naumov_b_marc_on_bin_image_seq