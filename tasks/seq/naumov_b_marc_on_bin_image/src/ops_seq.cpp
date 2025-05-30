#include "seq/naumov_b_marc_on_bin_image/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

std::vector<int> naumov_b_marc_on_bin_image_seq::GenerateRandomBinaryMatrix(int rows, int cols, double probability) {
  const int total_elements = rows * cols;
  const int target_ones = static_cast<int>(total_elements * probability);

  std::vector<int> matrix(total_elements, 1);

  const int zeros_needed = total_elements - target_ones;

  for (int i = 0; i < zeros_needed; ++i) {
    matrix[i] = 0;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(matrix.begin(), matrix.end(), g);

  return matrix;
}

std::vector<int> naumov_b_marc_on_bin_image_seq::GenerateSparseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

std::vector<int> naumov_b_marc_on_bin_image_seq::GenerateDenseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

void naumov_b_marc_on_bin_image_seq::TestTaskSequential::ProcessPixel(int row, int col) {
  std::vector<int> neighbors = FindAdjacentLabels(row, col);

  if (neighbors.empty()) {
    AssignNewLabel(row, col);
  } else {
    AssignMinLabel(row, col, neighbors);
  }
}

void naumov_b_marc_on_bin_image_seq::TestTaskSequential::AssignNewLabel(int row, int col) {
  output_image_[(row * cols_) + col] = ++current_label_;
  if (static_cast<size_t>(current_label_) >= label_parent_.size()) {
    label_parent_.resize(current_label_ + 1, 0);
  }
  label_parent_[current_label_] = current_label_;
}

void naumov_b_marc_on_bin_image_seq::TestTaskSequential::AssignMinLabel(int row, int col,
                                                                        const std::vector<int>& neighbors) {
  if (neighbors.empty()) {
    AssignNewLabel(row, col);
    return;
  }

  int min_label = *std::ranges::min_element(neighbors);
  output_image_[(row * cols_) + col] = min_label;

  for (int neighbor_label : neighbors) {
    if (neighbor_label != min_label) {
      UnionLabels(min_label, neighbor_label);
    }
  }
}

std::vector<int> naumov_b_marc_on_bin_image_seq::TestTaskSequential::FindAdjacentLabels(int row, int col) {
  std::vector<int> neighbors;

  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    return neighbors;
  }

  if (col > 0 && output_image_[(row * cols_) + (col - 1)] != 0) {
    neighbors.push_back(output_image_[(row * cols_) + (col - 1)]);
  }

  if (row > 0 && output_image_[((row - 1) * cols_) + col] != 0) {
    neighbors.push_back(output_image_[((row - 1) * cols_) + col]);
  }

  return neighbors;
}

void naumov_b_marc_on_bin_image_seq::TestTaskSequential::UnionLabels(int label1, int label2) {
  if (static_cast<size_t>(label1) >= label_parent_.size() || static_cast<size_t>(label2) >= label_parent_.size()) {
    return;
  }

  int root1 = FindRoot(label1);
  int root2 = FindRoot(label2);

  if (root1 != root2) {
    label_parent_[root2] = root1;
  }
}

int naumov_b_marc_on_bin_image_seq::TestTaskSequential::FindRoot(int label) {
  if (static_cast<size_t>(label) >= label_parent_.size()) {
    return label;
  }

  if (label_parent_[label] == label) {
    return label;
  }

  label_parent_[label] = FindRoot(label_parent_[label]);
  return label_parent_[label];
}

void naumov_b_marc_on_bin_image_seq::TestTaskSequential::MergeLabels() {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (input_image_[(i * cols_) + j] == 1) {
        int root = FindRoot(output_image_[(i * cols_) + j]);
        output_image_[(i * cols_) + j] = root;
      }
    }
  }
}

bool naumov_b_marc_on_bin_image_seq::TestTaskSequential::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  cols_ = static_cast<int>(task_data->inputs_count[1]);

  input_image_.resize(rows_ * cols_, 0);
  output_image_.resize(rows_ * cols_, 0);
  label_parent_.clear();
  current_label_ = 0;

  int* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  for (int i = 0; i < rows_ * cols_; ++i) {
    input_image_[i] = input_data[i];
  }

  return true;
}

bool naumov_b_marc_on_bin_image_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count[0] <= 0 || task_data->inputs_count[1] <= 0) {
    return false;
  }

  size_t expected_size = task_data->inputs_count[0] * task_data->inputs_count[1];
  if (task_data->inputs[0] == nullptr) {
    return false;
  }

  int* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  for (size_t i = 0; i < expected_size; ++i) {
    if (input_data[i] != 0 && input_data[i] != 1) {
      return false;
    }
  }

  return true;
}

bool naumov_b_marc_on_bin_image_seq::TestTaskSequential::RunImpl() {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (input_image_[(i * cols_) + j] == 1) {
        ProcessPixel(i, j);
      }
    }
  }

  MergeLabels();

  return true;
}

bool naumov_b_marc_on_bin_image_seq::TestTaskSequential::PostProcessingImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }

  int* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
  const size_t data_size = output_image_.size();

  for (size_t i = 0; i < data_size; ++i) {
    output_data[i] = output_image_[i];
  }

  return true;
}