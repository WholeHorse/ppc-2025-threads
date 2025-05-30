#include "tbb/naumov_b_marc_on_bin_img/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateRandomBinaryMatrix(int rows, int cols, double probability) {
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

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateSparseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateDenseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::UnionLabels(int a, int b) {
  int ra = FindRoot(a);
  int rb = FindRoot(b);
  if (ra != rb) {
    if (ra < rb) {
      label_parent_[rb] = ra;
    } else {
      label_parent_[ra] = rb;
    }
  }
}

int naumov_b_marc_on_bin_img_tbb::TestTaskTBB::FindRoot(int x) {
  int root = x;
  while (root != label_parent_[root]) {
    root = label_parent_[root];
  }

  while (x != root) {
    int next = label_parent_[x];
    label_parent_[x] = root;
    x = next;
  }
  return root;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  size_t total = size_t(rows_) * size_t(cols_);

  input_image_.assign(reinterpret_cast<int*>(task_data->inputs[0]),
                      reinterpret_cast<int*>(task_data->inputs[0]) + total);

  output_image_.assign(total, 0);

  label_parent_.clear();
  label_parent_.resize(total + 1, 0);

  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr) {
    return false;
  }

  const int m = static_cast<int>(task_data->inputs_count[0]);
  const int n = static_cast<int>(task_data->inputs_count[1]);
  if (m <= 0 || n <= 0) {
    return false;
  }

  const int* in = reinterpret_cast<int*>(task_data->inputs[0]);
  const size_t total = static_cast<size_t>(m) * n;
  return std::all_of(in, in + total, [](int val) { return val == 0 || val == 1; });
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::FirstPass() {
  int next_label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      ProcessPixel(i, j, next_label);
    }
  }
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::ProcessPixel(int i, int j, int& next_label) {
  const size_t idx = (static_cast<size_t>(i) * cols_) + j;
  if (input_image_[idx] == 0) {
    return;
  }

  const int left_label = GetLeftLabel(i, j);
  const int top_label = GetTopLabel(i, j);

  if (left_label == 0 && top_label == 0) {
    AssignNewLabel(idx, next_label);
  } else {
    HandleLabelCollision(idx, left_label, top_label);
  }
}

int naumov_b_marc_on_bin_img_tbb::TestTaskTBB::GetLeftLabel(int i, int j) const {
  return (j > 0) ? output_image_[(static_cast<size_t>(i) * cols_) + (j - 1)] : 0;
}

int naumov_b_marc_on_bin_img_tbb::TestTaskTBB::GetTopLabel(int i, int j) const {
  return (i > 0) ? output_image_[(static_cast<size_t>(i - 1) * cols_) + j] : 0;
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::AssignNewLabel(size_t idx, int& next_label) {
  output_image_[idx] = next_label;
  label_parent_[next_label] = next_label;
  ++next_label;
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::HandleLabelCollision(size_t idx, int left, int top) {
  if (left != 0 && top != 0) {
    HandleBothLabels(left, top);
    output_image_[idx] = std::min(left, top);
  } else {
    output_image_[idx] = std::max(left, top);
  }
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::HandleBothLabels(int a, int b) {
  const int min_label = std::min(a, b);
  const int max_label = std::max(a, b);
  UnionLabels(min_label, max_label);
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::ResolveLabels() {
  for (size_t i = 1; i < label_parent_.size(); ++i) {
    label_parent_[i] = FindRoot(static_cast<int>(i));
  }
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::SecondPass() {
  const size_t total = static_cast<size_t>(rows_) * cols_;
  tbb::parallel_for(size_t(0), total, [this](size_t idx) {
    if (output_image_[idx] > 0) {
      output_image_[idx] = label_parent_[output_image_[idx]];
    }
  });
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::RunImpl() {
  std::iota(label_parent_.begin(), label_parent_.end(), 0);
  FirstPass();
  ResolveLabels();
  SecondPass();
  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PostProcessingImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  int* out = reinterpret_cast<int*>(task_data->outputs[0]);
  size_t total = output_image_.size();
  for (size_t i = 0; i < total; ++i) {
    out[i] = output_image_[i];
  }
  return true;
}
