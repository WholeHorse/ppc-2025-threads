#include "tbb/laganina_e_component_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <vector>

bool laganina_e_component_labeling_tbb::TestTaskTBB::ValidationImpl() {
  if ((task_data == nullptr) || (task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }
  const unsigned int* input = reinterpret_cast<unsigned int*>(task_data->inputs[0]);
  const unsigned int size = task_data->inputs_count[0] * task_data->inputs_count[1];

  return std::all_of(input, input + size, [](int val) { return val == 0 || val == 1; });
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  const int size = rows_ * cols_;

  data_.resize(size);
  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);
  std::ranges::copy(input, input + size, data_.begin());

  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(data_.begin(), data_.end(), output);
  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::RunImpl() {
  LabelComponents();
  return true;
}

void laganina_e_component_labeling_tbb::TestTaskTBB::AssignFinalLabels(int size, UnionFind& uf) {
  tbb::concurrent_unordered_map<int, int> label_map;

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i]) {
      data_[i] = uf.Find(i) + 1;
    }
  });

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i] > 0) {
      label_map.insert({data_[i], 0});
    }
  });

  std::vector<int> keys;
  for (auto& p : label_map) {
    keys.push_back(p.first);
  }
  std::ranges::sort(keys.begin(), keys.end());

  int next_label = 1;
  for (auto& k : keys) {
    label_map[k] = next_label++;
  }

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i] > 0) {
      data_[i] = label_map[data_[i]];
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::LabelComponents() {
  const int size = rows_ * cols_;
  UnionFind uf(size, data_);

  ProcessComponents(uf);

  AssignFinalLabels(size, uf);
}
void laganina_e_component_labeling_tbb::TestTaskTBB::ProcessComponents(UnionFind& uf) {
  tbb::parallel_for(tbb::blocked_range2d<int>(0, rows_, 16, 0, cols_, 64), [&](const auto& r) { ProcessRange(r, uf); });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::ProcessRange(const tbb::blocked_range2d<int>& range,
                                                                  UnionFind& uf) {
  for (int i = range.rows().begin(); i < range.rows().end(); ++i) {
    const tbb::blocked_range<int> col_range = range.cols();
    ProcessRow(i, col_range, uf);
  }
}

void laganina_e_component_labeling_tbb::TestTaskTBB::ProcessRow(int row, const tbb::blocked_range<int>& col_range,
                                                                UnionFind& uf) {
  for (int j = col_range.begin(); j < col_range.end(); ++j) {
    const int idx = (row * cols_) + j;
    if (data_[idx] == 0) {
      continue;
    }

    CheckAllNeighbors(row, j, idx, uf);
  }
}
void laganina_e_component_labeling_tbb::TestTaskTBB::CheckAllNeighbors(int row, int col, int idx, UnionFind& uf) {
  if (col > 0 && (data_[idx - 1] != 0)) {
    uf.Unite(idx, idx - 1);
  }
  if (row > 0 && (data_[idx - cols_] != 0)) {
    uf.Unite(idx, idx - cols_);
  }

  if (col < cols_ - 1 && (data_[idx + 1] != 0)) {
    uf.Unite(idx, idx + 1);
  }

  if (row < rows_ - 1 && (data_[idx + cols_] != 0)) {
    uf.Unite(idx, idx + cols_);
  }
}