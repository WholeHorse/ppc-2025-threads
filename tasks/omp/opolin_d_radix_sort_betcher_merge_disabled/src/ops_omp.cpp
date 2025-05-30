#include "omp/opolin_d_radix_sort_betcher_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::RunImpl() {
  int num_threads = omp_get_max_threads();
  int block_size = (size_ + num_threads - 1) / num_threads;

  std::vector<int> starts;
  std::vector<int> ends;
  for (int i = 0; i < num_threads; i++) {
    int start = i * block_size;
    int end = std::min(start + block_size, size_);
    if (start < end) {
      starts.push_back(start);
      ends.push_back(end);
    }
  }
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(starts.size()); i++) {
    RadixSort(input_, starts[i], ends[i]);
  }
  while (starts.size() > 1) {
    int merge_pairs = static_cast<int>(starts.size()) / 2;
    std::vector<int> new_starts(merge_pairs);
    std::vector<int> new_ends(merge_pairs);
#pragma omp parallel for
    for (int i = 0; i < merge_pairs; i++) {
      int idx = i * 2;
      int start = starts[idx];
      int mid = ends[idx];
      int end = ends[idx + 1];
      BatcherOddEvenMerge(input_, start, mid, end);
      new_starts[i] = start;
      new_ends[i] = end;
    }
    if (starts.size() % 2 == 1) {
      new_starts.push_back(starts.back());
      new_ends.push_back(ends.back());
    }
    starts = std::move(new_starts);
    ends = std::move(new_ends);
  }
  output_ = input_;
  return true;
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void opolin_d_radix_batcher_sort_omp::RadixSort(std::vector<int> &input, int start, int end) {
  std::vector<int> local_input(input.begin() + start, input.begin() + end);
  std::vector<int> positives;
  std::vector<int> negatives;
  for (size_t j = 0; j < local_input.size(); j++) {
    if (local_input[j] >= 0) {
      positives.push_back(local_input[j]);
    } else {
      negatives.push_back(-local_input[j]);
    }
  }
  int max_abs = 0;
  for (size_t j = 0; j < local_input.size(); j++) {
    max_abs = std::max(max_abs, std::abs(local_input[j]));
  }
  int digit_count = (max_abs == 0) ? 1 : static_cast<int>(std::log10(max_abs)) + 1;
  for (int place = 1; digit_count > 0; place *= 10, digit_count--) {
    if (!positives.empty()) {
      SortByDigit(positives, place);
    }
    if (!negatives.empty()) {
      SortByDigit(negatives, place);
    }
  }
  if (!negatives.empty()) {
    std::ranges::reverse(negatives);
    for (size_t j = 0; j < negatives.size(); j++) {
      negatives[j] = -negatives[j];
    }
  }
  std::vector<int> sorted_local;
  sorted_local.insert(sorted_local.end(), negatives.begin(), negatives.end());
  sorted_local.insert(sorted_local.end(), positives.begin(), positives.end());
  std::ranges::copy(sorted_local, input.begin() + start);
}

void opolin_d_radix_batcher_sort_omp::SortByDigit(std::vector<int> &array, int digit_place) {
  const int base = 10;
  std::vector<int> result(array.size());
  std::vector<int> buckets(base, 0);
  for (size_t i = 0; i < array.size(); i++) {
    int digit = (array[i] / digit_place) % base;
    buckets[digit]++;
  }
  for (int i = 1; i < base; i++) {
    buckets[i] += buckets[i - 1];
  }
  for (int i = static_cast<int>(array.size() - 1); i >= 0; i--) {
    int digit = (array[i] / digit_place) % base;
    result[--buckets[digit]] = array[i];
  }
  array = result;
}

void opolin_d_radix_batcher_sort_omp::BatcherOddEvenMerge(std::vector<int> &array, int start, int mid, int end) {
  int n = end - start;
  if (n <= 1) {
    return;
  }
  int p = 0;
  if (n <= 1) {
    p = 0;
  } else {
    int exp = static_cast<int>(std::floor(std::log2(n)));
    p = 1 << exp;
  }
  while (p > 0) {
#pragma omp parallel for
    for (int i = start; i < end - p; i++) {
      if (array[i] > array[i + p]) {
        std::swap(array[i], array[i + p]);
      }
    }
    p /= 2;
  }
}