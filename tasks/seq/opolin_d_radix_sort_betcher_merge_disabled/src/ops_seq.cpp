#include "seq/opolin_d_radix_sort_betcher_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool opolin_d_radix_betcher_sort_seq::RadixBetcherSortTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_betcher_sort_seq::RadixBetcherSortTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_betcher_sort_seq::RadixBetcherSortTaskSequential::RunImpl() {
  std::vector<int> positives;
  std::vector<int> negatives;
  for (int i = 0; i < size_; i++) {
    if (input_[i] >= 0) {
      positives.push_back(input_[i]);
    } else {
      negatives.push_back(-input_[i]);
    }
  }
  int max_abs = 0;
  for (int i = 0; i < size_; i++) {
    max_abs = std::max(max_abs, std::abs(input_[i]));
  }
  int digit_count = 0;
  if (max_abs == 0) {
    digit_count = 1;
  } else {
    while (max_abs > 0) {
      max_abs /= 10;
      digit_count++;
    }
  }
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
  }

  for (size_t i = 0; i < negatives.size(); i++) {
    output_[i] = -negatives[i];
  }
  for (size_t i = 0; i < positives.size(); i++) {
    output_[negatives.size() + i] = positives[i];
  }
  return true;
}

bool opolin_d_radix_betcher_sort_seq::RadixBetcherSortTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void opolin_d_radix_betcher_sort_seq::SortByDigit(std::vector<int> &array, int digit_place) {
  const int base = 10;
  std::vector<int> result(array.size());
  std::vector<int> buckets(base, 0);

  for (int value : array) {
    int digit = (value / digit_place) % base;
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