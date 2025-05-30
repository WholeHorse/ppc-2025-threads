#include "seq/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mezhuev_m_bitwise_integer_sort_seq {

bool TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size > 0) {
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }

    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  } else {
    input_.clear();
  }

  output_.resize(output_size);
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size != output_size) {
    return false;
  }

  if (input_size > 0) {
    if (task_data->inputs[0] == nullptr) {
      return false;
    }
  }

  if (output_size > 0) {
    if (task_data->outputs[0] == nullptr) {
      return false;
    }
  }

  return true;
}

bool TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  std::vector<int> negative;
  std::vector<int> positive;
  for (int num : input_) {
    if (num < 0) {
      negative.push_back(-num);
    } else {
      positive.push_back(num);
    }
  }

  auto radix_sort = [](std::vector<int>& nums) {
    if (nums.empty()) {
      return;
    }
    int max_num = *std::ranges::max_element(nums);
    for (int exp = 1; max_num / exp > 0; exp *= 10) {
      std::vector<int> output(nums.size());
      std::vector<int> count(10, 0);
      for (int num : nums) {
        count[(num / exp) % 10]++;
      }
      for (int i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
      }
      for (size_t i = nums.size() - 1; i < nums.size(); --i) {
        int digit = (nums[i] / exp) % 10;
        output[count[digit] - 1] = nums[i];
        count[digit]--;
      }
      nums = output;
    }
  };

  radix_sort(positive);
  radix_sort(negative);

  std::ranges::reverse(negative);
  for (int& num : negative) {
    num = -num;
  }

  output_.clear();
  output_.insert(output_.end(), negative.begin(), negative.end());
  output_.insert(output_.end(), positive.begin(), positive.end());

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  if (!task_data || (task_data->outputs[0] == nullptr)) {
    return false;
  }

  if (output_.empty()) {
    return false;
  }

  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}

}  // namespace mezhuev_m_bitwise_integer_sort_seq