#include "omp/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mezhuev_m_bitwise_integer_sort_omp {

bool SortOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  if (input_size == 0) {
    output_ = input_;
    return true;
  }

  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  max_value_ = *std::ranges::max_element(input_, [](int a, int b) { return std::abs(a) < std::abs(b); });
  max_value_ = std::abs(max_value_);

  return true;
}

bool SortOpenMP::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SortOpenMP::RunImpl() {
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

  auto radix_sort = [](std::vector<int>& arr) {
    if (arr.empty()) {
      return;
    }

    int max_num = *std::ranges::max_element(arr);

    for (int exp = 1; max_num / exp > 0; exp *= 10) {
      std::vector<int> output(arr.size());
      std::vector<int> count(10, 0);
#pragma omp parallel
      {
        std::vector<int> local_count(10, 0);

#pragma omp for nowait
        for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
          int digit = (arr[i] / exp) % 10;
          local_count[digit]++;
        }

#pragma omp critical
        {
          for (int j = 0; j < 10; ++j) {
            count[j] += local_count[j];
          }
        }
      }
      for (int j = 1; j < 10; ++j) {
        count[j] += count[j - 1];
      }

      for (int i = static_cast<int>(arr.size()) - 1; i >= 0; --i) {
        int digit = (arr[i] / exp) % 10;
        output[--count[digit]] = arr[i];
      }

#pragma omp parallel for
      for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
        arr[i] = output[i];
      }
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

bool SortOpenMP::PostProcessingImpl() {
  if (input_.empty()) {
    return true;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace mezhuev_m_bitwise_integer_sort_omp
