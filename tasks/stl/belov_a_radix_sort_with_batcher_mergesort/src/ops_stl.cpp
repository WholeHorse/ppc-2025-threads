#include "stl/belov_a_radix_sort_with_batcher_mergesort/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_stl {

constexpr int kDecimalBase = 10;

int RadixBatcherMergesortParallel::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortParallel::Sort(std::span<Bigint> arr) {
  std::vector<Bigint> pos;
  std::vector<Bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  RadixSort(pos, false);
  RadixSort(neg, true);

  size_t index = 0;

  for (const auto& num : neg) {
    arr[index++] = -num;
  }

  for (const auto& num : pos) {
    arr[index++] = num;
  }
}

void RadixBatcherMergesortParallel::RadixSort(std::vector<Bigint>& arr, bool invert) {
  if (arr.empty()) {
    return;
  }

  Bigint max_val = *std::ranges::max_element(arr);
  int max_val_digit_capacity = GetNumberDigitCapacity(max_val);
  int iter = 1;

  for (Bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    CountingSort(arr, digit_place);
  }

  if (invert) {
    std::ranges::reverse(arr);
  }
}

void RadixBatcherMergesortParallel::CountingSort(std::vector<Bigint>& arr, Bigint digit_place) {
  std::vector<Bigint> output(arr.size());
  int count[kDecimalBase] = {};

  for (const auto& num : arr) {
    Bigint index = (num / digit_place) % kDecimalBase;
    count[index]++;
  }

  for (int i = 1; i < kDecimalBase; i++) {
    count[i] += count[i - 1];
  }

  for (size_t i = arr.size() - 1; i < arr.size(); i--) {
    Bigint num = arr[i];
    Bigint index = (num / digit_place) % kDecimalBase;
    output[--count[index]] = num;
  }

  std::ranges::copy(output, arr.begin());
}

void RadixBatcherMergesortParallel::SortParallel(std::vector<Bigint>& arr) const {
  if (arr.empty()) {
    return;
  }

  constexpr size_t kSortThreshold = 1000;
  if (arr.size() <= kSortThreshold) {
    Sort(std::span<Bigint>(arr.data(), arr.size()));
    return;
  }

  const size_t n = arr.size();
  const size_t num_threads = this->num_threads_;
  const size_t chunk_size = (n + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
    const size_t start = thread_id * chunk_size;
    const size_t end = std::min(start + chunk_size, n);
    if (start < n) {
      threads.emplace_back([start, end, &arr]() {
        std::span<Bigint> local_span(arr.data() + start, end - start);
        Sort(local_span);
      });
    }
  }

  for (auto& t : threads) {
    t.join();
  }
}

void RadixBatcherMergesortParallel::BatcherMergeParallel(std::vector<Bigint>& arr) const {
  const size_t n = arr.size();
  if (n <= 1) {
    return;
  }

  constexpr size_t kMergeThreshold = 32;
  size_t step = (n + num_threads_ - 1) / num_threads_;

  while (step < n) {
    const size_t block_size = 2 * step;
    const bool use_parallel = (block_size >= kMergeThreshold);

    if (use_parallel) {
      const size_t num_blocks = (n + block_size - 1) / block_size;
      std::vector<std::thread> threads;
      threads.reserve(num_blocks);

      for (size_t i = 0; i < num_blocks; ++i) {
        threads.emplace_back([i, n, step, block_size, &arr]() {
          const size_t left = i * block_size;
          const size_t mid = std::min(left + step, n);
          const size_t right = std::min(left + block_size, n);
          if (mid < right) {
            std::inplace_merge(arr.begin() + static_cast<int64_t>(left), arr.begin() + static_cast<int64_t>(mid),
                               arr.begin() + static_cast<int64_t>(right));
          }
        });
      }
      for (auto& t : threads) {
        t.join();
      }
    } else {
      for (size_t left = 0; left < n; left += block_size) {
        const size_t mid = std::min(left + step, n);
        const size_t right = std::min(left + block_size, n);
        if (mid < right) {
          std::inplace_merge(arr.begin() + static_cast<int64_t>(left), arr.begin() + static_cast<int64_t>(mid),
                             arr.begin() + static_cast<int64_t>(right));
        }
      }
    }
    step *= 2;
  }
}

bool RadixBatcherMergesortParallel::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);
  array_.assign(input_array_data, input_array_data + n_);

  return true;
}

bool RadixBatcherMergesortParallel::ValidationImpl() {
  return (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
          (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty());
}

bool RadixBatcherMergesortParallel::RunImpl() {
  num_threads_ = std::max(static_cast<size_t>(ppc::util::GetPPCNumThreads()), size_t(1));
  SortParallel(array_);
  BatcherMergeParallel(array_);
  return true;
}

bool RadixBatcherMergesortParallel::PostProcessingImpl() {
  std::ranges::copy(array_, reinterpret_cast<Bigint*>(task_data->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_stl