#include "omp/burykin_m_radix/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

std::array<int, 256> burykin_m_radix_omp::RadixOMP::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};
  const int n = static_cast<int>(a.size());

#pragma omp parallel
  {
    std::array<int, 256> local_count = {};

#pragma omp for nowait schedule(static)
    for (int i = 0; i < n; ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++local_count[key];
    }

    for (int bucket = 0; bucket < 256; ++bucket) {
      if (local_count[bucket] > 0) {
#pragma omp atomic
        count[bucket] += local_count[bucket];
      }
    }
  }

  return count;
}

std::array<int, 256> burykin_m_radix_omp::RadixOMP::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_omp::RadixOMP::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  const int n = static_cast<int>(a.size());

  const int num_threads = omp_get_max_threads();
  std::vector<std::array<int, 256>> thread_indices(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    thread_indices[t] = index;
  }

  std::vector<std::array<int, 256>> thread_counts(num_threads);
  for (auto& tc : thread_counts) {
    tc.fill(0);
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int chunk_size = (n + num_threads - 1) / num_threads;
    const int start = tid * chunk_size;
    const int end = std::min(start + chunk_size, n);

    for (int i = start; i < end; ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++thread_counts[tid][key];
    }
  }

  for (int bucket = 0; bucket < 256; ++bucket) {
    int offset = index[bucket];
    for (int t = 0; t < num_threads; ++t) {
      thread_indices[t][bucket] = offset;
      offset += thread_counts[t][bucket];
    }
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int chunk_size = (n + num_threads - 1) / num_threads;
    const int start = tid * chunk_size;
    const int end = std::min(start + chunk_size, n);

    auto local_indices = thread_indices[tid];

    for (int i = start; i < end; ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }

      b[local_indices[key]++] = v;
    }
  }
}

bool burykin_m_radix_omp::RadixOMP::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_omp::RadixOMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_omp::RadixOMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  for (int shift = 0; shift < 32; shift += 8) {
    auto count = ComputeFrequency(a, shift);
    const auto index = ComputeIndices(count);
    DistributeElements(a, b, index, shift);
    a.swap(b);
  }

  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_omp::RadixOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  const auto output_size = static_cast<int>(output_.size());

#pragma omp parallel for schedule(static, 1024)
  for (int i = 0; i < output_size; ++i) {
    output_ptr[i] = output_[i];
  }
  return true;
}
