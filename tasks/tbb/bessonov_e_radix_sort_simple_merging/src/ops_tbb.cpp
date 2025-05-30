#include "tbb/bessonov_e_radix_sort_simple_merging/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_scan.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_tbb {

TestTaskTbb::TestTaskTbb(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

void TestTaskTbb::ConvertToSortableBits(const std::vector<double>& in, std::vector<uint64_t>& out) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, in.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      uint64_t bits = 0;
      std::memcpy(&bits, &in[i], sizeof(double));
      out[i] = (bits & (1ULL << 63)) ? ~bits : bits ^ (1ULL << 63);
    }
  });
}

void TestTaskTbb::ConvertToDoubles(const std::vector<uint64_t>& in, std::vector<double>& out) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, in.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      uint64_t bits = in[i];
      bits = (bits & (1ULL << 63)) ? bits ^ (1ULL << 63) : ~bits;
      std::memcpy(&out[i], &bits, sizeof(double));
    }
  });
}

void TestTaskTbb::RadixSort(std::vector<uint64_t>& data) {
  if (data.empty()) {
    return;
  }

  const size_t n = data.size();
  std::vector<uint64_t> temp(n);
  const int passes = 8;

  for (int pass = 0; pass < passes; pass++) {
    int shift = pass * 8;
    std::vector<size_t> count(256, 0);

    count = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), std::vector<size_t>(256, 0),
        [&](const tbb::blocked_range<size_t>& r, std::vector<size_t> local_count) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            local_count[(data[i] >> shift) & 0xFF]++;
          }
          return local_count;
        },
        [](std::vector<size_t> a, const std::vector<size_t>& b) {
          for (size_t i = 0; i < 256; ++i) {
            a[i] += b[i];
          }
          return a;
        });

    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, 256), size_t(0),
        [&](const tbb::blocked_range<size_t>& r, size_t sum, bool is_final_scan) {
          size_t local_sum = sum;
          for (size_t i = r.begin(); i < r.end(); ++i) {
            local_sum += count[i];
            if (is_final_scan) {
              count[i] = local_sum;
            }
          }
          return local_sum;
        },
        [](size_t a, size_t b) { return a + b; });

    for (size_t i = n; i-- > 0;) {
      size_t bucket = (data[i] >> shift) & 0xFF;
      temp[--count[bucket]] = data[i];
    }

    data.swap(temp);
  }
}

bool TestTaskTbb::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size);
  return true;
}

bool TestTaskTbb::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr &&
         task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0;
}

bool TestTaskTbb::RunImpl() {
  const size_t n = input_.size();
  if (n == 0) {
    return true;
  }

  const size_t num_chunks = tbb::this_task_arena::max_concurrency();
  const size_t chunk_size = (n + num_chunks - 1) / num_chunks;

  std::vector<std::vector<uint64_t>> chunks(num_chunks);

  tbb::parallel_for(size_t(0), num_chunks, [&](size_t chunk_idx) {
    size_t begin = chunk_idx * chunk_size;
    size_t end = std::min(begin + chunk_size, n);
    if (begin >= end) {
      return;
    }
    std::vector<double> local(input_.begin() + static_cast<ptrdiff_t>(begin),
                              input_.begin() + static_cast<ptrdiff_t>(end));
    std::vector<uint64_t> sortable(end - begin);
    ConvertToSortableBits(local, sortable);
    RadixSort(sortable);
    chunks[chunk_idx] = std::move(sortable);
  });

  while (chunks.size() > 1) {
    size_t new_size = (chunks.size() + 1) / 2;
    std::vector<std::vector<uint64_t>> new_chunks(new_size);

    tbb::parallel_for(size_t(0), new_size, [&](size_t i) {
      size_t left_idx = i * 2;
      size_t right_idx = left_idx + 1;
      if (right_idx < chunks.size()) {
        new_chunks[i].resize(chunks[left_idx].size() + chunks[right_idx].size());
        std::merge(chunks[left_idx].begin(), chunks[left_idx].end(), chunks[right_idx].begin(), chunks[right_idx].end(),
                   new_chunks[i].begin());
      } else {
        new_chunks[i] = std::move(chunks[left_idx]);
      }
    });

    chunks = std::move(new_chunks);
  }

  ConvertToDoubles(chunks[0], output_);

  return true;
}

bool TestTaskTbb::PostProcessingImpl() {
  std::ranges::copy(output_.begin(), output_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}

}  // namespace bessonov_e_radix_sort_simple_merging_tbb