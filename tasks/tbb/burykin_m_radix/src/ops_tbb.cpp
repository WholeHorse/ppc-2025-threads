#include "tbb/burykin_m_radix/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeFrequencyParallel(const std::vector<int>& a,
                                                                             const int shift) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, a.size(), 10000), std::array<int, 256>{},
      [&](const tbb::blocked_range<size_t>& range, std::array<int, 256> local_count) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
          if (shift == 24) {
            key ^= 0x80;
          }
          ++local_count[key];
        }
        return local_count;
      },
      [](const std::array<int, 256>& left, const std::array<int, 256>& right) {
        std::array<int, 256> result{};
        for (int i = 0; i < 256; ++i) {
          result[i] = left[i] + right[i];
        }
        return result;
      },
      tbb::auto_partitioner());
}

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_tbb::RadixTBB::DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                                               const std::array<int, 256>& index, const int shift) {
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  const size_t n = a.size();

  const size_t optimal_chunk_size = std::max(size_t(1000), n / (num_threads * 4));
  const size_t num_chunks = (n + optimal_chunk_size - 1) / optimal_chunk_size;

  struct ChunkInfo {
    std::array<int, 256> counts{};
    size_t start_idx{};
    size_t end_idx{};
  };

  std::vector<ChunkInfo> chunks(num_chunks);

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_chunks, 1),
      [&](const tbb::blocked_range<size_t>& chunk_range) {
        for (size_t chunk_id = chunk_range.begin(); chunk_id < chunk_range.end(); ++chunk_id) {
          const size_t start = chunk_id * optimal_chunk_size;
          const size_t end = std::min(start + optimal_chunk_size, n);

          chunks[chunk_id].start_idx = start;
          chunks[chunk_id].end_idx = end;
          chunks[chunk_id].counts.fill(0);

          for (size_t i = start; i < end; ++i) {
            unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
            if (shift == 24) {
              key ^= 0x80;
            }
            ++chunks[chunk_id].counts[key];
          }
        }
      },
      tbb::auto_partitioner());

  std::vector<std::array<int, 256>> chunk_offsets(num_chunks);

  for (int bucket = 0; bucket < 256; ++bucket) {
    int offset = index[bucket];
    for (size_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
      chunk_offsets[chunk_id][bucket] = offset;
      offset += chunks[chunk_id].counts[bucket];
    }
  }

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_chunks, 1),
      [&](const tbb::blocked_range<size_t>& chunk_range) {
        for (size_t chunk_id = chunk_range.begin(); chunk_id < chunk_range.end(); ++chunk_id) {
          auto local_offsets = chunk_offsets[chunk_id];
          const size_t start = chunks[chunk_id].start_idx;
          const size_t end = chunks[chunk_id].end_idx;

          for (size_t i = start; i < end; ++i) {
            unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
            if (shift == 24) {
              key ^= 0x80;
            }
            b[local_offsets[key]++] = a[i];
          }
        }
      },
      tbb::auto_partitioner());
}

bool burykin_m_radix_tbb::RadixTBB::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_tbb::RadixTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_tbb::RadixTBB::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  const size_t num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(static_cast<int>(num_threads));

  arena.execute([&] {
    std::vector<int> a = std::move(input_);
    std::vector<int> b(a.size());

    for (int shift = 0; shift < 32; shift += 8) {
      auto count = ComputeFrequencyParallel(a, shift);
      const auto index = ComputeIndices(count);
      DistributeElementsParallel(a, b, index, shift);
      a.swap(b);
    }

    output_ = std::move(a);
  });

  return true;
}

bool burykin_m_radix_tbb::RadixTBB::PostProcessingImpl() {
  const size_t output_size = output_.size();
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);

  if (output_size > 1000) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, output_size, 10000),
        [&](const tbb::blocked_range<size_t>& range) {
          using DiffT = std::vector<int>::difference_type;
          auto begin = output_.begin() + static_cast<DiffT>(range.begin());
          auto end = output_.begin() + static_cast<DiffT>(range.end());
          std::copy(begin, end, output_ptr + range.begin());
        },
        tbb::auto_partitioner());
  } else {
    std::copy(output_.begin(), output_.end(), output_ptr);
  }

  return true;
}
