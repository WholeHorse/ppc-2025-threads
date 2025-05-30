#include "all/opolin_d_radix_sort_batcher_merge/include/ops_all.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/enumerable_thread_specific.h"  // NOLINT(misc-include-cleaner)
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + size_);
    unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<int>(output_size, 0);
  }
  boost::mpi::broadcast(world_, size_, 0);
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::ValidationImpl() {
  if (world_.rank() == 0) {
    size_ = static_cast<int>(task_data->inputs_count[0]);
    if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
      return false;
    }
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::RunImpl() {
  int rank = world_.rank();
  int world_size = world_.size();
  std::vector<int> local_data;
  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);
  int local_size = 0;
  if (rank == 0) {
    int chunk = size_ / world_size;
    int remainder = size_ % world_size;
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      counts[i] = chunk + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += counts[i];
    }
  }
  boost::mpi::scatter(world_, counts, local_size, 0);
  local_data.resize(local_size);
  if (rank == 0) {
    boost::mpi::scatterv(world_, input_, counts, displs, local_data.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(world_, local_data.data(), local_size, 0);
  }
  std::vector<uint32_t> keys(local_size);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, local_size), [&](auto& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertIntToUint(local_data[i]);
    }
  });

  RadixSort(keys);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, local_size), [&](auto& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      local_data[i] = ConvertUintToInt(keys[i]);
    }
  });

  if (local_size > 0) {
    BatcherOddEvenMerge(local_data, 0, local_size);
  }
  std::vector<int> gathered_data;
  if (rank == 0) {
    gathered_data.resize(size_);
  }
  if (rank == 0) {
    boost::mpi::gatherv(world_, local_data.data(), local_size, gathered_data.data(), counts, displs, 0);
  } else {
    boost::mpi::gatherv(world_, local_data.data(), local_size, 0);
  }
  if (rank == 0) {
    output_ = std::move(gathered_data);
    int offset = counts[0];
    for (int i = 1; i < world_size; ++i) {
      int next = offset + counts[i];
      std::inplace_merge(output_.begin(), output_.begin() + offset, output_.begin() + next);
      offset = next;
    }
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_all::ConvertIntToUint(int num) { return static_cast<uint32_t>(num) ^ 0x80000000U; }

int opolin_d_radix_batcher_sort_all::ConvertUintToInt(uint32_t unum) { return static_cast<int>(unum ^ 0x80000000U); }

void opolin_d_radix_batcher_sort_all::RadixSort(std::vector<uint32_t>& uns_vec) {
  size_t sz = uns_vec.size();
  if (sz <= 1) {
    return;
  }
  const int rad = 256;
  std::vector<uint32_t> res(sz);
  for (int stage = 0; stage < 4; stage++) {
    tbb::enumerable_thread_specific<std::vector<size_t>> local_counts([&] { return std::vector<size_t>(rad, 0); });
    int shift = stage * 8;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, sz), [&](const tbb::blocked_range<size_t>& r) {
      auto& lc = local_counts.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
        lc[byte]++;
      }
    });
    std::vector<size_t> pref(rad, 0);
    for (auto& lc_instance : local_counts) {
      for (int j = 0; j < rad; ++j) {
        pref[j] += lc_instance[j];
      }
    }
    for (int j = 1; j < rad; ++j) {
      pref[j] += pref[j - 1];
    }
    for (int i = static_cast<int>(sz) - 1; i >= 0; --i) {
      const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
      res[--pref[byte]] = uns_vec[i];
    }
    uns_vec.swap(res);
  }
}

void opolin_d_radix_batcher_sort_all::BatcherOddEvenMerge(std::vector<int>& vec, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;
  tbb::parallel_invoke([&] { BatcherOddEvenMerge(vec, low, mid); }, [&] { BatcherOddEvenMerge(vec, mid, high); });

  int first_half_len = mid - low;
  int second_half_len = high - mid;
  int common_len = std::min(first_half_len, second_half_len);

  if (common_len > 0) {
    tbb::parallel_for(tbb::blocked_range<int>(0, common_len), [&](const auto& r) {
      for (int i_offset = r.begin(); i_offset < r.end(); ++i_offset) {
        if (vec[low + i_offset] > vec[mid + i_offset]) {
          std::swap(vec[low + i_offset], vec[mid + i_offset]);
        }
      }
    });
  }
}