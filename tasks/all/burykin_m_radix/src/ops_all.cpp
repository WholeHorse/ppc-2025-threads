#include "../include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

namespace {
template <typename T>
constexpr size_t Bytes() {
  return sizeof(T);
}

template <typename T>
constexpr size_t Bits() {
  return Bytes<T>() * CHAR_BIT;
}

class Bitutil {
 public:
  static constexpr uint32_t AsU32(int x) {
    const uint32_t ux = static_cast<uint32_t>(x);
    return ux ^ (1U << 31);
  }

  template <typename T>
    requires std::is_integral_v<T>
  static constexpr uint8_t ByteAt(const T &val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

void RadixSort(std::span<int> v) {
  if (v.empty()) return;

  constexpr size_t kBase = 1 << CHAR_BIT;

  std::vector<int> aux_buf(v.size());
  std::span<int> aux{aux_buf};

  std::array<std::size_t, kBase> count;

  std::span<int> current = v;
  std::span<int> temp = aux;

  for (std::size_t ib = 0; ib < Bytes<int>(); ++ib) {
    std::ranges::fill(count, 0);

    std::ranges::for_each(current, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]; });

    std::partial_sum(count.begin(), count.end(), count.begin());

    std::ranges::for_each(std::ranges::reverse_view(current),
                          [&](auto el) { temp[--count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]] = el; });

    std::swap(current, temp);
  }

  if (current.data() != v.data()) {
    std::ranges::copy(current, v.begin());
  }
}

std::vector<std::span<int>> Distribute(std::span<int> arr, std::size_t n) {
  std::vector<std::span<int>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto *cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}

void SafeMerge(std::vector<int> &target, const std::vector<std::vector<int>> &sorted_chunks) {
  if (sorted_chunks.empty()) return;

  std::vector<const std::vector<int> *> non_empty_chunks;
  for (const auto &chunk : sorted_chunks) {
    if (!chunk.empty()) {
      non_empty_chunks.push_back(&chunk);
    }
  }

  if (non_empty_chunks.empty()) {
    target.clear();
    return;
  }

  if (non_empty_chunks.size() == 1) {
    target = *non_empty_chunks[0];
    return;
  }

  target = *non_empty_chunks[0];
  for (size_t i = 1; i < non_empty_chunks.size(); ++i) {
    std::vector<int> temp;
    temp.reserve(target.size() + non_empty_chunks[i]->size());
    std::merge(target.begin(), target.end(), non_empty_chunks[i]->begin(), non_empty_chunks[i]->end(),
               std::back_inserter(temp));
    target = std::move(temp);
  }
}

}  // namespace

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    std::span<int> src = {reinterpret_cast<int *>(task_data->inputs[0]), task_data->inputs_count[0]};
    input_.assign(src.begin(), src.end());
    output_.resize(input_.size());
  }
  return true;
}

void burykin_m_radix_all::RadixALL::Squash(boost::mpi::communicator &group) {
  const auto numprocs = static_cast<std::size_t>(group.size());
  const auto rank = static_cast<std::size_t>(group.rank());

  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (rank % (2 * i) == 0) {
      const int source_rank = static_cast<int>(rank + i);
      if (source_rank < static_cast<int>(numprocs)) {
        int size = 0;
        group.recv(source_rank, 0, size);
        if (size > 0) {
          std::vector<int> buf(size);
          group.recv(source_rank, 0, buf.data(), size);
          std::vector<int> merged;
          merged.reserve(procchunk_.size() + size);
          std::merge(procchunk_.begin(), procchunk_.end(), buf.begin(), buf.end(), std::back_inserter(merged));
          procchunk_ = std::move(merged);
        }
      }
    } else if (rank >= i && (rank - i) % (2 * i) == 0) {
      const int master_rank = static_cast<int>(rank - i);
      const int size = static_cast<int>(procchunk_.size());
      group.send(master_rank, 0, size);
      if (size > 0) {
        group.send(master_rank, 0, procchunk_.data(), size);
      }
      break;
    }
  }
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  std::size_t totalsize{};
  if (world_.rank() == 0) {
    totalsize = input_.size();
  }
  boost::mpi::broadcast(world_, totalsize, 0);

  if (totalsize == 0) {
    return true;
  }

  const auto numprocs = static_cast<std::size_t>(world_.size());

  if (world_.rank() == 0) {
    std::vector<std::span<int>> procchunks = Distribute(input_, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());

    for (int i = 1; i < static_cast<int>(numprocs); i++) {
      const auto &chunk = procchunks[i];
      const int chunksize = static_cast<int>(chunk.size());
      world_.send(i, 0, chunksize);
      if (chunksize > 0) {
        world_.send(i, 0, chunk.data(), chunksize);
      }
    }
  } else {
    int chunksize{};
    world_.recv(0, 0, chunksize);
    procchunk_.resize(chunksize);
    if (chunksize > 0) {
      world_.recv(0, 0, procchunk_.data(), chunksize);
    }
  }

  if (!procchunk_.empty()) {
    const auto available_threads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
    const auto numthreads = std::min(available_threads, std::size_t(8));

    if (numthreads <= 1) {
      RadixSort(procchunk_);
    } else {
      std::vector<std::span<int>> chunks = Distribute(procchunk_, numthreads);
      std::vector<std::vector<int>> thread_results(numthreads);

      for (size_t i = 0; i < numthreads; ++i) {
        if (!chunks[i].empty()) {
          thread_results[i].assign(chunks[i].begin(), chunks[i].end());
        }
      }

#pragma omp parallel for num_threads(static_cast<int>(numthreads)) schedule(static)
      for (int i = 0; i < static_cast<int>(numthreads); i++) {
        if (!thread_results[i].empty()) {
          std::span<int> thread_span{thread_results[i]};
          RadixSort(thread_span);
        }
      }

      SafeMerge(procchunk_, thread_results);
    }
  }

  Squash(world_);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (procchunk_.size() != input_.size()) {
      return false;
    }
    if (!output_.empty()) {
      std::ranges::copy(procchunk_, output_.begin());
      std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
    }
  }
  return true;
}
