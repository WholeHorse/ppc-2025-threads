#include "../include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

  for (std::size_t ib = 0; ib < Bytes<int>(); ++ib) {
    std::ranges::fill(count, 0);

    std::ranges::for_each(v, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]; });

    std::partial_sum(count.begin(), count.end(), count.begin());

    std::ranges::for_each(std::ranges::reverse_view(v),
                          [&](auto el) { aux[--count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]] = el; });

    std::swap(v, aux);
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
}  // namespace

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    std::span<int> src = {reinterpret_cast<int *>(task_data->inputs[0]), task_data->inputs_count[0]};
    input_.assign(src.begin(), src.end());
    output_.reserve(input_.size());
  }
  return true;
}

void burykin_m_radix_all::RadixALL::Squash(boost::mpi::communicator &group) {
  const auto numprocs = static_cast<std::size_t>(group.size());

  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (group.rank() % (2 * i) == 0) {
      const int slave = group.rank() + static_cast<int>(i);
      if (slave < static_cast<int>(numprocs)) {
        int size{};
        group.recv(int(slave), 0, size);

        if (size > 0) {
          const std::size_t threshold = procchunk_.size();
          procchunk_.resize(threshold + size);
          group.recv(int(slave), 0, procchunk_.data() + threshold, size);

          std::ranges::inplace_merge(procchunk_, procchunk_.begin() + std::int64_t(threshold));
        }
      }
    } else if ((group.rank() % i) == 0) {
      const int size = static_cast<int>(procchunk_.size());
      const int master = group.rank() - static_cast<int>(i);
      group.send(master, 0, size);
      if (size > 0) {
        group.send(master, 0, procchunk_.data(), size);
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

  const auto numprocs = std::min<std::size_t>(totalsize, world_.size());

  if (world_.rank() >= int(numprocs)) {
    world_.split(1);
    return true;
  }

  auto group = world_.split(0);

  if (group.rank() == 0) {
    std::vector<std::span<int>> procchunks = Distribute(input_, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());

    for (int i = 1; i < int(procchunks.size()); i++) {
      const auto &chunk = procchunks[i];
      const int chunksize = int(chunk.size());
      group.send(i, 0, chunksize);
      if (chunksize > 0) {
        group.send(i, 0, chunk.data(), chunksize);
      }
    }
  } else {
    int chunksize{};
    group.recv(0, 0, chunksize);
    procchunk_.resize(chunksize);
    if (chunksize > 0) {
      group.recv(0, 0, procchunk_.data(), chunksize);
    }
  }

  if (!procchunk_.empty()) {
    const auto numthreads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
    std::vector<std::span<int>> chunks = Distribute(procchunk_, numthreads);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numthreads); i++) {
      RadixSort(chunks[i]);
    }

    for (std::size_t i = 1; i < numthreads; i *= 2) {
      const auto multithreaded = chunks.front().size() > 48;
      const auto active_threads = numthreads - i;

#pragma omp parallel for if (multithreaded)
      for (int j = 0; j < static_cast<int>(active_threads); j += 2 * static_cast<int>(i)) {
        if (static_cast<std::size_t>(j + i) < chunks.size()) {
          auto &left = chunks[j];
          auto &right = chunks[j + i];

          std::vector<int> merged;
          merged.reserve(left.size() + right.size());

          std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(merged));

          std::copy(merged.begin(), merged.end(), left.begin());

          left = std::span{left.begin(), left.begin() + merged.size()};
        }
      }
    }

    if (!chunks.empty()) {
      procchunk_.assign(chunks[0].begin(), chunks[0].end());
    }
  }

  Squash(group);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(procchunk_, reinterpret_cast<int *>(task_data->outputs[0]));
  }
  return true;
}