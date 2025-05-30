#include "../include/ops_all.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/util/include/util.hpp"

namespace {
void RadixIntegerSortHomogenous(std::span<int> arr) {
  using TInt = std::decay_t<decltype(arr)>::value_type;

  const auto convert = [](TInt num, std::size_t i) {
    constexpr TInt kMask = 1 << ((sizeof(TInt) * CHAR_BIT) - 1);
    num = ((num & kMask) == 0U) ? (num | kMask) : (~num);
    return (num >> (i * 8)) & 0xFF;
  };

  std::vector<TInt> bufbuf(arr.size());
  std::span<TInt> buf{bufbuf};
  std::size_t counts[1 << 8];

  for (std::size_t pp = 0; pp < sizeof(TInt); ++pp) {
    std::ranges::fill(counts, 0);
    for (auto &num : arr) {
      ++counts[convert(num, pp)];
    }
    std::partial_sum(std::begin(counts), std::end(counts), std::begin(counts));
    for (auto &num : std::views::reverse(arr)) {
      buf[--counts[convert(num, pp)]] = num;
    }
    std::swap(arr, buf);
  }
}

std::vector<int> RadixIntegerSort(std::span<int> arr) {
  std::vector<int> res(arr.size());

  std::size_t left = 0;
  std::size_t right = arr.size();
  for (auto num : arr) {
    if (num < 0) {
      res[left++] = num;
    } else {
      res[--right] = num;
    }
  }

  // NOLINTBEGIN(bugprone-narrowing-conversions) :: std::size_t -> std::vector<T>::difference_type
  std::span<int> neg(res.begin(), res.begin() + left);
  std::span<int> pos(res.begin() + right, res.end());
  // NOLINTEND(bugprone-narrowing-conversions)

  RadixIntegerSortHomogenous(neg);
  RadixIntegerSortHomogenous(pos);

  std::ranges::reverse(neg);
  return res;
}
}  // namespace

bool koshkin_m_radix_int_simple_merge::AllT::PreProcessingImpl() {
  if (boost::mpi::communicator{}.rank() != 0) {
    return true;
  }

  const auto &[src, cnt] = std::pair(reinterpret_cast<int *>(task_data->inputs[0]), task_data->inputs_count[0]);
  if (cnt == 0) {
    return true;
  }

  in_ = {src, src + cnt};

  return true;
}

bool koshkin_m_radix_int_simple_merge::AllT::ValidationImpl() {
  if (boost::mpi::communicator{}.rank() != 0) {
    return true;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

namespace {
template <typename T>
struct Partitioned {
  std::size_t parts;
  T value;
};

template <typename T>
Partitioned<std::vector<std::span<T>>> Explode(std::span<T> v, std::size_t parts) {
  const std::size_t per = v.size() / parts;
  const std::size_t ext = v.size() % parts;
  auto *ibeg = v.data();
  std::vector<std::span<int>> intermediate_blocks(parts);
  for (std::size_t i = 0; i < parts; i++) {
    auto *iend = ibeg + per + ((i < ext) ? 1 : 0);
    intermediate_blocks[i] = std::span{ibeg, iend};
    ibeg = iend;
  }
  return {
      .parts = parts,
      .value = std::move(intermediate_blocks),
  };
}

std::vector<int> LocalParallelSort(std::span<int> v) {
  std::vector<std::vector<int>> blocks;
  blocks.resize(std::min(v.size(), static_cast<std::size_t>(ppc::util::GetPPCNumThreads())));

  if (blocks.empty()) {
    return {};
  }

  auto intermediate_blocks = Explode(v, blocks.size()).value;
  oneapi::tbb::task_arena arena(static_cast<int>(blocks.size()));
  arena.execute([&] {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, tbb::this_task_arena::max_concurrency(), 1),
                              [&](const auto &r) {
                                for (std::size_t i = r.begin(); i < r.end(); i++) {
                                  blocks[i] = RadixIntegerSort(intermediate_blocks[i]);
                                }
                              });
  });

  for (std::size_t blcnt = blocks.size(), dx = 1; blcnt > 1; blcnt /= 2, dx *= 2) {
    const auto bsize = blocks[0].size();
    const auto after = blcnt / 2;

    const auto merge_and_subst = [&](std::vector<int> &a, std::vector<int> &b) {
      std::vector<int> c(a.size() + b.size());
      std::ranges::merge(a, b, c.begin());
      a = std::move(c);
      b.clear();
    };
    arena.execute([&] {
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(
                                    0, after, (bsize > 64) ? after / tbb::this_task_arena::max_concurrency() : bsize),
                                [&](const auto &r) {
                                  for (std::size_t g = r.begin(); g < r.end(); ++g) {
                                    merge_and_subst(blocks[2 * dx * g], blocks[(2 * dx * g) + dx]);
                                  }
                                });
    });

    if (after % 2 != 0) {
      if (after == 1) {
        merge_and_subst(blocks.front(), blocks.back());
      } else {
        merge_and_subst(blocks[2 * dx * (after - 2)], blocks[2 * dx * (after - 1)]);
      }
    }
  }

  return blocks.front();
}

std::vector<int> InterprocessMerge(boost::mpi::communicator &comm, std::vector<int> &&partial) {
  using SizeType = int;
  for (int i = 1; i < comm.size(); i *= 2) {
    if (comm.rank() % (2 * i) == 0) {
      const int transmitter_rank = comm.rank() + i;
      if (transmitter_rank < comm.size()) {
        SizeType sz = 0;
        comm.recv<SizeType>(transmitter_rank, 0, sz);

        const auto swtch = partial.size();
        partial.resize(swtch + sz);
        // NOLINTNEXTLINE(bugprone-narrowing-conversions) :: std::size_t -> std::vector<T>::difference_type
        comm.recv(transmitter_rank, 0, partial.data() + swtch, sz);

        std::ranges::inplace_merge(
            partial,
            partial.begin() +
                swtch);  // NOLINT(bugprone-narrowing-conversions) :: std::size_t -> std::vector<T>::difference_type
      }
    } else if ((comm.rank() % i) == 0) {
      comm.send<SizeType>(comm.rank() - i, 0, SizeType(partial.size()));
      comm.send(comm.rank() - i, 0, partial.data(), static_cast<int>(partial.size()));
      break;
    }
  }
  return partial;
}

constexpr int kProcColorActive = 0;
constexpr int kProcColorInactive = 1;
}  // namespace

bool koshkin_m_radix_int_simple_merge::AllT::RunImpl() {
  boost::mpi::communicator general_comm{};

  std::size_t elements_count = 0;
  if (general_comm.rank() == 0) {
    elements_count = in_.size();
  }
  boost::mpi::broadcast(general_comm, elements_count, 0);

  if (elements_count == 0) {
    return true;
  }

  const auto participants = std::min(elements_count, static_cast<std::size_t>(general_comm.size()));

  if (general_comm.rank() >= static_cast<int>(participants)) {
    general_comm.split(kProcColorInactive);
    return true;
  }
  auto comm = general_comm.split(kProcColorActive);

  using SizeType = int;
  if (comm.rank() == 0) {
    auto intermediate_blocks = Explode(in_, participants).value;
    partial_.assign(intermediate_blocks.begin()->begin(), intermediate_blocks.begin()->end());
    for (std::size_t r = 1; r < participants; r++) {
      const auto &outgoing_partial = intermediate_blocks[r];
      comm.send<SizeType>(static_cast<int>(r), 0, SizeType(outgoing_partial.size()));
      comm.send(static_cast<int>(r), 0, outgoing_partial.data(), static_cast<int>(outgoing_partial.size()));
    }
  } else {
    SizeType s{};
    comm.recv<SizeType>(0, 0, s);
    partial_.resize(s);
    comm.recv(0, 0, partial_.data(), static_cast<int>(s));
  }

  partial_ = InterprocessMerge(comm, LocalParallelSort(partial_));

  return true;
}

bool koshkin_m_radix_int_simple_merge::AllT::PostProcessingImpl() {
  if (boost::mpi::communicator{}.rank() != 0) {
    return true;
  }
  auto *tgt = reinterpret_cast<decltype(partial_)::value_type *>(task_data->outputs[0]);
  std::ranges::copy(partial_, tgt);
  return true;
}
