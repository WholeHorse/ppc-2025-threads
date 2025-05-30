#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
// false positive, comes from <vector> first included from the header
#include "../include/ops_all.hpp"
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "mpi.h"

namespace {
auto Translate(double e, std::size_t i) {
  const uint64_t mask = 1ULL << ((sizeof(uint64_t) * 8) - 1);
  const union {
    double dbl;
    uint64_t uint64;
  } uni{.dbl = e};
  const uint64_t u = ((uni.uint64 & mask) == 0U) ? (uni.uint64 | mask) : (~uni.uint64);
  return (u >> (i * 8)) & 0xFF;
}

void Radix(double *p, int len) {
  std::vector<double> tmpb(len);
  double *tmp = tmpb.data();
  std::vector<std::size_t> cnt(1 << 8, 0);

  for (std::size_t i = 0; i < sizeof(double); ++i) {
    for (auto &c : cnt) {
      c = 0;
    }
    for (int j = 0; j < len; j++) {
      ++cnt[Translate(p[j], i)];
    }
    for (std::size_t j = 1; j < cnt.size(); ++j) {
      cnt[j] += cnt[j - 1];
    }
    for (int j = len; j-- > 0;) {
      auto &e = p[j];
      const auto nidx = --cnt[Translate(e, i)];
      tmp[nidx] = e;
    }
    std::swap(p, tmp);
  }
}

void OddEvenBatcherMergeBlocksStep(std::pair<double *, int> &left, std::pair<double *, int> &right) {
  std::inplace_merge(left.first, right.first, right.first + right.second);
  left.second += right.second;
}

void ParallelOddEvenBatcherMerge(int bsz, std::vector<std::pair<double *, int>> &vb, int par_if_greater) {
  for (int step = 1, pack = int(vb.size()); pack > 1; step *= 2, pack /= 2) {
#pragma omp parallel for if ((bsz / step) > par_if_greater)
    for (int off = 0; off < pack / 2; ++off) {
      OddEvenBatcherMergeBlocksStep(vb[2 * step * off], vb[(2 * step * off) + step]);
    }
    if ((pack / 2) - 1 == 0) {
      OddEvenBatcherMergeBlocksStep(vb.front(), vb.back());
    } else if ((pack / 2) % 2 != 0) {
      OddEvenBatcherMergeBlocksStep(vb[2 * step * ((pack / 2) - 2)], vb[2 * step * ((pack / 2) - 1)]);
    }
  }
}

void ParallelIntraprocessSort(std::span<double> arr) {
  const int sz = static_cast<int>(arr.size());
  const int thr = std::min(sz, ppc::util::GetPPCNumThreads());

  const int bsz = sz / thr;
  const int bex = sz % thr;

  std::vector<std::pair<double *, int>> vb(thr);
  for (int i = 0; i < thr; i++) {
    vb[i] = std::make_pair(arr.data() + (i * bsz), bsz);  // NOLINT(*narrow*)
  }
  vb[vb.size() - 1].second += bex;

#pragma omp parallel for
  for (int i = 0; i < thr; i++) {
    const auto &[p, l] = vb[i];
    Radix(p, l);
  }

  ParallelOddEvenBatcherMerge(bsz, vb, 33);
}

bool ParallelInterprocessScatter(std::span<double> arr, MPI_Comm *newcomm, std::vector<double> *outv) {
  int processes{};
  int rank{};
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int sz = [&] {
    int outsz = static_cast<int>(arr.size());
    MPI_Bcast(&outsz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return outsz;
  }();
  if (0 == sz) {
    return false;
  }

  const int thr = std::min(sz, processes);

  if (rank >= thr) {
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, newcomm);
    return false;
  }
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, newcomm);

  if (rank == 0) {
    const int bsz = sz / thr;
    const int bex = sz % thr;

    std::vector<std::pair<double *, int>> vb(thr);
    for (int i = 0; i < thr; i++) {
      vb[i] = std::make_pair(arr.data() + (i * bsz), bsz);  // NOLINT(*narrow*)
    }
    vb[vb.size() - 1].second += bex;

    if (vb.empty()) {
      return false;
    }

    auto &fvb = vb[0];
    outv->assign(fvb.first, fvb.first + fvb.second);
    for (int r = 1; r < thr; r++) {
      const auto &[ptr, ptrsz] = vb[r];
      MPI_Send(&ptrsz, 1, MPI_INT, r, 0, *newcomm);
      MPI_Send(ptr, ptrsz, MPI_DOUBLE, r, 0, *newcomm);
    }
  } else {
    int recvsz{};
    MPI_Recv(&recvsz, 1, MPI_INT, 0, 0, *newcomm, MPI_STATUS_IGNORE);
    outv->resize(recvsz);
    MPI_Recv(outv->data(), recvsz, MPI_DOUBLE, 0, 0, *newcomm, MPI_STATUS_IGNORE);
  }

  return true;
}
void ParallelInterprocessGather(MPI_Comm *comm, std::vector<double> &arr) {
  int processes{};
  int rank{};
  MPI_Comm_size(*comm, &processes);
  MPI_Comm_rank(*comm, &rank);

  for (int i = 1; i < processes; i *= 2) {
    if (rank % (2 * i) == 0) {
      const int transmitter_rank = rank + i;
      if (transmitter_rank < processes) {
        auto arrs = std::uint64_t{};
        MPI_Recv(&arrs, 1, MPI_UINT64_T, transmitter_rank, 0, *comm, MPI_STATUS_IGNORE);
        const auto insert_pos = std::uint64_t{arr.size()};
        arr.resize(insert_pos + arrs);
        MPI_Recv(arr.data() + insert_pos, int(arrs), MPI_DOUBLE, transmitter_rank, 0, *comm, MPI_STATUS_IGNORE);
        std::ranges::inplace_merge(arr, arr.begin() + int(insert_pos));
      }
    } else if ((rank % i) == 0) {
      const std::uint64_t arrs = arr.size();
      MPI_Send(&arrs, 1, MPI_UINT64_T, rank - i, 0, *comm);
      MPI_Send(arr.data(), int(arrs), MPI_DOUBLE, rank - i, 0, *comm);
      break;
    }
  }
}
}  // namespace

bool petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi::ValidationImpl() {
  return global_rank_ != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi::PreProcessingImpl() {
  if (global_rank_ == 0) {
    in_ = {reinterpret_cast<double *>(task_data->inputs[0]),
           reinterpret_cast<double *>(task_data->inputs[0]) + task_data->inputs_count[0]};
  }
  return true;
}

bool petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi::RunImpl() {
  MPI_Comm comm{};
  std::vector<double> part;

  if (!ParallelInterprocessScatter(in_, &comm, &part)) {
    MPI_Comm_free(&comm);
    return true;
  }
  ParallelIntraprocessSort(part);
  ParallelInterprocessGather(&comm, part);

  MPI_Comm_free(&comm);

  if (global_rank_ == 0) {
    res_ = std::move(part);
  }

  return true;
}

bool petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi::PostProcessingImpl() {
  if (global_rank_ == 0) {
    std::ranges::copy(res_, reinterpret_cast<double *>(task_data->outputs[0]));
  }
  return true;
}
