#include "all/belov_a_radix_sort_with_batcher_mergesort/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/util/include/util.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_all {

constexpr int kDecimalBase = 10;

namespace {
std::vector<std::span<Bigint>> PartitionArray(std::span<Bigint> arr, std::size_t n) {
  std::vector<std::span<Bigint>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto* cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}
}  // namespace

int RadixBatcherMergesortParallelAll::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortParallelAll::Sort(std::span<Bigint> arr) {
  vector<Bigint> pos;
  vector<Bigint> neg;

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

void RadixBatcherMergesortParallelAll::RadixSort(vector<Bigint>& arr, bool invert) {
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

void RadixBatcherMergesortParallelAll::CountingSort(vector<Bigint>& arr, Bigint digit_place) {
  vector<Bigint> output(arr.size());
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

  arr = output;
}

void RadixBatcherMergesortParallelAll::SortParallel(vector<Bigint>& arr, boost::mpi::communicator& comm) {
  size_t totalsize = arr.size();
  boost::mpi::broadcast(comm, totalsize, 0);

  if (totalsize == 0) {
    return;
  }

  const auto numprocs = std::min<std::size_t>(totalsize, comm.size());
  procchunk_.resize(totalsize);

  boost::mpi::communicator group = comm.split(comm.rank() < static_cast<int>(numprocs) ? 0 : 1);

  if (comm.rank() >= static_cast<int>(numprocs)) {
    return;
  }

  if (group.rank() == 0) {
    std::vector<std::span<Bigint>> procchunks = PartitionArray(arr, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());
    for (int i = 1; i < static_cast<int>(procchunks.size()); i++) {
      const auto& chunk = procchunks[i];

      const int chunksize = static_cast<int>(chunk.size());
      group.send(i, 0, chunksize);
      group.send(i, 0, chunk.data(), chunksize);
    }
  } else {
    int chunksize = 0;
    group.recv(0, 0, chunksize);
    procchunk_.resize(chunksize);
    group.recv(0, 0, procchunk_.data(), chunksize);
  }

  const auto numthreads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
  std::vector<std::span<Bigint>> chunks = PartitionArray(procchunk_, numthreads);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(numthreads); i++) {
    Sort(chunks[i]);
  }

  BatcherMergeParallel(procchunk_, static_cast<int>(numthreads));
}

void RadixBatcherMergesortParallelAll::BatcherMergeParallel(vector<Bigint>& arr, int num_threads) {
  size_t n = arr.size();

  if (n == 0) {
    return;
  }

  num_threads = (num_threads < 1) ? 1 : num_threads;
  const auto chunk_size = n / num_threads;
  const auto multithreaded = chunk_size > 64;

  std::vector<std::span<Bigint>> chunks = PartitionArray(arr, num_threads);

  int num_iterations = 0;
  for (int i = 1; i < num_threads; i *= 2) {
    num_iterations++;
  }

  for (int iter = 0; iter < num_iterations; iter++) {
    int i = 1 << iter;
    const auto active_threads = num_threads - i;

#pragma omp parallel for if (multithreaded)
    for (int j = 0; j < static_cast<int>(active_threads); j += 1) {
      int left_idx = j * (2 * i);
      int right_idx = left_idx + i;
      if (right_idx < num_threads) {
        auto& left = chunks[left_idx];
        auto& right = chunks[right_idx];
        std::inplace_merge(left.begin(), left.end(), right.end());
        left = std::span{left.begin(), right.end()};
      }
    }
  }
}

void RadixBatcherMergesortParallelAll::MergeAcross(boost::mpi::communicator& group) {
  const auto numprocs = static_cast<std::size_t>(group.size());
  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (group.rank() % (2 * i) == 0) {
      const int slave = group.rank() + static_cast<int>(i);
      if (slave < static_cast<int>(numprocs)) {
        int size = 0;
        group.recv(slave, 0, size);

        const std::size_t threshold = procchunk_.size();
        procchunk_.resize(threshold + size);
        group.recv(slave, 0, procchunk_.data() + threshold, size);

        std::ranges::inplace_merge(procchunk_.begin(), procchunk_.begin() + static_cast<int64_t>(threshold),
                                   procchunk_.end());
      }
    } else if ((group.rank() % i) == 0) {
      const int size = static_cast<int>(procchunk_.size());
      const int master = group.rank() - static_cast<int>(i);
      group.send(master, 0, size);
      group.send(master, 0, procchunk_.data(), size);
      break;
    }
  }
}

bool RadixBatcherMergesortParallelAll::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);

  if (world_.rank() == 0) {
    array_.assign(input_array_data, input_array_data + n_);
  }

  return true;
}

bool RadixBatcherMergesortParallelAll::ValidationImpl() {
  return (world_.rank() != 0 ||
          (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
           (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty() &&
           world_.size() > 0));
}

bool RadixBatcherMergesortParallelAll::RunImpl() {
  this->SortParallel(array_, world_);
  boost::mpi::communicator group =
      world_.split(world_.rank() < static_cast<int>(std::min<std::size_t>(n_, world_.size())) ? 0 : 1);

  if (world_.rank() < static_cast<int>(std::min<std::size_t>(n_, world_.size()))) {
    MergeAcross(group);
  }

  return true;
}

bool RadixBatcherMergesortParallelAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(procchunk_, reinterpret_cast<Bigint*>(task_data->outputs[0]));
  }
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_all