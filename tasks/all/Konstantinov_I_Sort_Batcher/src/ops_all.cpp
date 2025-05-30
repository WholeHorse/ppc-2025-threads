#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <ranges>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace konstantinov_i_sort_batcher_all {
namespace {

uint64_t DoubleToKey(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double KeyToDouble(uint64_t transformed_data) {
  if ((transformed_data >> 63) != 0) {
    transformed_data ^= (1ULL << 63);
  } else {
    transformed_data = ~transformed_data;
  }

  double result = 0.0;
  std::memcpy(&result, &transformed_data, sizeof(result));
  return result;
}

void RadixSort(std::vector<uint64_t>& array, int thread_count) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<std::vector<int>> local_frequencies(thread_count, std::vector<int>(bucket_count, 0));

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::vector<std::thread> threads;
    size_t n = array.size();
    size_t block_size = (n + thread_count - 1) / thread_count;

    for (int t = 0; t < thread_count; ++t) {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);

      threads.emplace_back([&, begin, end, t]() {
        for (size_t i = begin; i < end; ++i) {
          auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
          local_frequencies[t][bucket]++;
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }

    std::vector<int> frequency(bucket_count, 0);
    for (int b = 0; b < bucket_count; ++b) {
      for (int t = 0; t < thread_count; ++t) {
        frequency[b] += local_frequencies[t][b];
        local_frequencies[t][b] = 0;
      }
    }

    for (int i = 1; i < bucket_count; i++) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PreProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank == 0) {
    size_t total = task_data->inputs_count[0];
    size_t per_proc = (total + size - 1) / size;
    mas_.resize(per_proc * size, std::numeric_limits<double>::max());
    auto* src = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(src, src + total, mas_.begin());
  }

  size_t per_proc = 0;
  if (rank == 0) {
    per_proc = mas_.size() / size;
  }
  boost::mpi::broadcast(world_, per_proc, 0);

  std::vector<double> local(per_proc);
  boost::mpi::scatter(world_, mas_, local.data(), static_cast<int>(per_proc), 0);
  mas_.swap(local);

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }

  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  std::vector<uint64_t> local;
  local.reserve(mas_.size());
  for (auto d : mas_) {
    local.push_back(DoubleToKey(d));
  }
  size_t n = local.size();
  const int thread_count = std::max(1, std::min(static_cast<int>(n), ppc::util::GetPPCNumThreads()));
  RadixSort(local, thread_count);

  int stages = static_cast<int>(std::ceil(std::log2(size)));
  for (int stage = 0; stage < stages; ++stage) {
    int offset = 1 << (stages - stage - 1);

    for (int step = offset; step > 0; step >>= 1) {
      int partner = rank ^ step;
      if (partner >= size) {
        continue;
      }

      const int data_size = static_cast<int>(local.size());

      boost::mpi::request reqs[2];
      std::vector<uint64_t> recv_data(local.size());

      if (rank < partner) {
        reqs[0] = world_.isend(partner, 0, local.data(), data_size);
        reqs[1] = world_.irecv(partner, 0, recv_data.data(), data_size);
      } else {
        reqs[0] = world_.irecv(partner, 0, recv_data.data(), data_size);
        reqs[1] = world_.isend(partner, 0, local.data(), data_size);
      }
      reqs[0].wait();
      reqs[1].wait();

      std::vector<uint64_t> merged;
      std::ranges::merge(local, recv_data, std::back_inserter(merged));

      if (rank < partner) {
        local.assign(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(local.size()));
      } else {
        local.assign(merged.end() - static_cast<std::ptrdiff_t>(local.size()), merged.end());
      }
    }
    world_.barrier();
  }

  output_.resize(local.size());
  for (size_t i = 0; i < local.size(); ++i) {
    output_[i] = KeyToDouble(local[i]);
  }

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PostProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();
  const int local_size = static_cast<int>(output_.size());

  std::vector<double> gathered;

  if (rank == 0) {
    gathered.resize(local_size * size);
  }

  boost::mpi::gather(world_, output_.data(), local_size, gathered.data(), 0);

  if (rank == 0) {
    auto removed = std::ranges::remove(gathered, std::numeric_limits<double>::max());
    gathered.erase(removed.begin(), removed.end());

    std::ranges::sort(gathered);

    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(gathered, out_ptr);
  }

  return true;
}