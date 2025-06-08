#include "all/burykin_m_radix/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner) - needed for MPI serialization
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                              reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
    output_.resize(input_.size());
  }
  return true;
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  int size = world_.size();
  int rank = world_.rank();

  // Broadcast array size to all processes
  size_t array_size = 0;
  if (rank == 0) {
    array_size = input_.size();
  }
  boost::mpi::broadcast(world_, array_size, 0);

  if (array_size == 0) {
    return true;
  }

  // Distribute data using scatter approach
  local_data_ = DistributeData(input_, rank, size);

  // Sort local data
  if (!local_data_.empty()) {
    RadixSortLocal(local_data_);
  }

  // Gather and merge results using tree-based approach
  if (rank == 0) {
    output_ = GatherAndMerge(local_data_, rank, size);
  } else {
    GatherAndMerge(local_data_, rank, size);
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::memcpy(task_data->outputs[0], output_.data(), output_.size() * sizeof(int));
  }
  return true;
}

void burykin_m_radix_all::RadixALL::RadixSortLocal(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  // Separate negative and positive numbers in parallel
  std::vector<int> negatives;
  std::vector<int> positives;

#pragma omp parallel
  {
    std::vector<int> local_neg;
    std::vector<int> local_pos;

#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
      if (arr[i] < 0) {
        local_neg.push_back(-arr[i]);  // Store as positive
      } else {
        local_pos.push_back(arr[i]);
      }
    }

#pragma omp critical
    {
      negatives.insert(negatives.end(), local_neg.begin(), local_neg.end());
      positives.insert(positives.end(), local_pos.begin(), local_pos.end());
    }
  }

// Sort both parts in parallel
#pragma omp parallel sections
  {
#pragma omp section
    {
      if (!negatives.empty()) {
        RadixSortPositive(negatives);
        // Reverse and negate for correct order
        std::reverse(negatives.begin(), negatives.end());
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(negatives.size()); ++i) {
          negatives[i] = -negatives[i];
        }
      }
    }

#pragma omp section
    {
      if (!positives.empty()) {
        RadixSortPositive(positives);
      }
    }
  }

  // Merge results
  arr.clear();
  arr.reserve(negatives.size() + positives.size());
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void burykin_m_radix_all::RadixALL::RadixSortPositive(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  // Find maximum value in parallel
  int max_val = 0;
#pragma omp parallel
  {
    int local_max = 0;
#pragma omp for
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
      local_max = std::max(arr[i], local_max);
    }
#pragma omp critical
    { max_val = std::max(local_max, max_val); }
  }

  // Process each digit position
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    CountingSortByDigit(arr, exp);
  }
}

void burykin_m_radix_all::RadixALL::CountingSortByDigit(std::vector<int>& arr, int exp) {
  const int n = static_cast<int>(arr.size());
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

  // Parallel counting with thread-local counters
  const int num_threads = omp_get_max_threads();
  std::vector<std::vector<int>> thread_counts(num_threads, std::vector<int>(10, 0));

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();

#pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
      const int digit = (arr[i] / exp) % 10;
      thread_counts[tid][digit]++;
    }
  }

  // Merge thread counts
  for (int t = 0; t < num_threads; ++t) {
    for (int d = 0; d < 10; ++d) {
      count[d] += thread_counts[t][d];
    }
  }

  // Convert to cumulative counts
  std::partial_sum(count.begin(), count.end(), count.begin());

  // Build output array
  for (int i = n - 1; i >= 0; i--) {
    const int digit = (arr[i] / exp) % 10;
    output[--count[digit]] = arr[i];
  }

  arr = std::move(output);
}

std::vector<int> burykin_m_radix_all::RadixALL::DistributeData(const std::vector<int>& data, int rank, int size) {
  std::vector<int> local_data;
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    // Calculate optimal distribution
    const size_t total_size = data.size();
    const size_t base_chunk = total_size / size;
    const size_t remainder = total_size % size;

    size_t offset = 0;
    for (int i = 0; i < size; ++i) {
      send_counts[i] = static_cast<int>(base_chunk + (i < static_cast<int>(remainder) ? 1 : 0));
      displs[i] = static_cast<int>(offset);
      offset += send_counts[i];
    }
  }

  // Broadcast distribution info
  boost::mpi::broadcast(world_, send_counts, 0);

  // Resize local buffer
  local_data.resize(send_counts[rank]);

  // Scatter data efficiently
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      if (i == 0) {
        std::copy(data.begin(), data.begin() + send_counts[0], local_data.begin());
      } else {
        std::vector<int> chunk(data.begin() + displs[i], data.begin() + displs[i] + send_counts[i]);
        world_.send(i, 0, chunk);
      }
    }
  } else {
    world_.recv(0, 0, local_data);
  }

  return local_data;
}

std::vector<int> burykin_m_radix_all::RadixALL::GatherAndMerge(const std::vector<int>& local_sorted, int rank,
                                                               int size) {
  std::vector<int> current_data = local_sorted;

  // Tree-based reduction for better scalability
  for (int step = 1; step < size; step *= 2) {
    if (rank % (2 * step) == 0) {
      // Receiver
      int sender = rank + step;
      if (sender < size) {
        std::vector<int> received_data;
        world_.recv(sender, 0, received_data);
        current_data = MergeTwoSorted(current_data, received_data);
      }
    } else if (rank % step == 0) {
      // Sender
      int receiver = rank - step;
      world_.send(receiver, 0, current_data);
      break;
    }
  }

  return current_data;
}

std::vector<int> burykin_m_radix_all::RadixALL::MergeTwoSorted(const std::vector<int>& left,
                                                               const std::vector<int>& right) {
  if (left.empty()) {
    return right;
  }
  if (right.empty()) {
    return left;
  }

  std::vector<int> result;
  result.reserve(left.size() + right.size());

  size_t i = 0;
  size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

void burykin_m_radix_all::RadixALL::SplitBySign(const std::vector<int>& arr, std::vector<int>& negatives,
                                                std::vector<int>& positives) {
  negatives.clear();
  positives.clear();

  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }
}

void burykin_m_radix_all::RadixALL::MergeResults(std::vector<int>& result, const std::vector<int>& negatives,
                                                 const std::vector<int>& positives) {
  result.clear();
  result.reserve(negatives.size() + positives.size());
  result.insert(result.end(), negatives.begin(), negatives.end());
  result.insert(result.end(), positives.begin(), positives.end());
}
