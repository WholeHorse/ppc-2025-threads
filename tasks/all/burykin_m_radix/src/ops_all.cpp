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

  // Handle empty array case
  if (array_size == 0) {
    return true;
  }

  // Split data on rank 0 and distribute directly
  std::vector<int> negatives;
  std::vector<int> positives;
  if (rank == 0) {
    SplitBySign(input_, negatives, positives);
  }

  // Distribute negatives and positives separately to all processes
  local_data_ = DistributeData(negatives, rank, size);
  std::vector<int> local_positives = DistributeData(positives, rank, size);

  // Sort local data
  std::vector<int> sorted_local_negatives;
  std::vector<int> sorted_local_positives;

  if (!local_data_.empty()) {
    RadixSortPositive(local_data_);
    sorted_local_negatives = local_data_;
  }

  if (!local_positives.empty()) {
    RadixSortPositive(local_positives);
    sorted_local_positives = local_positives;
  }

  // Gather results
  std::vector<int> sorted_negatives;
  std::vector<int> sorted_positives;

  if (rank == 0) {
    sorted_negatives = GatherAndMerge(sorted_local_negatives, rank, size);
    sorted_positives = GatherAndMerge(sorted_local_positives, rank, size);

    // Process negatives (reverse and negate)
    if (!sorted_negatives.empty()) {
      std::ranges::reverse(sorted_negatives);
      std::ranges::transform(sorted_negatives, sorted_negatives.begin(), [](int x) { return -x; });
    }

    // Combine results
    output_.clear();
    output_.reserve(sorted_negatives.size() + sorted_positives.size());
    output_.insert(output_.end(), sorted_negatives.begin(), sorted_negatives.end());
    output_.insert(output_.end(), sorted_positives.begin(), sorted_positives.end());
  } else {
    GatherAndMerge(sorted_local_negatives, rank, size);
    GatherAndMerge(sorted_local_positives, rank, size);
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::memcpy(task_data->outputs[0], output_.data(), output_.size() * sizeof(int));
  }

  return true;
}

void burykin_m_radix_all::RadixALL::RadixSortPositive(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max_val = *std::ranges::max_element(arr);
  for (int exp = 1; max_val / exp > 0; exp *= 10) {
    CountingSortByDigit(arr, exp);
  }
}

void burykin_m_radix_all::RadixALL::CountingSortByDigit(std::vector<int>& arr, int exp) {
  const int n = static_cast<int>(arr.size());
  std::vector<int> output(n);
  std::vector<int> count(10, 0);

#pragma omp parallel
  {
    std::vector<int> local_count(10, 0);

#pragma omp for
    for (int i = 0; i < n; i++) {
      local_count[(arr[i] / exp) % 10]++;
    }

#pragma omp critical
    {
      for (int i = 0; i < 10; i++) {
        count[i] += local_count[i];
      }
    }
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (int i = n - 1; i >= 0; i--) {
    int digit = (arr[i] / exp) % 10;
    output[--count[digit]] = arr[i];
  }

  arr = std::move(output);
}

void burykin_m_radix_all::RadixALL::SplitBySign(const std::vector<int>& arr, std::vector<int>& negatives,
                                                std::vector<int>& positives) {
  negatives.clear();
  positives.clear();

  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(-num);  // Store as positive for radix sort
    } else {
      positives.push_back(num);
    }
  }
}

std::vector<int> burykin_m_radix_all::RadixALL::DistributeData(const std::vector<int>& data, int rank, int size) {
  std::vector<int> local_data;

  if (rank == 0) {
    // Calculate chunk sizes
    const size_t chunk_size = data.empty() ? 0 : data.size() / size;
    const size_t remainder = data.empty() ? 0 : data.size() % size;

    std::vector<std::vector<int>> chunks(size);

    if (!data.empty()) {
      size_t start = 0;
      for (int i = 0; i < size; i++) {
        size_t current_chunk_size = chunk_size + (i < static_cast<int>(remainder) ? 1 : 0);
        size_t end = start + current_chunk_size;

        if (current_chunk_size > 0 && end <= data.size()) {
          chunks[i].assign(data.begin() + static_cast<ptrdiff_t>(start), data.begin() + static_cast<ptrdiff_t>(end));
        }
        start = end;
      }
    }

    // Send chunks to other processes (even if empty)
    for (int i = 1; i < size; i++) {
      world_.send(i, 0, chunks[i]);
    }

    local_data = std::move(chunks[0]);
  } else {
    // Non-root processes always receive their chunk (may be empty)
    world_.recv(0, 0, local_data);
  }

  return local_data;
}

std::vector<int> burykin_m_radix_all::RadixALL::GatherAndMerge(const std::vector<int>& local_sorted, int rank,
                                                               int size) {
  if (rank == 0) {
    std::vector<std::vector<int>> all_chunks;
    all_chunks.reserve(size);

    // Add rank 0's data first
    all_chunks.push_back(local_sorted);

    // Receive from other processes
    for (int i = 1; i < size; i++) {
      std::vector<int> received_chunk;
      world_.recv(i, 0, received_chunk);
      all_chunks.push_back(std::move(received_chunk));
    }

    // Merge all chunks
    std::vector<int> result;
    for (const auto& chunk : all_chunks) {
      if (!chunk.empty()) {
        if (result.empty()) {
          result = chunk;
        } else {
          result = MergeTwoSorted(result, chunk);
        }
      }
    }

    return result;
  }

  // Non-root processes send their data
  world_.send(0, 0, local_sorted);
  return {};
}

std::vector<int> burykin_m_radix_all::RadixALL::MergeTwoSorted(const std::vector<int>& left,
                                                               const std::vector<int>& right) {
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