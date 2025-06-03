#include "all/burykin_m_radix/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

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

  // Broadcast the input vector size to all processes
  // if (world_.rank() == 0) {
  //   std::cerr << "[DEBUG] input_: ";
  //   for (int x : input_) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  // }

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

  // Broadcast the entire array to all processes first
  std::vector<int> global_data;
  if (rank == 0) {
    global_data = input_;
  } else {
    global_data.resize(array_size);
  }
  boost::mpi::broadcast(world_, global_data, 0);

  // Split into positive and negative numbers
  std::vector<int> negatives;
  std::vector<int> positives;
  SplitBySign(global_data, negatives, positives);

  // Debug output for negatives and positives
  // if (rank == 0) {
  //   std::cerr << "[DEBUG] negatives: ";
  //   for (int x : negatives) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  //   std::cerr << "[DEBUG] positives: ";
  //   for (int x : positives) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  // }

  // Process negatives and positives separately
  std::vector<int> sorted_negatives;
  std::vector<int> sorted_positives;

  // Sort negatives
  if (!negatives.empty()) {
    local_data_ = DistributeData(negatives, rank, size);
    if (!local_data_.empty()) {
      RadixSortPositive(local_data_);  // Sort absolute values
    }
    if (rank == 0) {
      sorted_negatives = GatherAndMerge(local_data_, rank, size);
      // Reverse order and make negative
      std::ranges::reverse(sorted_negatives);
      std::ranges::transform(sorted_negatives, sorted_negatives.begin(), [](int x) { return -x; });
    } else {
      GatherAndMerge(local_data_, rank, size);
    }
  }

  // Sort positives
  if (!positives.empty()) {
    local_data_ = DistributeData(positives, rank, size);
    if (!local_data_.empty()) {
      RadixSortPositive(local_data_);
    }
    if (rank == 0) {
      sorted_positives = GatherAndMerge(local_data_, rank, size);
    } else {
      GatherAndMerge(local_data_, rank, size);
    }
  }

  // Debug output for sorted arrays
  // if (rank == 0) {
  //   std::cerr << "[DEBUG] sorted_negatives: ";
  //   for (int x : sorted_negatives) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  //   std::cerr << "[DEBUG] sorted_positives: ";
  //   for (int x : sorted_positives) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  // }

  // Merge results on rank 0
  if (rank == 0) {
    output_.clear();
    output_.reserve(sorted_negatives.size() + sorted_positives.size());
    output_.insert(output_.end(), sorted_negatives.begin(), sorted_negatives.end());
    output_.insert(output_.end(), sorted_positives.begin(), sorted_positives.end());
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  // Ensure all processes have the same output size
  // if (world_.rank() == 0) {
  //   std::cerr << "[DEBUG] output_: ";
  //   for (int x : output_) {
  //     std::cerr << x << " ";
  //   }
  //   std::cerr << '\n';
  //   std::memcpy(task_data->outputs[0], output_.data(), output_.size() * sizeof(int));
  // }

  return true;
}

// New method for sorting only positive numbers
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

// Count occurrences of each digit using OpenMP
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

  // Transform count array to cumulative sum
  std::partial_sum(count.begin(), count.end(), count.begin());

  // Build output array from right to left
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

void burykin_m_radix_all::RadixALL::MergeResults(std::vector<int>& result, const std::vector<int>& negatives,
                                                 const std::vector<int>& positives) {
  result.clear();
  result.reserve(negatives.size() + positives.size());

  result.insert(result.end(), negatives.begin(), negatives.end());
  result.insert(result.end(), positives.begin(), positives.end());
}

std::vector<int> burykin_m_radix_all::RadixALL::DistributeData(const std::vector<int>& data, int rank, int size) {
  std::vector<int> local_data;

  if (data.empty()) {
    return local_data;  // Return immediately if data is empty
  }

  if (rank == 0) {
    // Calculate chunk sizes
    const size_t chunk_size = data.size() / size;
    const size_t remainder = data.size() % size;

    std::vector<std::vector<int>> chunks(size);

    size_t start = 0;
    for (int i = 0; i < size; i++) {
      size_t current_chunk_size = chunk_size + (i < static_cast<int>(remainder) ? 1 : 0);
      size_t end = start + current_chunk_size;

      if (current_chunk_size > 0) {
        chunks[i].assign(data.begin() + start, data.begin() + end);
      }
      start = end;
    }

    // Send chunks to other processes
    for (int i = 1; i < size; i++) {
      world_.send(i, 0, chunks[i]);
    }

    local_data = std::move(chunks[0]);
  } else {
    // Receive chunk from rank 0
    world_.recv(0, 0, local_data);
  }

  return local_data;
}

std::vector<int> burykin_m_radix_all::RadixALL::GatherAndMerge(const std::vector<int>& local_sorted, int rank,
                                                               int size) {
  if (rank == 0) {
    // Initialize vector of vectors with proper size
    std::vector<std::vector<int>> all_chunks(size);
    all_chunks[0] = local_sorted;  // Copy local_sorted to first chunk

    // Receive sorted chunks from other processes
    for (int i = 1; i < size; i++) {
      std::vector<int> received_chunk;
      world_.recv(i, 0, received_chunk);  // Receive into temporary vector
      if (!received_chunk.empty()) {
        all_chunks[i] = std::move(received_chunk);  // Move valid data to all_chunks
      }
    }

    // Merge all non-empty sorted chunks
    std::vector<int> result;
    for (const auto& chunk : all_chunks) {
      if (!chunk.empty()) {
        result = result.empty() ? chunk : MergeTwoSorted(result, chunk);
      }
    }

    return result;
  } else {
    // Send local sorted data to rank 0
    if (!local_sorted.empty()) {
      world_.send(0, 0, local_sorted);
    }
    return std::vector<int>{};  // Explicitly return empty vector
  }
}

std::vector<int> burykin_m_radix_all::RadixALL::MergeTwoSorted(const std::vector<int>& left,
                                                               const std::vector<int>& right) {
  std::vector<int> result;
  result.reserve(left.size() + right.size());

  size_t i = 0;
  size_t j = 0;

  // Merge the two sorted arrays
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  // Add remaining elements
  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}