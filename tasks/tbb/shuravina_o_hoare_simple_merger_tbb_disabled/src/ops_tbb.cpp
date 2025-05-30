#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "oneapi/tbb/parallel_invoke.h"

namespace shuravina_o_hoare_simple_merger_tbb {

HoareSortTBB::HoareSortTBB(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

bool HoareSortTBB::Validation() { return ValidationImpl(); }

bool HoareSortTBB::PreProcessing() { return PreProcessingImpl(); }

bool HoareSortTBB::Run() { return RunImpl(); }

bool HoareSortTBB::PostProcessing() {
  bool result = PostProcessingImpl();
  data_.clear();
  data_.shrink_to_fit();
  return result;
}

bool HoareSortTBB::ValidationImpl() {
  if (!task_data) {
    return false;
  }
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if ((task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }
  return task_data->inputs_count.size() == 1 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool HoareSortTBB::PreProcessingImpl() {
  try {
    if (!ValidationImpl()) {
      return false;
    }

    auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
    data_ = std::vector<int>(input_data, input_data + task_data->inputs_count[0]);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "PreProcessing error: " << e.what() << '\n';
    return false;
  } catch (...) {
    std::cerr << "Unknown PreProcessing error\n";
    return false;
  }
}

bool HoareSortTBB::RunImpl() {
  if (data_.empty()) {
    return true;
  }

  try {
    ParallelQuickSort(data_.data(), 0, data_.size() - 1);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Run error: " << e.what() << '\n';
    return false;
  } catch (...) {
    std::cerr << "Unknown Run error\n";
    return false;
  }
}

bool HoareSortTBB::PostProcessingImpl() {
  try {
    if (!ValidationImpl()) {
      return false;
    }

    auto* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(data_, output_data);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "PostProcessing error: " << e.what() << '\n';
    return false;
  } catch (...) {
    std::cerr << "Unknown PostProcessing error\n";
    return false;
  }
}

std::size_t HoareSortTBB::Partition(int* arr, std::size_t left, std::size_t right) {
  if (right == 0) {
    return 0;
  }

  int pivot = arr[(left + right) / 2];
  while (left <= right) {
    while (arr[left] < pivot) {
      ++left;
    }
    while (arr[right] > pivot) {
      if (right == 0) {
        break;
      }
      --right;
    }
    if (left <= right) {
      std::swap(arr[left], arr[right]);
      ++left;
      if (right > 0) {
        --right;
      }
    }
  }
  return left;
}

void HoareSortTBB::SequentialQuickSort(int* arr, std::size_t left, std::size_t right) {
  if (left >= right) {
    return;
  }

  std::size_t p = Partition(arr, left, right);

  if (p > 0 && left < p - 1) {
    SequentialQuickSort(arr, left, p - 1);
  }
  if (p < right) {
    SequentialQuickSort(arr, p, right);
  }
}

void HoareSortTBB::ParallelQuickSort(int* arr, std::size_t left, std::size_t right) {
  if (left >= right) {
    return;
  }

  if (right - left < kThreshold) {
    SequentialQuickSort(arr, left, right);
    return;
  }

  std::size_t p = Partition(arr, left, right);

  try {
    if (p > 0 && left < p - 1 && p < right) {
      tbb::parallel_invoke([&] { ParallelQuickSort(arr, left, p - 1); }, [&] { ParallelQuickSort(arr, p, right); });
    } else if (p > 0 && left < p - 1) {
      ParallelQuickSort(arr, left, p - 1);
    } else if (p < right) {
      ParallelQuickSort(arr, p, right);
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception in ParallelQuickSort: " << e.what() << '\n';
    throw;
  } catch (...) {
    std::cerr << "Unknown exception in ParallelQuickSort\n";
    throw;
  }
}

}  // namespace shuravina_o_hoare_simple_merger_tbb