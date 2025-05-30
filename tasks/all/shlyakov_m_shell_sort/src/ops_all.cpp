#include "all/shlyakov_m_shell_sort/include/ops_all.hpp"

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

namespace shlyakov_m_shell_sort_all {

bool TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const std::size_t sz = task_data->inputs_count[0];
    auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_.assign(ptr, ptr + sz);
  }
  return true;
}

bool TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[0] == task_data->outputs_count[0]);
  }
  return true;
}

bool TestTaskALL::RunImpl() {
  int size = static_cast<int>(input_.size());
  boost::mpi::broadcast(world_, size, 0);

  int nprocs = world_.size();
  int delta = size / nprocs;
  int extra = size % nprocs;
  std::vector<int> local_sizes(nprocs, delta);
  if (world_.rank() == 0) {
    for (int i = 0; i < extra; ++i) {
      local_sizes[i]++;
    }
  }

  std::vector<int> local_data;
  if (world_.rank() == 0) {
    std::vector<std::vector<int>> chunks;
    int offset = 0;
    for (int i = 0; i < nprocs; ++i) {
      chunks.emplace_back(input_.begin() + offset, input_.begin() + offset + local_sizes[i]);
      offset += local_sizes[i];
    }
    boost::mpi::scatter(world_, chunks, local_data, 0);
  } else {
    boost::mpi::scatter(world_, local_data, 0);
  }

  int n = static_cast<int>(local_data.size());
  if (n > 1) {
    int max_threads = ppc::util::GetPPCNumThreads();
    int threads = std::min(max_threads, n);
    int seg_size = (n + threads - 1) / threads;
    std::vector<int> buffer;

    tbb::task_arena arena(threads);
    arena.execute([&]() {
      tbb::task_group tg;
      for (int t = 0; t < threads; ++t) {
        int l = t * seg_size;
        int r = std::min(n - 1, ((t + 1) * seg_size) - 1);
        tg.run([l, r, &local_data]() { ShellSort(l, r, local_data); });
      }
      tg.wait();

      int end = seg_size - 1;
      for (int t = 1; t < threads; ++t) {
        int r = std::min(n - 1, ((t + 1) * seg_size) - 1);
        Merge(0, end, r, local_data, buffer);
        end = r;
      }
    });
  }

  std::vector<std::vector<int>> gathered;
  boost::mpi::gather(world_, local_data, gathered, 0);

  if (world_.rank() == 0) {
    output_.clear();
    std::vector<int> buffer;
    for (const auto& chunk : gathered) {
      int mid = static_cast<int>(output_.size()) - 1;
      output_.insert(output_.end(), chunk.begin(), chunk.end());
      int right = static_cast<int>(output_.size()) - 1;
      if (mid >= 0 && right > mid) {
        Merge(0, mid, right, output_, buffer);
      }
    }
  }

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int sz = static_cast<int>(output_.size());
    auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    for (int i = 0; i < sz; ++i) {
      out_ptr[i] = output_[i];
    }
  }
  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int gap = 1;
  int len = right - left + 1;
  while (gap <= len / 3) {
    gap = gap * 3 + 1;
  }
  for (; gap > 0; gap /= 3) {
    for (int i = left + gap; i <= right; ++i) {
      int tmp = arr[i];
      int j = i;
      while (j >= left + gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
  }
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  int total = right - left + 1;
  buffer.clear();
  buffer.reserve(total);

  int i = left;
  int j = mid + 1;
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      buffer.push_back(arr[i++]);
    } else {
      buffer.push_back(arr[j++]);
    }
  }
  while (i <= mid) {
    buffer.push_back(arr[i++]);
  }
  while (j <= right) {
    buffer.push_back(arr[j++]);
  }

  for (int k = 0; k < total; ++k) {
    arr[left + k] = buffer[k];
  }
}

}  // namespace shlyakov_m_shell_sort_all
