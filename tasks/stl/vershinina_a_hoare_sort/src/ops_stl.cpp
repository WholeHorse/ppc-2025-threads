#include "stl/vershinina_a_hoare_sort/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
int Partition(double *s_vec, int first, int last) {
  int i = first - 1;
  double value = s_vec[last];

  for (int j = first; j <= last - 1; j++) {
    if (s_vec[j] <= value) {
      i++;
      std::swap(s_vec[i], s_vec[j]);
    }
  }
  std::swap(s_vec[i + 1], s_vec[last]);
  return i + 1;
}

void HoareSort(double *s_vec, int first, int last) {
  if (first < last) {
    int iter = Partition(s_vec, first, last);
    HoareSort(s_vec, first, iter - 1);
    HoareSort(s_vec, iter + 1, last);
  }
}

void BatcherMergeBlocksStep(double *left_pointer, int &left_size, double *right_pointer, int &right_size) {
  std::inplace_merge(left_pointer, right_pointer, right_pointer + right_size);
  left_size += right_size;
}

void BatcherMerge(int thread_input_size, std::vector<double *> &pointers, std::vector<int> &sizes, int par_if_greater) {
  const int batcherthreads = ppc::util::GetPPCNumThreads();

  std::vector<std::thread> threads(batcherthreads);

  for (int step = 1, pack = int(pointers.size()); pack > 1; step *= 2, pack /= 2) {
    if ((thread_input_size / step) > par_if_greater) {
      for (int k = 0; k < pack / 2; ++k) {
        threads[k] = std::thread(
            [&](int off) {
              BatcherMergeBlocksStep(pointers[2 * step * off], sizes[2 * step * off], pointers[(2 * step * off) + step],
                                     sizes[(2 * step * off) + step]);
            },
            k);
      }
      for (int k = 0; k < pack / 2; ++k) {
        threads[k].join();
      }
    } else {
      for (int off = 0; off < pack / 2; ++off) {
        BatcherMergeBlocksStep(pointers[2 * step * off], sizes[2 * step * off], pointers[(2 * step * off) + step],
                               sizes[(2 * step * off) + step]);
      }
    }
    if ((pack / 2) - 1 == 0) {
      BatcherMergeBlocksStep(pointers[0], sizes[sizes.size() - 1], pointers[pointers.size() - 1],
                             sizes[sizes.size() - 1]);
    } else if ((pack / 2) % 2 != 0) {
      BatcherMergeBlocksStep(pointers[2 * step * ((pack / 2) - 2)], sizes[2 * step * ((pack / 2) - 2)],
                             pointers[2 * step * ((pack / 2) - 1)], sizes[2 * step * ((pack / 2) - 1)]);
    }
  }
}
}  // namespace

bool vershinina_a_hoare_sort_stl::TestTaskSTL::PreProcessingImpl() {
  input_.assign(reinterpret_cast<double *>(task_data->inputs[0]),
                reinterpret_cast<double *>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool vershinina_a_hoare_sort_stl::TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && task_data->inputs.size() == 1 &&
         task_data->inputs_count.size() == 1 && task_data->outputs.size() == 1;
}

bool vershinina_a_hoare_sort_stl::TestTaskSTL::RunImpl() {
  int n = int(input_.size());
  if (n <= 1) {
    return true;
  }
  res_.resize(input_.size());
  std::ranges::copy(input_, res_.begin());

  const auto numthreads = std::min(n, ppc::util::GetPPCNumThreads());
  int thread_input_size = n / numthreads;
  int thread_input_remainder_size = n % numthreads;

  std::vector<double *> pointers(numthreads);
  std::vector<int> sizes(numthreads);
  for (int i = 0; i < numthreads; i++) {
    pointers[i] = res_.data() + (i * thread_input_size);
    sizes[i] = thread_input_size;
  }
  sizes[sizes.size() - 1] += thread_input_remainder_size;

  std::vector<std::thread> threads(numthreads);
  for (int i = 0; i < numthreads; i++) {
    threads[i] = std::thread(HoareSort, pointers[i], 0, sizes[i] - 1);
  }
  for (int i = 0; i < numthreads; i++) {
    threads[i].join();
  }
  BatcherMerge(thread_input_size, pointers, sizes, 32);
  return true;
}

bool vershinina_a_hoare_sort_stl::TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}