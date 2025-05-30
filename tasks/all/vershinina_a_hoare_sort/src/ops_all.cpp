#include "all/vershinina_a_hoare_sort/include/ops_all.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "mpi.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace {
int Partition(int* s_vec, int first, int last) {
  int i = first - 1;
  int value = s_vec[last];

  for (int j = first; j <= last - 1; j++) {
    if (s_vec[j] <= value) {
      i++;
      std::swap(s_vec[i], s_vec[j]);
    }
  }
  std::swap(s_vec[i + 1], s_vec[last]);
  return i + 1;
}

void HoareSort(int* s_vec, int first, int last) {
  if (first < last) {
    int iter = Partition(s_vec, first, last);
    HoareSort(s_vec, first, iter - 1);
    HoareSort(s_vec, iter + 1, last);
  }
}

void BatcherMergeBlocksStep(int* left_pointer, int& left_size, int* right_pointer, int& right_size) {
  std::inplace_merge(left_pointer, right_pointer, right_pointer + right_size);
  left_size += right_size;
}

void BatcherMerge(int thread_input_size, std::vector<int*>& pointers, std::vector<int>& sizes, int par_if_greater) {
  const int batcherthreads = ppc::util::GetPPCNumThreads();
  for (int batcher_iter = 1, batcher_pack = int(pointers.size()); batcher_pack > 1;
       batcher_iter *= 2, batcher_pack /= 2) {
    tbb::task_arena arena(((thread_input_size / batcher_iter) > par_if_greater) ? batcherthreads : 1);
    arena.execute([&] {
      tbb::parallel_for(tbb::blocked_range<int>(0, batcher_pack / 2), [&](const auto& r) {
        for (int off = r.begin(); off < r.end(); off++) {
          BatcherMergeBlocksStep(pointers[2 * batcher_iter * off], sizes[2 * batcher_iter * off],
                                 pointers[(2 * batcher_iter * off) + batcher_iter],
                                 sizes[(2 * batcher_iter * off) + batcher_iter]);
        }
      });
    });
    if ((batcher_pack / 2) - 1 == 0) {
      BatcherMergeBlocksStep(pointers[0], sizes[sizes.size() - 1], pointers[pointers.size() - 1],
                             sizes[sizes.size() - 1]);
    } else if ((batcher_pack / 2) % 2 != 0) {
      BatcherMergeBlocksStep(
          pointers[2 * batcher_iter * ((batcher_pack / 2) - 2)], sizes[2 * batcher_iter * ((batcher_pack / 2) - 2)],
          pointers[2 * batcher_iter * ((batcher_pack / 2) - 1)], sizes[2 * batcher_iter * ((batcher_pack / 2) - 1)]);
    }
  }
}
}  // namespace

bool vershinina_a_hoare_sort_mpi::TestTaskALL::PreProcessingImpl() {
  if (rank_ != 0) {
    return true;
  }
  input_.assign(reinterpret_cast<int*>(task_data->inputs[0]),
                reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool vershinina_a_hoare_sort_mpi::TestTaskALL::ValidationImpl() {
  return rank_ != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool vershinina_a_hoare_sort_mpi::TestTaskALL::RunImpl() {
  int global_size = static_cast<int>(input_.size());
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!static_cast<bool>(global_size)) {
    return false;
  }

  const auto active_procs_num = std::min(global_size, ws_);
  if (rank_ >= active_procs_num) {
    return false;
  }

  MPI_Comm active_procs_comm{};
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank_, &active_procs_comm);

  if (rank_ != 0) {
    int size_to_receive = 0;
    MPI_Recv(&size_to_receive, 1, MPI_INT, 0, 0, active_procs_comm, MPI_STATUS_IGNORE);
    res_.resize(size_to_receive);
    MPI_Recv(res_.data(), size_to_receive, MPI_INT, 0, 0, active_procs_comm, MPI_STATUS_IGNORE);
  } else {
    const int thread_input_size = global_size / active_procs_num;
    const int thread_input_remainder_size = global_size % active_procs_num;

    std::vector<int> sizes(active_procs_num);
    std::vector<int*> pointers(active_procs_num);
    for (int i = 0; i < active_procs_num; i++) {
      sizes[i] = thread_input_size;
      pointers[i] = input_.data() + (i * thread_input_size);
    }
    sizes[sizes.size() - 1] += thread_input_remainder_size;

    if (pointers.empty() || pointers.empty()) {
      return false;
    }

    res_.assign(pointers[0], pointers[0] + sizes[0]);

    for (int receiver_proc_rank = 1; receiver_proc_rank < active_procs_num; receiver_proc_rank++) {
      MPI_Send(sizes.data() + receiver_proc_rank, 1, MPI_INT, receiver_proc_rank, 0, active_procs_comm);
      MPI_Send(pointers[receiver_proc_rank], 1, MPI_INT, receiver_proc_rank, 0, active_procs_comm);
    }
  }

  const auto local_processing_size = int(res_.size());
  const auto numthreads = std::min(local_processing_size, ppc::util::GetPPCNumThreads());
  int thread_input_size = local_processing_size / numthreads;
  int thread_input_remainder_size = local_processing_size % numthreads;

  std::vector<int*> pointers(numthreads);
  std::vector<int> sizes(numthreads);
  for (int i = 0; i < numthreads; i++) {
    pointers[i] = res_.data() + (i * thread_input_size);
    sizes[i] = thread_input_size;
  }
  sizes[sizes.size() - 1] += thread_input_remainder_size;
  tbb::task_arena arena(numthreads);
  arena.execute([&] {
    tbb::parallel_for(tbb::blocked_range<int>(0, numthreads, 1), [&pointers, &sizes](const auto& r) {
      for (int i = r.begin(); i < r.end(); i++) {
        HoareSort(pointers[i], 0, sizes[i] - 1);
      }
    });
  });
  BatcherMerge(thread_input_size, pointers, sizes, 32);

  for (int i = 1; i < active_procs_num; i *= 2) {
    const auto kk = 2 * i;
    if (rank_ % kk == 0) {
      const int transmitter_rank = rank_ + i;
      if (transmitter_rank < active_procs_num) {
        int arrs = -1;
        MPI_Recv(&arrs, 1, MPI_INT, transmitter_rank, 0, active_procs_comm, MPI_STATUS_IGNORE);
        res_.resize(res_.size() + arrs);
        MPI_Recv(res_.data() + res_.size() - arrs, arrs, MPI_INT, transmitter_rank, 0, active_procs_comm,
                 MPI_STATUS_IGNORE);
        std::ranges::inplace_merge(res_, res_.begin() + int(res_.size()) - arrs);
      }
    } else if ((rank_ % i) == 0) {
      const auto size_buf = int(res_.size());
      MPI_Send(&size_buf, 1, MPI_INT, rank_ - i, 0, active_procs_comm);
      MPI_Send(res_.data(), size_buf, MPI_INT, rank_ - i, 0, active_procs_comm);
      break;
    }
  }

  MPI_Comm_free(&active_procs_comm);

  return true;
}

bool vershinina_a_hoare_sort_mpi::TestTaskALL::PostProcessingImpl() {
  if (rank_ != 0) {
    return true;
  }
  std::ranges::copy(res_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
