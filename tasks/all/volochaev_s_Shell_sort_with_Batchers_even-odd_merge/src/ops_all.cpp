#include "all/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_all.hpp"

#include <omp.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <climits>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstdlib>
#include <cstring>
#include <vector>

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::ShellSort(unsigned int start,
                                                                                      unsigned int size) {
  unsigned int n = size;

  int gap = 1;
  while (gap < static_cast<int>(n / 3)) {
    gap = 3 * gap + 1;
  }

  while (gap >= 1) {
    for (unsigned int i = start + gap; i < start + size; ++i) {
      long long temp = loc_[i];
      long long j = i;
      while (j >= start + gap && loc_[j - gap] > temp) {
        loc_[j] = loc_[j - gap];
        j -= gap;
      }
      loc_[j] = temp;
    }
    gap /= 3;
  }

  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::OddEvenMergeOMP(long long int* tmp,
                                                                                            long long int* l,
                                                                                            const long long int* r,
                                                                                            unsigned int len) {
  unsigned int iter_l = 0;
  unsigned int iter_r = 0;
  unsigned int iter_tmp = 0;

  while (iter_l < len && iter_r < len) {
    if (l[iter_l] < r[iter_r]) {
      tmp[iter_tmp] = l[iter_l];
      iter_l += 2;
    } else {
      tmp[iter_tmp] = r[iter_r];
      iter_r += 2;
    }

    iter_tmp += 2;
  }

  while (iter_l < len) {
    tmp[iter_tmp] = l[iter_l];
    iter_l += 2;
    iter_tmp += 2;
  }

  while (iter_r < len) {
    tmp[iter_tmp] = r[iter_r];
    iter_r += 2;
    iter_tmp += 2;
  }

  for (unsigned int i = 0; i < iter_tmp; i += 2) {
    l[i] = tmp[i];
  }
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::FinalMergeOMP(unsigned int n) {
  unsigned int iter_even = 0;
  unsigned int iter_odd = 1;
  unsigned int iter_tmp = 0;

  while (iter_even < n && iter_odd < n) {
    if (loc_[iter_even] < loc_[iter_odd]) {
      loc_tmp_[iter_tmp] = loc_[iter_even];
      iter_even += 2;
    } else {
      loc_tmp_[iter_tmp] = loc_[iter_odd];
      iter_odd += 2;
    }
    iter_tmp++;
  }

  while (iter_even < n) {
    loc_tmp_[iter_tmp] = loc_[iter_even];
    iter_even += 2;
    iter_tmp++;
  }

  while (iter_odd < n) {
    loc_tmp_[iter_tmp] = loc_[iter_odd];
    iter_odd += 2;
    iter_tmp++;
  }
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::BatcherSortOMP() {
  if (static_cast<unsigned int>(ppc::util::GetPPCNumThreads()) > 2 * loc_proc_lenght_) {
    bool ret = ShellSort(0, loc_proc_lenght_);
    memcpy(loc_tmp_.data(), loc_.data(), sizeof(long long int) * loc_proc_lenght_);
    return ret;
  }

  unsigned int effective_num_threads =
      static_cast<int>(std::pow(2, std::floor(std::log2(ppc::util::GetPPCNumThreads()))));
  unsigned int n_by_proc =
      loc_proc_lenght_ +
      (((2 * effective_num_threads) - (loc_proc_lenght_ % (2 * effective_num_threads))) % (2 * effective_num_threads));
  unsigned int loc_lenght = n_by_proc / effective_num_threads;

  loc_.resize(n_by_proc);
  loc_tmp_.resize(n_by_proc);

  for (unsigned int i = loc_proc_lenght_; i < n_by_proc; i++) {
    loc_[i] = LLONG_MAX;
  }

  bool ret1 = true;
  bool ret2 = true;
#pragma omp parallel num_threads(effective_num_threads)
  { ret1 = ret1 && ShellSort(omp_get_thread_num() * loc_lenght, loc_lenght); }

  for (unsigned int i = effective_num_threads; i > 1; i /= 2) {
#pragma omp parallel num_threads(i)
    {
      auto stride = static_cast<unsigned int>(omp_get_thread_num() / 2);
      unsigned int bias = omp_get_thread_num() % 2;
      unsigned int len = loc_lenght * (effective_num_threads / i);

      ret2 =
          ret2 && OddEvenMergeOMP(loc_tmp_.data() + (stride * 2 * len) + bias, loc_.data() + (stride * 2 * len) + bias,
                                  loc_.data() + (stride * 2 * len) + len + bias, len - bias);
    }
  }
  FinalMergeOMP(n_by_proc);

  void* ptr_tmp1 = loc_tmp_.data();
  void* ptr_loc1 = loc_.data();
  memcpy(ptr_loc1, ptr_tmp1, sizeof(long long int) * loc_proc_lenght_);

  return (ret1 && ret2);
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::PreProcessingImpl() {
  boost::mpi::broadcast(world_, n_input_, 0);

  effective_num_procs_ = static_cast<int>(std::pow(2, std::floor(std::log2(world_.size()))));
  auto e_n_f = static_cast<unsigned int>(effective_num_procs_);
  n_ = n_input_ + (((2 * e_n_f) - n_input_ % (2 * e_n_f))) % (2 * e_n_f);
  loc_proc_lenght_ = n_ / effective_num_procs_;

  if (world_.rank() == 0) {
    mas_.resize(n_);
    flag_ = true;
    void* ptr_input = task_data->inputs[0];
    void* ptr_vec = mas_.data();
    memcpy(ptr_vec, ptr_input, sizeof(long long int) * n_input_);
    for (unsigned int i = n_input_; i < n_; i++) {
      mas_[i] = LLONG_MAX;
    }
  }

  loc_.resize(loc_proc_lenght_);
  loc_tmp_.resize(loc_proc_lenght_);
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::RunImpl() {
  boost::mpi::scatter(world_, mas_.data(), loc_.data(), static_cast<int>(loc_proc_lenght_), 0);

  BatcherSortOMP();

  bool ret = true;
  for (unsigned int i = effective_num_procs_; i > 1; i /= 2) {
    if (world_.rank() < static_cast<int>(i)) {
      unsigned int len = loc_proc_lenght_ * (effective_num_procs_ / i);

      ret = ret && OddEvenMergeMPI(len);

      if (world_.rank() > 0 && world_.rank() % 2 == 0) {
        world_.send(world_.rank() / 2, 0, loc_tmp_.data(), 2 * static_cast<int>(len));
      }
      if (world_.rank() > 0 && world_.rank() < static_cast<int>(i) / 2) {
        loc_.resize(2 * len);
        world_.recv(world_.rank() * 2, 0, loc_.data(), 2 * static_cast<int>(len));
      } else if (world_.rank() == 0 && static_cast<int>(i) != 2) {
        void* ptr_tmp = loc_tmp_.data();
        void* ptr_loc = loc_.data();
        memcpy(ptr_loc, ptr_tmp, sizeof(long long int) * 2 * len);
      }
    }
  }
  return ret;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::OddEvenMergeMPI(unsigned int len) {
  if (world_.rank() % 2 == 0) {
    loc_.resize(2 * len);
    loc_tmp_.resize(2 * len);

    world_.recv(world_.rank() + 1, 0, loc_.data() + len, static_cast<int>(len));

    unsigned int iter_l = 0;
    unsigned int iter_r = 0;
    unsigned int iter_tmp = 0;

    while (iter_l < len && iter_r < len) {
      if (loc_[iter_l] < loc_[len + iter_r]) {
        loc_tmp_[iter_tmp] = loc_[iter_l];
        iter_l++;
      } else {
        loc_tmp_[iter_tmp] = loc_[len + iter_r];
        iter_r++;
      }
      iter_tmp++;
    }

    while (iter_l < len) {
      loc_tmp_[iter_tmp] = loc_[iter_l];
      iter_l++;
      iter_tmp++;
    }

    while (iter_r < len) {
      loc_tmp_[iter_tmp] = loc_[len + iter_r];
      iter_r++;
      iter_tmp++;
    }
  } else {
    world_.send(world_.rank() - 1, 0, loc_.data(), static_cast<int>(len));
  }
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    void* ptr_output = task_data->outputs[0];
    void* ptr_loc = loc_tmp_.data();
    memcpy(ptr_output, ptr_loc, sizeof(long long int) * n_input_);
  }
  return true;
}
