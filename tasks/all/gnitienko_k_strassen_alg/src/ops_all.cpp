#include "all/gnitienko_k_strassen_alg/include/ops_all.hpp"

#ifdef _MSC_VER
#pragma warning(disable : 4245)
#endif
#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

bool gnitienko_k_strassen_algorithm_all::StrassenAlgAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    size_t input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    input_1_ = std::vector<double>(in_ptr, in_ptr + input_size);

    in_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
    input_2_ = std::vector<double>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<double>(output_size, 0.0);

    size_ = static_cast<int>(std::sqrt(input_size));

    if ((input_size <= 0) || (input_size & (input_size - 1)) != 0) {
      int new_size = static_cast<int>(std::pow(2, std::ceil(std::log2(size_))));
      std::vector<double> extended_input_1(new_size * new_size, 0.0);
      std::vector<double> extended_input_2(new_size * new_size, 0.0);
      output_.resize(new_size * new_size);
      extend_ = new_size - size_;

      for (int i = 0; i < size_; ++i) {
        for (int j = 0; j < size_; ++j) {
          extended_input_1[(i * new_size) + j] = input_1_[(i * size_) + j];
          extended_input_2[(i * new_size) + j] = input_2_[(i * size_) + j];
        }
      }

      input_1_ = std::move(extended_input_1);
      input_2_ = std::move(extended_input_2);

      size_ = new_size;
    }
  }
  return true;
}

bool gnitienko_k_strassen_algorithm_all::StrassenAlgAll::ValidationImpl() {
  return world_.rank() == 0 ? task_data->inputs_count[0] == task_data->outputs_count[0] : true;
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::AddMatrix(const std::vector<double>& a,
                                                                   const std::vector<double>& b, std::vector<double>& c,
                                                                   int size) {
  for (int i = 0; i < size * size; ++i) {
    c[i] = a[i] + b[i];
  }
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::SubMatrix(const std::vector<double>& a,
                                                                   const std::vector<double>& b, std::vector<double>& c,
                                                                   int size) {
  for (int i = 0; i < size * size; ++i) {
    c[i] = a[i] - b[i];
  }
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::TrivialMultiply(const std::vector<double>& a,
                                                                         const std::vector<double>& b,
                                                                         std::vector<double>& c, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      c[(i * size) + j] = 0;
      for (int k = 0; k < size; ++k) {
        c[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
}

void gnitienko_k_strassen_algorithm_all::ParallelizeTasks(const std::vector<std::function<void(int)>>& tasks,
                                                          int avaliable_threads) {
  int total_tasks = (int)tasks.size();
  int current_threads = std::min(avaliable_threads, total_tasks);
  int remaining_threads = std::max(avaliable_threads - current_threads, 0);

  std::vector<std::thread> threads;
  std::vector<int> threads_for_subtask(total_tasks, 0);

  for (int i = 0; i < total_tasks; ++i) {
    threads_for_subtask[i] = (i < remaining_threads) ? 1 : 0;
  }

  for (int i = 0; i < current_threads; ++i) {
    int sub_threads = threads_for_subtask[i];
    threads.emplace_back([&, i, sub_threads]() { tasks[i](sub_threads); });
  }

  for (int i = current_threads; i < total_tasks; ++i) {
    tasks[i](0);
  }

  for (auto& t : threads) {
    t.join();
  }
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::StrassenMultiply(const std::vector<double>& a,
                                                                          const std::vector<double>& b,
                                                                          std::vector<double>& c, int size,
                                                                          int avaliable_threads) {
  if (size <= TRIVIAL_MULTIPLICATION_BOUND_) {
    TrivialMultiply(a, b, c, size);
    return;
  }

  int half_size = size / 2;

  std::vector<double> a11(half_size * half_size);
  std::vector<double> a12(half_size * half_size);
  std::vector<double> a21(half_size * half_size);
  std::vector<double> a22(half_size * half_size);
  std::vector<double> b11(half_size * half_size);
  std::vector<double> b12(half_size * half_size);
  std::vector<double> b21(half_size * half_size);
  std::vector<double> b22(half_size * half_size);

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      a11[(i * half_size) + j] = a[(i * size) + j];
      a12[(i * half_size) + j] = a[(i * size) + j + half_size];
      a21[(i * half_size) + j] = a[((i + half_size) * size) + j];
      a22[(i * half_size) + j] = a[((i + half_size) * size) + j + half_size];

      b11[(i * half_size) + j] = b[(i * size) + j];
      b12[(i * half_size) + j] = b[(i * size) + j + half_size];
      b21[(i * half_size) + j] = b[((i + half_size) * size) + j];
      b22[(i * half_size) + j] = b[((i + half_size) * size) + j + half_size];
    }
  }

  std::vector<double> d(half_size * half_size);
  std::vector<double> d1(half_size * half_size);
  std::vector<double> d2(half_size * half_size);
  std::vector<double> h1(half_size * half_size);
  std::vector<double> h2(half_size * half_size);
  std::vector<double> v1(half_size * half_size);
  std::vector<double> v2(half_size * half_size);

  auto compute_d = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    AddMatrix(a11, a22, temp_a, half_size);
    AddMatrix(b11, b22, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d, half_size, avaliable_threads);
  };

  auto compute_d1 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    SubMatrix(a12, a22, temp_a, half_size);
    AddMatrix(b21, b22, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d1, half_size, avaliable_threads);
  };

  auto compute_d2 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    SubMatrix(a21, a11, temp_a, half_size);
    AddMatrix(b11, b12, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d2, half_size, avaliable_threads);
  };

  auto compute_h1 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    AddMatrix(a11, a12, temp_a, half_size);
    StrassenMultiply(temp_a, b22, h1, half_size, avaliable_threads);
  };

  auto compute_h2 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    AddMatrix(a21, a22, temp_a, half_size);
    StrassenMultiply(temp_a, b11, h2, half_size, avaliable_threads);
  };

  auto compute_v1 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    SubMatrix(b21, b11, temp_b, half_size);
    StrassenMultiply(a22, temp_b, v1, half_size, avaliable_threads);
  };

  auto compute_v2 = [&](int avaliable_threads = 0) {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    SubMatrix(b12, b22, temp_b, half_size);
    StrassenMultiply(a11, temp_b, v2, half_size, avaliable_threads);
  };

  std::vector<std::function<void(int)>> tasks = {compute_d,  compute_d1, compute_d2, compute_h1,
                                                 compute_h2, compute_v1, compute_v2};

  ParallelizeTasks(tasks, avaliable_threads);

  std::vector<double> c11(half_size * half_size);
  std::vector<double> c12(half_size * half_size);
  std::vector<double> c21(half_size * half_size);
  std::vector<double> c22(half_size * half_size);

  AddMatrix(d, d1, c11, half_size);
  AddMatrix(c11, v1, c11, half_size);
  SubMatrix(c11, h1, c11, half_size);

  AddMatrix(v2, h1, c12, half_size);
  AddMatrix(v1, h2, c21, half_size);

  AddMatrix(d, d2, c22, half_size);
  AddMatrix(c22, v2, c22, half_size);
  SubMatrix(c22, h2, c22, half_size);

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      c[(i * size) + j] = c11[(i * half_size) + j];
      c[(i * size) + j + half_size] = c12[(i * half_size) + j];
      c[((i + half_size) * size) + j] = c21[(i * half_size) + j];
      c[((i + half_size) * size) + j + half_size] = c22[(i * half_size) + j];
    }
  }
}

// NOLINTBEGIN(clang-analyzer-optin.mpi.MPI-Checker)
// NOLINTBEGIN(clang-analyzer-optin.cplusplus.UninitializedObject)
void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::MainProcessRun(int half_size, int size_proc, int max_threads) {
  const int block_size = half_size * half_size;
  output_.resize(size_ * size_);

  std::vector<double> a11(block_size);
  std::vector<double> a12(block_size);
  std::vector<double> a21(block_size);
  std::vector<double> a22(block_size);
  std::vector<double> b11(block_size);
  std::vector<double> b12(block_size);
  std::vector<double> b21(block_size);
  std::vector<double> b22(block_size);

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      a11[(i * half_size) + j] = input_1_[(i * size_) + j];
      a12[(i * half_size) + j] = input_1_[(i * size_) + j + half_size];
      a21[(i * half_size) + j] = input_1_[((i + half_size) * size_) + j];
      a22[(i * half_size) + j] = input_1_[((i + half_size) * size_) + j + half_size];

      b11[(i * half_size) + j] = input_2_[(i * size_) + j];
      b12[(i * half_size) + j] = input_2_[(i * size_) + j + half_size];
      b21[(i * half_size) + j] = input_2_[((i + half_size) * size_) + j];
      b22[(i * half_size) + j] = input_2_[((i + half_size) * size_) + j + half_size];
    }
  }

  std::vector<std::vector<double>> as(7, std::vector<double>(block_size));
  std::vector<std::vector<double>> bs(7, std::vector<double>(block_size));

  AddMatrix(a11, a22, as[0], half_size);
  AddMatrix(b11, b22, bs[0], half_size);
  AddMatrix(a21, a22, as[1], half_size);
  bs[1] = b11;
  as[2] = a11;
  SubMatrix(b12, b22, bs[2], half_size);
  as[3] = a22;
  SubMatrix(b21, b11, bs[3], half_size);
  AddMatrix(a11, a12, as[4], half_size);
  bs[4] = b22;
  SubMatrix(a21, a11, as[5], half_size);
  AddMatrix(b11, b12, bs[5], half_size);
  SubMatrix(a12, a22, as[6], half_size);
  AddMatrix(b21, b22, bs[6], half_size);

  std::vector<std::vector<double>> mul_res(7, std::vector<double>(block_size));

  int total_tasks = 7;
  std::vector<std::vector<int>> tasks_for_procs(std::min(size_proc, 7));

  for (int i = 0; i < total_tasks; ++i) {
    tasks_for_procs[i % std::min(size_proc, 7)].push_back(i);
  }

  std::vector<boost::mpi::request> reqs_send;
  std::vector<boost::mpi::request> reqs_recv;

  for (int proc = 1; proc < std::min(size_proc, 7); ++proc) {
    int num_tasks_proc = (int)tasks_for_procs[proc].size();
    reqs_send.push_back(world_.isend(proc, 0, num_tasks_proc));
    reqs_send.push_back(world_.isend(proc, 50, tasks_for_procs[proc]));

    for (int t : tasks_for_procs[proc]) {
      reqs_send.push_back(world_.isend(proc, 101 + (t * 2), as[t]));
      reqs_send.push_back(world_.isend(proc, 102 + (t * 2), bs[t]));
    }
  }
  boost::mpi::wait_all(reqs_send.begin(), reqs_send.end());

  for (int t : tasks_for_procs[0]) {
    StrassenMultiply(as[t], bs[t], mul_res[t], half_size, max_threads);
  }

  for (int proc = 1; proc < std::min(size_proc, 7); ++proc) {
    for (int t : tasks_for_procs[proc]) {
      reqs_recv.push_back(world_.irecv(proc, t + 2, mul_res[t]));
    }
  }
  boost::mpi::wait_all(reqs_recv.begin(), reqs_recv.end());

  std::vector<double> c11(block_size);
  std::vector<double> c12(block_size);
  std::vector<double> c21(block_size);
  std::vector<double> c22(block_size);
  for (int i = 0; i < block_size; ++i) {
    c11[i] = mul_res[0][i] + mul_res[3][i] - mul_res[4][i] + mul_res[6][i];
    c12[i] = mul_res[2][i] + mul_res[4][i];
    c21[i] = mul_res[1][i] + mul_res[3][i];
    c22[i] = mul_res[0][i] - mul_res[1][i] + mul_res[2][i] + mul_res[5][i];
  }

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      output_[(i * size_) + j] = c11[(i * half_size) + j];
      output_[(i * size_) + j + half_size] = c12[(i * half_size) + j];
      output_[((i + half_size) * size_) + j] = c21[(i * half_size) + j];
      output_[((i + half_size) * size_) + j + half_size] = c22[(i * half_size) + j];
    }
  }

  if (extend_ != 0) {
    int original_size = size_ - extend_;
    std::vector<double> res(original_size * original_size);
    for (int i = 0; i < original_size; i++) {
      for (int j = 0; j < original_size; j++) {
        res[(i * original_size) + j] = output_[(i * size_) + j];
      }
    }
    output_ = std::move(res);
  }
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::WorkerProcessRun(int half_size, int max_threads) {
  const int block_size = half_size * half_size;
  int num_tasks = 0;
  world_.recv(0, 0, num_tasks);

  std::vector<int> tasks(num_tasks);
  world_.recv(0, 50, tasks);

  std::vector<std::vector<double>> a_local(num_tasks, std::vector<double>(block_size));
  std::vector<std::vector<double>> b_local(num_tasks, std::vector<double>(block_size));
  std::vector<boost::mpi::request> recv_reqs;

  for (int i = 0; i < num_tasks; ++i) {
    int t = tasks[i];
    recv_reqs.push_back(world_.irecv(0, 101 + (t * 2), a_local[i]));
    recv_reqs.push_back(world_.irecv(0, 102 + (t * 2), b_local[i]));
  }

  boost::mpi::wait_all(recv_reqs.begin(), recv_reqs.end());

  std::vector<boost::mpi::request> send_reqs;

  std::vector<std::vector<double>> c_local(num_tasks, std::vector<double>(block_size));
  for (int i = 0; i < num_tasks; ++i) {
    int t = tasks[i];
    StrassenMultiply(a_local[i], b_local[i], c_local[i], half_size, max_threads);
    send_reqs.push_back(world_.isend(0, t + 2, c_local[i]));
  }

  boost::mpi::wait_all(send_reqs.begin(), send_reqs.end());
}

void gnitienko_k_strassen_algorithm_all::StrassenAlgAll::MPILogic(int rank, int half_size, int size_proc,
                                                                  int max_threads) {
  if (rank == 0) {
    MainProcessRun(half_size, size_proc, max_threads);

  } else if (rank < std::min(size_proc, 7) && rank != 0) {
    WorkerProcessRun(half_size, max_threads);
  }
}

bool gnitienko_k_strassen_algorithm_all::StrassenAlgAll::RunImpl() {
  int rank = world_.rank();
  int size_proc = world_.size();
  int max_threads = ppc::util::GetPPCNumThreads();

  boost::mpi::broadcast(world_, size_, 0);

  const int half_size = size_ / 2;

  MPILogic(rank, half_size, size_proc, max_threads);

  world_.barrier();

  return true;
}
// NOLINTEND(clang-analyzer-optin.cplusplus.UninitializedObject)
// NOLINTEND(clang-analyzer-optin.mpi.MPI-Checker)

bool gnitienko_k_strassen_algorithm_all::StrassenAlgAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}