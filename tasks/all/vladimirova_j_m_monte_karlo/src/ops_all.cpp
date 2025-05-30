
#include "all/vladimirova_j_m_monte_karlo/include/ops_all.hpp"

#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {

thread_local std::mt19937 gen(std::random_device{}());

double CreateRandomVal(double min_v, double max_v) {
  std::uniform_real_distribution<double> dis(min_v, max_v);
  return dis(gen);
}

}  // namespace

bool vladimirova_j_m_monte_karlo_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  func_ = reinterpret_cast<bool (*)(std::vector<double>, size_t)>(task_data->inputs[1]);
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> var_vect = std::vector<double>(in_ptr, in_ptr + var_size_);
  var_size_ /= 2;
  var_integr_ = std::vector<vladimirova_j_m_monte_karlo_all::BoundariesIntegral>(var_size_);
  for (size_t i = 0; i < var_size_; i++) {
    var_integr_[i].min = var_vect[i * 2];
    var_integr_[i].max = var_vect[(i * 2) + 1];
  }
  accuracy_ = reinterpret_cast<size_t>(task_data->inputs[2]);
  return true;
}

bool vladimirova_j_m_monte_karlo_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  var_size_ = task_data->inputs_count[0];
  if ((var_size_ % 2 != 0) || (var_size_ < 3)) {
    return false;
  }
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> var_vect = std::vector<double>(in_ptr, in_ptr + var_size_);
  for (size_t i = 0; i < var_size_; i += 2) {
    if (var_vect[i] >= var_vect[i + 1]) {
      return false;
    }
  }
  return (task_data->inputs[1] != nullptr) && (reinterpret_cast<size_t>(task_data->inputs[2]) > 0);
}

bool vladimirova_j_m_monte_karlo_all::TestTaskALL::RunImpl() {
  // Multiply matrices
  size_t count_t = ppc::util::GetPPCNumThreads();
  if (count_t == 0 || world_.size() == 0) {
    return false;
  }

  size_t successful_point = 0;
  size_t local_accuracy = accuracy_ / world_.size();
  if (world_.rank() == 0) {
    local_accuracy += accuracy_ % world_.size();
  }
  size_t chank = var_size_ / world_.size();
  size_t global_successful_point = 0;
  double global_s = 1;

  std::vector<double> random_val = std::vector<double>(var_size_);

  std::vector<std::thread> threads(count_t);
  std::vector<size_t> local_res(count_t, 0);

  size_t dl = local_accuracy / count_t;

  for (size_t i = 1; i < count_t; i++) {
    local_res[i] = dl;
  }
  local_res[0] = dl + (local_accuracy % count_t);

  for (size_t t = 0; t < count_t; t++) {
    threads[t] = std::thread([&, t]() {
      std::vector<double> random_val = std::vector<double>(var_size_);
      size_t n = local_res[t];
      local_res[t] = 0;
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < var_size_; j++) {
          random_val[j] = CreateRandomVal(var_integr_[j].min, var_integr_[j].max);
        }
        local_res[t] += (int)(func_(random_val, var_size_));
      }
    });
  }

  for (size_t i = 0; i < count_t; i++) {
    threads[i].join();
    successful_point += local_res[i];
  }

  double s = 1;
  size_t begin = world_.rank() * chank;
  size_t end = (world_.rank() == world_.size() - 1) ? var_size_ : (world_.rank() + 1) * chank;
  for (size_t i = begin; i < end; i++) {
    s *= (var_integr_[i].max - var_integr_[i].min);
  }

  boost::mpi::all_reduce(world_, s, global_s, std::multiplies<>());
  boost::mpi::all_reduce(world_, successful_point, global_successful_point, std::plus<>());

  if (world_.rank() == 0) {
    global_s *= ((double)(global_successful_point) / (double)accuracy_);
    output_.push_back(global_s);
  }
  return true;
}

bool vladimirova_j_m_monte_karlo_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  reinterpret_cast<double*>(task_data->outputs[0])[0] = output_[0];
  return true;
}
