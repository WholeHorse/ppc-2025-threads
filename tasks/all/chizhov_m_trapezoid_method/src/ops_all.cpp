#include "all/chizhov_m_trapezoid_method/include/ops_all.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // IWYU pragma: keep
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

double chizhov_m_trapezoid_method_all::TrapezoidMethod(Function& f, size_t div, size_t dim,
                                                       std::vector<double>& lower_limits,
                                                       std::vector<double>& upper_limits,
                                                       const boost::mpi::communicator& world) {
  int int_dim = static_cast<int>(dim);
  std::vector<double> h(int_dim);
  std::vector<int> steps(int_dim);

  for (int i = 0; i < int_dim; i++) {
    steps[i] = static_cast<int>(div);
    h[i] = (upper_limits[i] - lower_limits[i]) / steps[i];
  }

  long long total_nodes = 1;
  for (int i = 0; i < int_dim; ++i) {
    total_nodes *= (steps[i] + 1);
  }

  int rank = world.rank();
  int size = world.size();

  long long base_count = total_nodes / size;
  long long remainder = total_nodes % size;

  long start = (rank < remainder) ? (rank * (base_count + 1))
                                  : (remainder * (base_count + 1)) + ((rank - remainder) * base_count);

  long end = start + base_count + (rank < remainder ? 1 : 0);

  double local_result = 0.0;

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    local_result = oneapi::tbb::parallel_reduce(
        tbb::blocked_range<long>(start, end, 16), 0.0,
        [&](const tbb::blocked_range<long>& r, double local_res) {
          for (long i = r.begin(); i != r.end(); ++i) {
            int temp = static_cast<int>(i);
            double weight = 1.0;
            std::vector<double> point(int_dim);

            for (int j = 0; j < int_dim; j++) {
              int node_index = temp % (steps[j] + 1);
              point[j] = lower_limits[j] + node_index * h[j];
              temp /= (steps[j] + 1);
            }

            for (int j = 0; j < int_dim; j++) {
              if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
                weight *= 1.0;
              } else {
                weight *= 2.0;
              }
            }

            local_res += weight * f(point);
          }
          return local_res;
        },
        [](double a, double b) { return a + b; });
  });

  double global_result = 0.0;
  boost::mpi::reduce(world, local_result, global_result, std::plus<>(), 0);

  if (rank == 0) {
    for (int i = 0; i < int_dim; ++i) {
      global_result *= h[i] / 2.0;
    }
    return std::round(global_result * 100.0) / 100.0;
  }

  return 0.0;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    div_ = *divisions_ptr;

    int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    dim_ = *dimension_ptr;

    auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    int lim_int = static_cast<int>(task_data->inputs_count[2]);
    for (int i = 0; i < lim_int; i += 2) {
      lower_limits_.push_back(limit_ptr[i]);
      upper_limits_.push_back(limit_ptr[i + 1]);
    }
  }

  return true;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    auto* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    auto* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    if (*divisions_ptr <= 0 || *dimension_ptr <= 0) {
      valid = false;
    }
    if (task_data->inputs_count[2] % 2 != 0) {
      valid = false;
    }
    auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    int lim_int = static_cast<int>(task_data->inputs_count[2]);
    for (int i = 0; i < lim_int; i += 2) {
      if (limit_ptr[i] >= limit_ptr[i + 1]) {
        valid = false;
      }
    }
  }

  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

void chizhov_m_trapezoid_method_all::TestTaskMPI::SetFunc(Function f) { f_ = std::move(f); };

bool chizhov_m_trapezoid_method_all::TestTaskMPI::RunImpl() {
  boost::mpi::broadcast(world_, div_, 0);
  boost::mpi::broadcast(world_, dim_, 0);
  if (world_.rank() != 0) {
    lower_limits_.resize(dim_);
    upper_limits_.resize(dim_);
  }
  boost::mpi::broadcast(world_, lower_limits_, 0);
  boost::mpi::broadcast(world_, upper_limits_, 0);
  res_ = TrapezoidMethod(f_, div_, dim_, lower_limits_, upper_limits_, world_);
  return true;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}