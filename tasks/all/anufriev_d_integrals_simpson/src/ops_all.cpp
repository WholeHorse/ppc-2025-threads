#include "all/anufriev_d_integrals_simpson/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#define OMPI_SKIP_MPICXX
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace {

struct ParsedRootInput {
  int dimension = 0;
  std::vector<double> a_vec;
  std::vector<double> b_vec;
  std::vector<int> n_vec;
  int func_code_val = 0;
  bool parse_successful = false;
};

ParsedRootInput ParseAndValidateOnRoot(const std::shared_ptr<ppc::core::TaskData>& task_data_ptr) {
  ParsedRootInput data;

  if (!task_data_ptr || task_data_ptr->inputs.empty() || task_data_ptr->inputs[0] == nullptr) {
    return data;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data_ptr->inputs[0]);
  size_t in_size_bytes = task_data_ptr->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);

  if (num_doubles < 1) {
    return data;
  }

  int d_parsed = static_cast<int>(in_ptr[0]);
  if (d_parsed < 0) {
    return data;
  }
  data.dimension = d_parsed;

  size_t required_elements = 0;
  if (d_parsed == 0) {
    required_elements = 2;
  } else {
    required_elements = 1 + static_cast<size_t>(3 * data.dimension) + 1;
  }

  if (num_doubles < required_elements) {
    return data;
  }

  data.a_vec.resize(data.dimension);
  data.b_vec.resize(data.dimension);
  data.n_vec.resize(data.dimension);

  int idx_ptr = 1;
  if (d_parsed > 0) {
    data.a_vec.resize(data.dimension);
    data.b_vec.resize(data.dimension);
    data.n_vec.resize(data.dimension);

    for (int i = 0; i < data.dimension; ++i) {
      data.a_vec[i] = in_ptr[idx_ptr++];
      data.b_vec[i] = in_ptr[idx_ptr++];
      double n_double = in_ptr[idx_ptr++];

      if (std::floor(n_double) != n_double || n_double > static_cast<double>(std::numeric_limits<int>::max()) ||
          n_double <= 0.0 || (static_cast<int>(n_double) % 2 != 0)) {
        return data;
      }
      data.n_vec[i] = static_cast<int>(n_double);
    }
  }

  double func_code_double = in_ptr[idx_ptr];
  if (std::floor(func_code_double) != func_code_double ||
      func_code_double > static_cast<double>(std::numeric_limits<int>::max()) ||
      func_code_double < static_cast<double>(std::numeric_limits<int>::min())) {
    return data;
  }
  data.func_code_val = static_cast<int>(func_code_double);

  data.parse_successful = true;
  return data;
}

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}

struct RunParameters {
  size_t total_points = 1;
  double coeff_mult = 1.0;
  bool success = true;
};

RunParameters CalculateRunParameters(int p_dimension, const std::vector<int>& p_n, const std::vector<double>& p_a,
                                     const std::vector<double>& p_b, std::vector<double>& p_steps) {
  RunParameters params;
  if (p_dimension == 0) {
    params.total_points = 0;
    params.coeff_mult = 1.0;
    return params;
  }

  p_steps.resize(p_dimension);
  for (int i = 0; i < p_dimension; i++) {
    if (p_n[i] == 0) {
      params.success = false;
      return params;
    }
    p_steps[i] = (p_b[i] - p_a[i]) / p_n[i];
    params.coeff_mult *= p_steps[i] / 3.0;
    size_t points_in_dim = static_cast<size_t>(p_n[i]) + 1;

    if (params.total_points > std::numeric_limits<size_t>::max() / points_in_dim) {
      params.success = false;
      return params;
    }
    params.total_points *= points_in_dim;
  }
  return params;
}

struct MpiWorkDistribution {
  size_t local_start_k = 0;
  size_t local_end_k = 0;
  size_t num_points_for_this_rank = 0;
};

MpiWorkDistribution DistributeWorkAmongMpiRanks(size_t p_total_points, int p_rank, int p_world_size) {
  MpiWorkDistribution dist;
  if (p_total_points == 0 || p_world_size == 0) {
    return dist;
  }

  size_t points_per_rank_base = p_total_points / static_cast<size_t>(p_world_size);
  size_t remainder_points = p_total_points % static_cast<size_t>(p_world_size);

  if (static_cast<size_t>(p_rank) < remainder_points) {
    dist.num_points_for_this_rank = points_per_rank_base + 1;
    dist.local_start_k = static_cast<size_t>(p_rank) * (points_per_rank_base + 1);
  } else {
    dist.num_points_for_this_rank = points_per_rank_base;
    dist.local_start_k = remainder_points * (points_per_rank_base + 1) +
                         (static_cast<size_t>(p_rank) - remainder_points) * points_per_rank_base;
  }
  dist.local_end_k = dist.local_start_k + dist.num_points_for_this_rank;

  if (dist.local_start_k >= p_total_points) {
    dist.local_start_k = p_total_points;
    dist.local_end_k = p_total_points;
    dist.num_points_for_this_rank = 0;
  } else {
    dist.local_end_k = std::min(dist.local_end_k, p_total_points);
    if (dist.local_start_k < dist.local_end_k) {
      dist.num_points_for_this_rank = dist.local_end_k - dist.local_start_k;
    } else {
      dist.num_points_for_this_rank = 0;
    }
  }
  return dist;
}
}  // namespace

namespace anufriev_d_integrals_simpson_all {

double IntegralsSimpsonAll::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

bool IntegralsSimpsonAll::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ParsedRootInput parsed_input_data;
  int root_preprocessing_status = 0;

  if (rank == 0) {
    parsed_input_data = ParseAndValidateOnRoot(task_data);
    root_preprocessing_status = parsed_input_data.parse_successful ? 1 : 0;
  }

  MPI_Bcast(&root_preprocessing_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (root_preprocessing_status == 0) {
    return false;
  }

  if (rank == 0) {
    dimension_ = parsed_input_data.dimension;
    func_code_ = parsed_input_data.func_code_val;
  }

  MPI_Bcast(&dimension_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&func_code_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (dimension_ == 0) {
    result_ = 0.0;
    return true;
  }

  a_.resize(dimension_);
  b_.resize(dimension_);
  n_.resize(dimension_);

  if (rank == 0) {
    a_ = parsed_input_data.a_vec;
    b_ = parsed_input_data.b_vec;
    n_ = parsed_input_data.n_vec;
    func_code_ = parsed_input_data.func_code_val;
  }

  MPI_Bcast(a_.data(), dimension_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_.data(), dimension_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(n_.data(), dimension_, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    for (int val_n : n_) {
      if (val_n < 0 || (val_n % 2) != 0) {
        return false;
      }
    }
  }

  result_ = 0.0;
  return true;
}

bool IntegralsSimpsonAll::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int validation_status_root = 1;

  if (rank == 0) {
    if (task_data == nullptr || task_data->outputs.empty() || task_data->outputs[0] == nullptr ||
        task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
      validation_status_root = 0;
    }
  }

  MPI_Bcast(&validation_status_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (validation_status_root == 1);
}

bool IntegralsSimpsonAll::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (dimension_ == 0) {
    if (rank == 0) {
      result_ = 0.0;
    }
    return true;
  }

  std::vector<double> steps;
  RunParameters params = CalculateRunParameters(dimension_, n_, a_, b_, steps);

  if (!params.success) {
    return false;
  }

  if (params.total_points == 0) {
    if (rank == 0) {
      result_ = 0.0;
    }
    return true;
  }

  MpiWorkDistribution dist = DistributeWorkAmongMpiRanks(params.total_points, rank, world_size);

  double local_sum = 0.0;
  if (dist.num_points_for_this_rank > 0) {
    local_sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(dist.local_start_k, dist.local_end_k), 0.0,
        [&](const tbb::blocked_range<size_t>& r, double running_sum) {
          std::vector<double> coords(dimension_);
          std::vector<int> current_idx(dimension_);

          for (size_t k_iter = r.begin(); k_iter != r.end(); ++k_iter) {
            double current_coeff_prod = 1.0;
            size_t current_k_val = k_iter;

            for (int dim_idx = 0; dim_idx < dimension_; ++dim_idx) {
              size_t points_in_this_dim = static_cast<size_t>(n_[dim_idx]) + 1;
              size_t index_in_this_dim = current_k_val % points_in_this_dim;
              current_idx[dim_idx] = static_cast<int>(index_in_this_dim);
              current_k_val /= points_in_this_dim;

              coords[dim_idx] = a_[dim_idx] + current_idx[dim_idx] * steps[dim_idx];
              current_coeff_prod *= SimpsonCoeff(current_idx[dim_idx], n_[dim_idx]);
            }
            running_sum += current_coeff_prod * FunctionN(coords);
          }
          return running_sum;
        },
        [](double x, double y) { return x + y; });
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_ = params.coeff_mult * global_sum;
  } else {
    result_ = 0.0;
  }

  return true;
}

bool IntegralsSimpsonAll::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (task_data == nullptr || task_data->outputs.empty() || task_data->outputs[0] == nullptr ||
        task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
      return false;
    }
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    out_ptr[0] = result_;
  }
  return true;
}

}  // namespace anufriev_d_integrals_simpson_all