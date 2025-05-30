#pragma once

#include <oneapi/tbb/parallel_for.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"
#include "core/task/include/task.hpp"

namespace sidorina_p_gradient_method_all {

inline std::vector<double> MultiplyMatrixByVector(boost::mpi::communicator& world, const std::vector<double>& a,
                                                  std::vector<double>& vec, int size) {
  int rank = world.rank();
  int wsize = world.size();
  std::vector<int> rows_per_process(wsize, size / wsize);
  for (int i = 0; i < size % wsize; ++i) {
    rows_per_process[i]++;
  }
  std::vector<int> displs(wsize, 0);
  std::vector<int> sizes(wsize, 0);
  for (int i = 0; i < wsize; ++i) {
    sizes[i] = rows_per_process[i] * size;
    if (i > 0) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }
  int local_rows = rows_per_process[rank];
  std::vector<double> local_matrix(local_rows * size);
  scatterv(world, a.data(), sizes, displs, local_matrix.data(), sizes[rank], 0);
  broadcast(world, vec, 0);
  std::vector<double> local_result(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < size; ++j) {
      local_result[i] += local_matrix[(i * size) + j] * vec[j];
    }
  }
  std::vector<double> result;
  if (rank == 0) {
    result.resize(size);
  }
  std::vector<int> recv_sizes = rows_per_process;
  std::vector<int> recv_displs(wsize, 0);
  for (int i = 1; i < wsize; i++) {
    recv_displs[i] = recv_displs[i - 1] + recv_sizes[i - 1];
  }
  gatherv(world, local_result.data(), int(local_result.size()), result.data(), recv_sizes, recv_displs, 0);
  broadcast(world, result, 0);

  return result;
}

inline double VectorNorm(const std::vector<double>& vec) {
  double sum = 0;
  for (double value : vec) {
    sum += std::pow(value, 2);
  }
  return std::sqrt(sum);
}

inline double Dot(boost::mpi::communicator& world, const std::vector<double>& vec1, const std::vector<double>& vec2) {
  int rank = world.rank();
  int size = world.size();
  int global_size = int(vec1.size());
  std::vector<int> sizes(size, global_size / size);
  std::vector<int> displs(size, 0);
  for (int i = 0; i < global_size % size; ++i) {
    sizes[i]++;
  }
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sizes[i - 1];
  }
  std::vector<double> local_v1(sizes[rank]);
  std::vector<double> local_v2(sizes[rank]);
  scatterv(world, vec1.data(), sizes, displs, local_v1.data(), sizes[rank], 0);
  scatterv(world, vec2.data(), sizes, displs, local_v2.data(), sizes[rank], 0);
  double local_sum = 0.0;
  for (int i = 0; i < int(local_v1.size()); ++i) {
    local_sum += local_v1[i] * local_v2[i];
  }
  double global_sum = 0.0;
  all_reduce(world, local_sum, global_sum, std::plus<>());

  return global_sum;
}

inline double Dot(const std::vector<double>& vec) {
  double sum = 0;
  for (unsigned long i = 0; i < vec.size(); i++) {
    sum += std::pow(vec[i], 2);
  }
  return sum;
}

inline std::vector<double> ConjugateGradientMethod(boost::mpi::communicator& world, std::vector<double>& a,
                                                   std::vector<double>& b, std::vector<double> solution,
                                                   double tolerance, int size) {
  std::vector<double> matrix_times_solution = MultiplyMatrixByVector(world, a, solution, size);

  auto residual = std::vector<double>(size);
  auto direction = std::vector<double>(size);

  tbb::parallel_for(tbb::blocked_range<int>(0, size),
                    [&residual, &b, &matrix_times_solution](const tbb::blocked_range<int>& r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        residual[i] = b[i] - matrix_times_solution[i];
                      }
                    });

  double residual_norm_squared = Dot(residual);
  if (std::sqrt(residual_norm_squared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrix_times_direction(size);

  while (std::sqrt(residual_norm_squared) > tolerance) {
    matrix_times_direction = MultiplyMatrixByVector(world, a, direction, size);
    double direction_dot_matrix_times_direction = Dot(world, direction, matrix_times_direction);
    double alpha = residual_norm_squared / direction_dot_matrix_times_direction;

    tbb::parallel_for(tbb::blocked_range<int>(0, size),
                      [&solution, &alpha, &direction](const tbb::blocked_range<int>& r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          solution[i] += alpha * direction[i];
                        }
                      });

    tbb::parallel_for(tbb::blocked_range<int>(0, size),
                      [&residual, &alpha, &matrix_times_direction](const tbb::blocked_range<int>& r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          residual[i] -= alpha * matrix_times_direction[i];
                        }
                      });

    double new_residual_norm_squared = Dot(residual);
    double beta = new_residual_norm_squared / residual_norm_squared;
    residual_norm_squared = new_residual_norm_squared;

    tbb::parallel_for(tbb::blocked_range<int>(0, size),
                      [&direction, &residual, &beta](const tbb::blocked_range<int>& r) {
                        for (int i = r.begin(); i != r.end(); ++i) {
                          direction[i] = residual[i] + beta * direction[i];
                        }
                      });
  }

  return solution;
}

inline bool Cholesky(const std::vector<double>& matrix, int w, int h, double tolerance = 1e-5) {
  if (w != h) {
    return false;
  }

  int n = w;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (std::abs(matrix[(i * n) + j] - matrix[(j * n) + i]) > tolerance) {
        return false;
      }
    }
  }

  std::vector<double> lower_triangular(n * n, 0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += lower_triangular[(i * n) + k] * lower_triangular[(j * n) + k];
      }

      if (i == j) {
        double diag_val = matrix[(i * n) + i] - sum;
        if (diag_val <= tolerance) {
          return false;
        }
        lower_triangular[(i * n) + i] = std::sqrt(diag_val);
      } else {
        lower_triangular[(i * n) + j] = (1.0 / lower_triangular[(j * n) + j]) * (matrix[(i * n) + j] - sum);
      }
    }
  }

  return true;
}

class GradientMethod : public ppc::core::Task {
 public:
  explicit GradientMethod(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int size_;
  double tolerance_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> solution_;
  std::vector<double> result_;
  boost::mpi::communicator world_;
};

}  // namespace sidorina_p_gradient_method_all