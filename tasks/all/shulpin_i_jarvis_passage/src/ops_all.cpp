#include "all/shulpin_i_jarvis_passage/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
int Orientation(const shulpin_i_jarvis_all::Point& p, const shulpin_i_jarvis_all::Point& q,
                const shulpin_i_jarvis_all::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

shulpin_i_jarvis_all::Point FindLocalCandidate(const shulpin_i_jarvis_all::Point& current,
                                               const std::vector<shulpin_i_jarvis_all::Point>& points,
                                               int num_threads) {
  std::vector<shulpin_i_jarvis_all::Point> local_cand(num_threads);

  auto worker = [&](int tid) {
    size_t chunk = points.size() / num_threads;
    size_t start = tid * chunk;
    size_t end = (tid == num_threads - 1 ? points.size() : start + chunk);

    shulpin_i_jarvis_all::Point candidate;
    if (start < end) {
      candidate = points[start];
      for (size_t i = start; i < end; ++i) {
        const auto& p = points[i];
        if (p == current) {
          continue;
        }
        double cross =
            ((p.y - current.y) * (candidate.x - current.x)) - ((p.x - current.x) * (candidate.y - current.y));
        double d_p = std::pow(p.x - current.x, 2) + std::pow(p.y - current.y, 2);
        double d_c = std::pow(candidate.x - current.x, 2) + std::pow(candidate.y - current.y, 2);
        if (cross > 0 || (cross == 0 && d_p > d_c)) {
          candidate = p;
        }
      }
    } else {
      candidate = current;
    }

    local_cand[tid] = candidate;
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    th.join();
  }

  shulpin_i_jarvis_all::Point best = local_cand[0];
  for (int t = 1; t < num_threads; ++t) {
    const auto& cand = local_cand[t];
    double cross = ((cand.y - current.y) * (best.x - current.x)) - ((cand.x - current.x) * (best.y - current.y));
    double d_c = std::pow(cand.x - current.x, 2) + std::pow(cand.y - current.y, 2);
    double d_b = std::pow(best.x - current.x, 2) + std::pow(best.y - current.y, 2);
    if (cross > 0 || (cross == 0 && d_c > d_b)) {
      best = cand;
    }
  }
  return best;
}

}  // namespace

void shulpin_i_jarvis_all::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_all::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_all::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    output_jar.emplace_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

    for (size_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_all::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_all::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_all::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_all::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_all::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_all::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity, readability-make-member-function-const)
void shulpin_i_jarvis_all::JarvisALLParallel::MakeJarvisPassageALL(
    std::vector<shulpin_i_jarvis_all::Point>& input_jar, std::vector<shulpin_i_jarvis_all::Point>& output_jar) {
  output_jar.clear();

  MPI_Datatype mpi_point = MPI_DATATYPE_NULL;
  MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point);
  MPI_Type_commit(&mpi_point);

  size_t n = input_jar.size();
  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    input_jar.resize(n);
  }

  MPI_Bcast(input_jar.data(), static_cast<int>(n), mpi_point, 0, MPI_COMM_WORLD);

  size_t most_left = 0;
  if (rank_ == 0) {
    for (size_t i = 1; i < n; ++i) {
      if (input_jar[i].x < input_jar[most_left].x ||
          (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
        most_left = i;
      }
    }
  }
  MPI_Bcast(&most_left, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  Point min_point = input_jar[most_left];

  std::vector<int> counts(world_size_);
  std::vector<int> displs(world_size_);
  int base = static_cast<int>(n / world_size_);
  int rem = static_cast<int>(n % world_size_);
  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  std::vector<Point> local_points(counts[rank_]);
  MPI_Scatterv(input_jar.data(), counts.data(), displs.data(), mpi_point, local_points.data(), counts[rank_], mpi_point,
               0, MPI_COMM_WORLD);

  std::unordered_set<Point, PointHash, PointEqual> unique_points;
  if (rank_ == 0) {
    output_jar.push_back(min_point);
    unique_points.insert(min_point);
  }

  Point prev_point = min_point;
  Point next_point;
  int num_threads = ppc::util::GetPPCNumThreads();

  bool done = false;

  do {
    MPI_Bcast(&prev_point, 1, mpi_point, 0, MPI_COMM_WORLD);

    Point local_candidate = FindLocalCandidate(prev_point, local_points, num_threads);

    std::vector<Point> all_cand;
    if (rank_ == 0) {
      all_cand.resize(world_size_);
    }
    MPI_Gather(&local_candidate, 1, mpi_point, all_cand.data(), 1, mpi_point, 0, MPI_COMM_WORLD);

    if (rank_ == 0) {
      next_point = all_cand[0];
      for (int i = 1; i < world_size_; ++i) {
        const auto& cand = all_cand[i];
        double cross = ((cand.y - prev_point.y) * (next_point.x - prev_point.x)) -
                       ((cand.x - prev_point.x) * (next_point.y - prev_point.y));
        double d_c = std::pow(cand.x - prev_point.x, 2) + std::pow(cand.y - prev_point.y, 2);
        double d_n = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);
        if (cross > 0 || (cross == 0 && d_c > d_n)) {
          next_point = cand;
        }
      }

      done = (next_point == min_point);

      if (!done && unique_points.find(next_point) == unique_points.end()) {
        output_jar.push_back(next_point);
        unique_points.insert(next_point);
      }
    }

    MPI_Bcast(&next_point, 1, mpi_point, 0, MPI_COMM_WORLD);
    MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    prev_point = next_point;

  } while (!done);

  MPI_Type_free(&mpi_point);
}

bool shulpin_i_jarvis_all::JarvisALLParallel::PreProcessingImpl() {
  if (rank_ == 0) {
    std::vector<shulpin_i_jarvis_all::Point> tmp_input;

    auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_all::Point*>(task_data->inputs[0]);
    size_t tmp_size = task_data->inputs_count[0];
    tmp_input.assign(tmp_data, tmp_data + tmp_size);

    input_stl_ = tmp_input;

    size_t output_size = task_data->outputs_count[0];
    output_stl_.resize(output_size);
  }
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::ValidationImpl() {
  if (rank_ == 0) {
    return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
  }
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::RunImpl() {
  MakeJarvisPassageALL(input_stl_, output_stl_);
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::PostProcessingImpl() {
  if (rank_ == 0) {
    auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
    std::ranges::copy(output_stl_.begin(), output_stl_.end(), result);
  }
  return true;
}