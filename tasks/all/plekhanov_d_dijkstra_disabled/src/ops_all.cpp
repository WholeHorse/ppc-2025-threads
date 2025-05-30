#include "all/plekhanov_d_dijkstra/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/operations.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

namespace {

bool ConvertGraphToAdjacencyList(const std::vector<int>& graph_data, size_t num_vertices,
                                 std::vector<std::vector<std::pair<int, int>>>& graph) {
  graph.assign(num_vertices, {});
  size_t current_vertex = 0;
  size_t i = 0;
  while (i < graph_data.size() && current_vertex < num_vertices) {
    if (graph_data[i] == -1) {
      current_vertex++;
      i++;
      continue;
    }
    if (i + 1 >= graph_data.size()) {
      break;
    }
    size_t dest = graph_data[i];
    int weight = graph_data[i + 1];
    if (weight < 0) {
      return false;
    }
    if (dest < num_vertices) {
      graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    }
    i += 2;
  }
  return true;
}

void ProcessLocalChunk(const std::vector<std::vector<std::pair<int, int>>>& adj_list, std::vector<int>& local_dist,
                       size_t start, size_t end, bool& updated) {
  const int inf = INT_MAX;
#pragma omp parallel for schedule(dynamic)
  for (int u = static_cast<int>(start); u < static_cast<int>(end); ++u) {
    if (local_dist[u] == inf) {
      continue;
    }

    for (const auto& [neighbor, weight] : adj_list[u]) {
      int new_dist = local_dist[u] + weight;
      if (new_dist < local_dist[neighbor]) {
#pragma omp critical
        {
          if (new_dist < local_dist[neighbor]) {
            local_dist[neighbor] = new_dist;
            updated = true;
          }
        }
      }
    }
  }
}

void UpdateLocalDistances(const std::vector<int>& global_dist, std::vector<int>& local_dist, bool& updated) {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(local_dist.size()); ++i) {
    if (global_dist[i] < local_dist[i]) {
      local_dist[i] = global_dist[i];
      updated = true;
    }
  }
}

void ProcessAllVertices(const std::vector<std::vector<std::pair<int, int>>>& adj_list, std::vector<int>& local_dist,
                        bool& updated) {
  const int inf = INT_MAX;
#pragma omp parallel for schedule(dynamic)
  for (int u = 0; u < static_cast<int>(local_dist.size()); ++u) {
    if (local_dist[u] == inf) {
      continue;
    }

    for (const auto& [neighbor, weight] : adj_list[u]) {
      int new_dist = local_dist[u] + weight;
      if (new_dist < local_dist[neighbor]) {
#pragma omp critical
        {
          if (new_dist < local_dist[neighbor]) {
            local_dist[neighbor] = new_dist;
            updated = true;
          }
        }
      }
    }
  }
}

bool CheckGlobalUpdate(const boost::mpi::communicator& world, bool local_updated) {
  int local_updated_int = local_updated ? 1 : 0;
  int global_updated = 0;
  boost::mpi::all_reduce(world, local_updated_int, global_updated, std::plus<>());
  return (global_updated > 0);
}

void UpdateFinalDistances(const boost::mpi::communicator& world, const std::vector<int>& local_dist,
                          std::vector<int>& distances) {
  std::vector<int> final_dist(local_dist.size());
  boost::mpi::all_reduce(world, local_dist.data(), static_cast<int>(local_dist.size()), final_dist.data(),
                         boost::mpi::minimum<int>());

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(distances.size()); ++i) {
    distances[i] = final_dist[i];
  }
}

}  // namespace

const int plekhanov_d_dijkstra_all::TestTaskALL::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_all::TestTaskALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
           task_data->outputs_count[0] > 0;
  }
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::RunImpl() {
  std::vector<std::vector<std::pair<int, int>>> adj_list(num_vertices_);
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, adj_list)) {
    return false;
  }

  const int inf = INT_MAX;
  std::vector<int> local_dist(num_vertices_, inf);
  local_dist[start_vertex_] = 0;

  const size_t chunk_size = (num_vertices_ + world_.size() - 1) / world_.size();
  const size_t start = world_.rank() * chunk_size;
  const size_t end = std::min(start + chunk_size, num_vertices_);

  bool updated = true;
  size_t iteration = 0;
  const size_t max_iterations = num_vertices_;

  while (updated && iteration < max_iterations) {
    updated = false;
    iteration++;

    ProcessLocalChunk(adj_list, local_dist, start, end, updated);

    std::vector<int> global_dist(num_vertices_);
    boost::mpi::all_reduce(world_, local_dist.data(), static_cast<int>(num_vertices_), global_dist.data(),
                           boost::mpi::minimum<int>());

    UpdateLocalDistances(global_dist, local_dist, updated);
    ProcessAllVertices(adj_list, local_dist, updated);
    updated = CheckGlobalUpdate(world_, updated);
  }

  ::UpdateFinalDistances(world_, local_dist, distances_);
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < distances_.size(); ++i) {
      output[i] = distances_[i];
    }
  }
  return true;
}