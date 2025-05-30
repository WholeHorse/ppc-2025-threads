#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "all/plekhanov_d_dijkstra/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace plekhanov_d_dijkstra_all {

namespace {

std::vector<std::vector<std::pair<size_t, int>>> static CreateRandomAdjacencyList(size_t num_vertices);
std::vector<int> static CalculateExpectedResult(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list,
                                                size_t start_vertex);

std::vector<std::vector<std::pair<size_t, int>>> CreateRandomAdjacencyList(size_t num_vertices) {
  srand(static_cast<unsigned int>(time(nullptr)));
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < num_vertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }
  return adj_list;
}

std::vector<int> CalculateExpectedResult(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list,
                                         size_t start_vertex) {
  size_t n = adj_list.size();
  const int inf = INT_MAX;
  std::vector<int> distances(n, inf);
  distances[start_vertex] = 0;

  using Pii = std::pair<int, size_t>;
  std::priority_queue<Pii, std::vector<Pii>, std::greater<>> pq;
  pq.emplace(0, start_vertex);

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();

    if (d != distances[u]) {
      continue;
    }

    for (const auto &edge : adj_list[u]) {
      size_t v = edge.first;
      int weight = edge.second;
      if (distances[u] != inf && (distances[u] + weight < distances[v])) {
        distances[v] = distances[u] + weight;
        pq.emplace(distances[v], v);
      }
    }
  }
  return distances;
}

}  // namespace

}  // namespace plekhanov_d_dijkstra_all

TEST(plekhanov_d_dijkstra_all, test_pipeline_run) {
  constexpr size_t kNumVertices = 3000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_all::CreateRandomAdjacencyList(kNumVertices);

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_all->inputs_count.emplace_back(graph_data.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_all->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_all->outputs_count.emplace_back(kNumVertices);

  auto test_task_all = std::make_shared<plekhanov_d_dijkstra_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = plekhanov_d_dijkstra_all::CalculateExpectedResult(adj_list, start_vertex);
  if (test_task_all->GetRank() == 0) {
    EXPECT_EQ(distances, expected);
  }
}

TEST(plekhanov_d_dijkstra_all, test_task_run) {
  constexpr size_t kNumVertices = 3000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_all::CreateRandomAdjacencyList(kNumVertices);

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_all->inputs_count.emplace_back(graph_data.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_all->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_all->outputs_count.emplace_back(kNumVertices);

  auto test_task_all = std::make_shared<plekhanov_d_dijkstra_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = plekhanov_d_dijkstra_all::CalculateExpectedResult(adj_list, start_vertex);
  if (test_task_all->GetRank() == 0) {
    EXPECT_EQ(distances, expected);
  }
}