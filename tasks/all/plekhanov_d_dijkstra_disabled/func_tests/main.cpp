#include <gtest/gtest.h>

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
#include "core/task/include/task.hpp"

namespace plekhanov_d_dijkstra_all {

namespace {
template <typename ExpectedResultType>
void ValidateTaskSuccess(TestTaskALL &test_task, std::vector<int> &distances,
                         const std::vector<ExpectedResultType> &expected_result) {
  ASSERT_TRUE(test_task.Run());
  test_task.PostProcessing();
  if (test_task.GetRank() == 0) {
    for (size_t i = 0; i < distances.size(); ++i) {
      EXPECT_EQ(distances[i], expected_result[i]);
    }
  }
}

template <typename ExpectedResultType>
void ExecuteAndValidateTask(std::shared_ptr<ppc::core::TaskData> &task_data, std::vector<int> &distances,
                            const std::vector<ExpectedResultType> &expected_result, bool expect_success) {
  TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();

  if (expect_success) {
    ValidateTaskSuccess(test_task, distances, expected_result);
  } else {
    ASSERT_FALSE(test_task.Run());
  }
}

std::vector<int> ConvertToGraphData(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list) {
  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

template <typename ExpectedResultType>
void RunTest(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list, size_t start_vertex,
             const std::vector<ExpectedResultType> &expected_result, bool expect_success = true) {
  const size_t k_num_vertices = adj_list.size();
  std::vector<int> distances(k_num_vertices, INT_MAX);
  std::vector<int> graph_data = ConvertToGraphData(adj_list);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_all->inputs_count.emplace_back(graph_data.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_all->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_all->outputs_count.emplace_back(k_num_vertices);

  ExecuteAndValidateTask(task_data_all, distances, expected_result, expect_success);
}

void RunValidationFailureTest() {
  std::vector<int> graph_data;
  size_t start_vertex = 0;
  size_t num_vertices = 0;
  std::vector<int> distances(num_vertices, INT_MAX);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_all->inputs_count.emplace_back(graph_data.size());
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_all->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_all->outputs_count.emplace_back(num_vertices);

  TestTaskALL test_task_all(task_data_all);
  if (test_task_all.GetRank() == 0) {
    ASSERT_FALSE(test_task_all.Validation());
  }
}

std::vector<std::vector<std::pair<size_t, int>>> GenerateRandomGraph(size_t num_vertices) {
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

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Simple_Path_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {
      {{1, 1}}, {{0, 1}, {2, 2}}, {{1, 2}, {3, 3}}, {{2, 3}, {4, 4}}, {{3, 4}}};
  std::vector<int> expected = {0, 1, 3, 6, 10};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Complete_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 10}, {2, 3}, {3, 20}, {4, 7}},
                                                               {{0, 10}, {2, 5}, {3, 4}, {4, 11}},
                                                               {{0, 3}, {1, 5}, {3, 2}, {4, 6}},
                                                               {{0, 20}, {1, 4}, {2, 2}, {4, 8}},
                                                               {{0, 7}, {1, 11}, {2, 6}, {3, 8}}};
  std::vector<int> expected = {3, 5, 0, 2, 6};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 2, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Disconnected_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 5}, {2, 3}}, {{0, 5}, {2, 1}}, {{0, 3}, {1, 1}},
                                                               {{4, 2}, {5, 8}}, {{3, 2}, {5, 1}}, {{3, 8}, {4, 1}}};
  std::vector<int> expected = {0, 4, 3, INT_MAX, INT_MAX, INT_MAX};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Large_Sparse_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {
      {{1, 4}, {2, 2}}, {{3, 5}, {4, 10}}, {{5, 3}, {6, 2}}, {{7, 4}}, {{8, 11}},
      {{8, 1}},         {{9, 3}},          {{9, 5}},         {{9, 7}}, {}};

  std::vector<int> expected = {0, 4, 2, 9, 14, 5, 4, 13, 6, 7};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_validation_failure) {
  plekhanov_d_dijkstra_all::RunValidationFailureTest();
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Directed_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 2}, {2, 4}}, {{2, 1}, {3, 7}}, {{3, 3}}, {}};
  std::vector<int> expected = {0, 2, 3, 6};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_SelfLoops) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {
      {{0, 0}, {1, 2}}, {{1, 0}, {2, 3}}, {{2, 0}, {3, 1}}, {{3, 0}}};
  std::vector<int> expected = {0, 2, 5, 6};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Multigraph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 10}, {1, 3}}, {{2, 5}}, {{3, 2}, {3, 8}}, {}};
  std::vector<int> expected = {0, 3, 8, 10};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Dense_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 2}, {2, 9}, {3, 4}},
                                                               {{0, 2}, {2, 1}, {3, 7}, {4, 3}},
                                                               {{0, 9}, {1, 1}, {4, 5}},
                                                               {{0, 4}, {1, 7}, {4, 6}},
                                                               {{1, 3}, {2, 5}, {3, 6}}};
  std::vector<int> expected = {0, 2, 3, 4, 5};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Cyclic_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 1}}, {{2, 2}}, {{0, 4}, {3, 1}}, {}};
  std::vector<int> expected = {0, 1, 3, 4};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_EqualOptimalPaths) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 5}, {2, 5}}, {{3, 5}}, {{3, 5}}, {}};
  std::vector<int> expected = {0, 5, 5, 10};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Negative_Edges) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 4}, {2, -2}}, {{0, 4}, {2, 3}}, {{0, -2}, {1, 3}}};
  std::vector<int> expected = {0, 0, 0};
  plekhanov_d_dijkstra_all::RunTest(adj_list, 0, expected, false);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Random_Graph_10) {
  size_t num_vertices = 10;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_all::GenerateRandomGraph(num_vertices);
  size_t start_vertex = 0;
  std::vector<int> expected = plekhanov_d_dijkstra_all::CalculateExpectedResult(adj_list, start_vertex);

  plekhanov_d_dijkstra_all::RunTest(adj_list, start_vertex, expected);
}

TEST(plekhanov_d_dijkstra_all, test_dijkstra_Random_Graph_150) {
  size_t num_vertices = 150;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_all::GenerateRandomGraph(num_vertices);
  size_t start_vertex = 0;
  std::vector<int> expected = plekhanov_d_dijkstra_all::CalculateExpectedResult(adj_list, start_vertex);

  plekhanov_d_dijkstra_all::RunTest(adj_list, start_vertex, expected);
}