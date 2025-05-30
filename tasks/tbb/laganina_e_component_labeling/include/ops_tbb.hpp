#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <atomic>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;

  bool PreProcessingImpl() override;

  bool PostProcessingImpl() override;
  bool RunImpl() override;

 private:
  int rows_;
  int cols_;
  std::vector<int> data_;

  class UnionFind {
    std::unique_ptr<std::atomic<int>[]> parent;

   public:
    UnionFind(int size, const std::vector<int>& data) {
      parent = std::make_unique<std::atomic<int>[]>(size);
      for (int i = 0; i < size; ++i) {
        parent[i].store(data[i] ? i : -1);
      }
    }

    int Find(int x) {
      while (true) {
        int p = parent[x].load();
        if (p == x || p == -1) return p;
        int gp = parent[p].load();
        if (gp != p) {
          parent[x].compare_exchange_weak(p, gp);
        }
        x = parent[x].load();
      }
    }

    void Unite(int x, int y) {
      while (true) {
        int rx = Find(x);
        int ry = Find(y);
        if (rx == ry || rx == -1 || ry == -1) return;
        if (rx > ry) std::swap(rx, ry);
        int expected = ry;
        if (parent[ry].compare_exchange_strong(expected, rx)) {
          return;
        }
      }
    }
  };

  void ProcessComponents(UnionFind& uf);
  void AssignFinalLabels(int size, UnionFind& uf);
  void ProcessRange(const tbb::blocked_range2d<int>& range, UnionFind& uf);
  void ProcessRow(int row, const tbb::blocked_range<int>& col_range, UnionFind& uf);
  void CheckAllNeighbors(int row, int col, int idx, UnionFind& uf);
  void LabelComponents();
};

inline void NormalizeLabels(std::vector<int>& vec) {
  std::vector<int> unique_labels;
  std::unordered_set<int> seen;

  for (int val : vec) {
    if (val != 0 && seen.find(val) == seen.end()) {
      unique_labels.push_back(val);
      seen.insert(val);
    }
  }

  std::unordered_map<int, int> label_map;
  int current_label = 1;
  for (int val : unique_labels) {
    label_map[val] = current_label++;
  }
  for (int& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}

}  // namespace laganina_e_component_labeling_tbb
