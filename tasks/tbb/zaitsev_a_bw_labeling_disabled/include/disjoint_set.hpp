#pragma once

#include <vector>

namespace zaitsev_a_disjoint_set {

template <typename T>
class DisjointSet {
  std::vector<T> rank_, size_, parent_;

 public:
  DisjointSet(T n = 10) {
    rank_.resize(n + 1, 0);
    size_.resize(n + 1, 1);
    for (int i = 0; i <= n; i++) {
      parent_.push_back(i);
    }
  }

  T FindParent(T n) {
    if (parent_[n] == n) {
      return n;
    }

    return parent_[n] = FindParent(parent_[n]);
  }

  void UnionRank(T n1, T n2) {
    T ulp_u = FindParent(n1);
    T ulp_v = FindParent(n2);

    if (ulp_u == ulp_v) {
      return;
    }

    if (rank_[ulp_u] < rank_[ulp_v]) {
      parent_[ulp_u] = parent_[ulp_v];
    } else if (rank_[ulp_u] > rank_[ulp_v]) {
      parent_[ulp_v] = parent_[ulp_u];
    } else {
      parent_[ulp_v] = parent_[ulp_u];
      rank_[ulp_v]++;
    }
  }
};
}  // namespace zaitsev_a_disjoint_set