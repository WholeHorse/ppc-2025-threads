#pragma once

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace pikarychev_i_hoare_sort_simple_merge {

template <class T>
class HoareThreadBB : public ppc::core::Task {
  struct Block {
    T* e;
    std::size_t sz;
    bool merged{false};
  };

 public:
  explicit HoareThreadBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override {
    const auto input_size = task_data->inputs_count[0];
    const auto output_size = task_data->outputs_count[0];
    return input_size == output_size;
  }

  bool PreProcessingImpl() override {
    input_.assign(reinterpret_cast<T*>(task_data->inputs[0]),
                  reinterpret_cast<T*>(task_data->inputs[0]) + task_data->inputs_count[0]);
    res_.resize(input_.size());
    reverse_ = *reinterpret_cast<bool*>(task_data->inputs[1]);
    return true;
  }

  void SortInParallel(std::size_t parallelism, std::vector<Block>& blocks) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, parallelism, 1), [&](const auto& r) {
      for (std::size_t i = r.begin(); i < r.end(); i++) {
        DoSort(blocks[i].e, 0, blocks[i].sz - 1, reverse_ ? ReverseComp : StandardComp);
      }
    });
  }

  void MergeGroup(std::size_t b, std::size_t e, std::size_t wide, std::vector<Block>& blocks) {
    for (std::size_t k = b; k < e; ++k) {
      const auto idx = 2 * wide * k;
      DoInplaceMerge(blocks.data() + idx, blocks.data() + idx + wide, reverse_ ? ReverseComp : StandardComp);
    }
  }

  void MergeInParallel(std::size_t parallelism, std::vector<Block>& blocks) {
    auto partind = parallelism;
    for (auto wide = decltype(parallelism){1}; partind > 1; wide *= 2, partind /= 2) {
      if (blocks[wide].sz < 40) {
        MergeGroup(0, partind / 2, wide, blocks);
      } else {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, (partind / 2), (partind / 2) / parallelism),
            [&](const auto& r) { MergeGroup(r.begin(), r.end(), wide, blocks); });
      }
      if ((partind / 2) == 1) {
        DoInplaceMerge(&blocks.front(), &blocks.back(), reverse_ ? ReverseComp : StandardComp);
      } else if ((partind / 2) % 2 != 0) {
        DoInplaceMerge(blocks.data() + (2 * wide * ((partind / 2) - 2)),
                       blocks.data() + (2 * wide * ((partind / 2) - 1)), reverse_ ? ReverseComp : StandardComp);
      }
    }
  }

  bool RunImpl() override {
    const auto size = input_.size();
    std::ranges::copy_n(input_.begin(), size, res_.begin());

    const auto parallelism = std::min<std::size_t>(size, ppc::util::GetPPCNumThreads());
    if (size < 1) {
      return true;
    }

    const auto fair = size / parallelism;
    const auto extra = size % parallelism;

    std::vector<Block> blocks(parallelism);
    T* pointer = res_.data();
    for (std::size_t i = 0; i < parallelism; i++) {
      blocks[i] = {.e = pointer, .sz = fair + ((i < extra) ? 1 : 0)};
      pointer += blocks[i].sz;
    }

    oneapi::tbb::task_arena arena(parallelism);
    arena.execute([&] {
      SortInParallel(parallelism, blocks);
      MergeInParallel(parallelism, blocks);
    });

    return true;
  }

  static bool StandardComp(const T& a, const T& b) { return a > b; }
  static bool ReverseComp(const T& a, const T& b) { return a < b; }

  bool PostProcessingImpl() override {
    std::ranges::copy_n(res_.begin(), res_.size(), reinterpret_cast<T*>(task_data->outputs[0]));
    return true;
  }

 private:
  static void DoInplaceMerge(Block* b1, Block* b2, bool (*comp)(const T&, const T&)) {
    std::inplace_merge(b1->e, b2->e, b2->e + b2->sz, comp);
    b2->merged = true;
    b1->sz += b2->sz;
  }

  static int Partition(T* block, int low, int high, bool (*comp)(const T&, const T&)) {
    int e = low - 1;
    for (int j = low; j <= high - 1; j++) {
      if (!comp(block[j], block[high])) {
        continue;
      }
      e++;
      std::swap(block[e], block[j]);
    }
    std::swap(block[e + 1], block[high]);
    return ++e;
  };
  static void DoSort(T* arr, int low, int high, bool (*comp)(const T&, const T&)) {
    if (low >= high) {
      return;
    }
    const auto border = Partition(arr, low, high, comp);
    DoSort(arr, low, border - 1, comp);
    DoSort(arr, border + 1, high, comp);
  }

  bool reverse_;
  std::vector<T> input_;
  std::vector<T> res_;
};

}  // namespace pikarychev_i_hoare_sort_simple_merge