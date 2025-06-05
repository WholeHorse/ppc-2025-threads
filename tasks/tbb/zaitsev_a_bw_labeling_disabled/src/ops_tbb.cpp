#include "tbb/zaitsev_a_bw_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

using zaitsev_a_labeling_tbb::Labeler;

namespace {

void FirstScan(Labels& labels, Equivalency& eq, Ordinal& ordinal, Length width) {
  for (Length i = 0; i < labels.size(); i++) {
    if (labels[i] == 0) {
      continue;
    }

    Labels neighbours;
    neighbours.reserve(4);

    for (int shift = 0; shift < 4; shift++) {
      long x = ((long)i % width) + (shift % 3 - 1);
      long y = ((long)i / width) + (shift / 3 - 1);
      long neighbour_index = x + (y * width);
      Ordinal value = 0;
      if (x >= 0 && x < static_cast<long>(width) && y >= 0) {
        value = labels[neighbour_index];
      }
      if (value > 0) {
        neighbours.push_back(value);
      }
    }

    if (neighbours.empty()) {
      labels[i] = ++ordinal;
      eq[ordinal].insert(ordinal);
    } else {
      labels[i] = *std::ranges::min_element(neighbours);
      for (auto& first : neighbours) {
        for (auto& second : neighbours) {
          eq[first].insert(second);
        }
      }
    }
  }
}

}  // namespace

bool Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  image_.resize(size_, 0);
  std::copy(task_data->inputs[0], task_data->inputs[0] + size_, image_.begin());
  return true;
}

bool Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && (!task_data->inputs.empty()) &&
         (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1]);
}

void Labeler::LabelingRasterScan(Equivalencies& eqs, Ordinals& ordinals) {
  LabelsList lbls(ppc::util::GetPPCNumThreads(), Labels(chunk_, 0));
  for (Length i = 0; i < size_; i++) {
    lbls[i / chunk_][i % chunk_] = image_[i];
  }

  oneapi::tbb::task_group tg;
  for (int i = 0; i < ppc::util::GetPPCNumThreads(); i++) {
    tg.run([i, &lbls, &eqs, &ordinals, this] {
      FirstScan(std::ref(lbls[i]), std::ref(eqs[i]), std::ref(ordinals[i]), width_);
    });
  }

  tg.wait();

  for (Length i = 0; i < size_; i++) {
    labels_[i] = lbls[i / chunk_][i % chunk_];
  }
}

void Labeler::UniteChunks(DisjointSet& dsj, Ordinals& ordinals) {
  long start_pos = 0;
  long end_pos = width_;
  for (long i = 1; i < ppc::util::GetPPCNumThreads(); i++) {
    start_pos += chunk_;
    end_pos += chunk_;
    for (long pos = start_pos; pos < end_pos; pos++) {
      if (pos >= static_cast<long>(size_) || pos < 0 || labels_[pos] == 0) {
        continue;
      }
      Length lower = labels_[pos];
      for (long shift = -1; shift < 2; shift++) {
        long neighbour_pos = std::clamp(pos + shift, start_pos, end_pos - 1) - width_;
        if (neighbour_pos < 0 || neighbour_pos >= static_cast<long>(size_) || labels_[neighbour_pos] == 0) {
          continue;
        }
        Length upper = labels_[neighbour_pos];
        dsj.UnionRank(upper, lower);
      }
    }
  }
}

void Labeler::CalculateReplacements(Replacements& replacements, Equivalencies& eqs, Ordinals& ordinals) {
  Ordinal labels_amount = std::reduce(ordinals.begin(), ordinals.end(), 0);
  Length shift = 0;

  DisjointSet disjoint_labels(labels_amount + 1);

  for (int i = 0; i < ppc::util::GetPPCNumThreads(); i++) {
    for (auto& eq : eqs[i]) {
      for (const auto& equal : eq.second) {
        disjoint_labels.UnionRank(eq.first + shift, equal + shift);
      }
    }
    shift += ordinals[i];
  }

  UniteChunks(disjoint_labels, ordinals);

  replacements.resize(labels_amount + 1);
  std::set<std::uint16_t> unique_labels;

  for (Ordinal tmp_label = 1; tmp_label < labels_amount + 1; tmp_label++) {
    replacements[tmp_label] = disjoint_labels.FindParent(tmp_label);
    unique_labels.insert(replacements[tmp_label]);
  }

  Ordinal true_label = 0;
  std::map<std::uint16_t, std::uint16_t> reps;
  for (const auto& it : unique_labels) {
    reps[it] = ++true_label;
  }

  for (Length i = 0; i < replacements.size(); i++) {
    replacements[i] = reps[replacements[i]];
  }
}

void Labeler::PerformReplacements(Replacements& replacements) {
  for (Length i = 0; i < size_; i++) {
    labels_[i] = replacements[labels_[i]];
  }
}

void Labeler::GlobalizeLabels(Ordinals& ordinals) {
  Length shift = 0;
  for (Length i = chunk_; i < size_; i++) {
    if (i % chunk_ == 0) {
      shift += ordinals[(i / chunk_) - 1];
    }
    if (labels_[i] != 0) {
      labels_[i] += shift;
    }
  }
}

bool Labeler::RunImpl() {
  labels_.clear();
  labels_.resize(size_, 0);
  Equivalencies eqs(ppc::util::GetPPCNumThreads());
  Ordinals ordinals(ppc::util::GetPPCNumThreads(), 0);
  Replacements replacements;

  chunk_ = static_cast<long>(std::ceil(static_cast<double>(height_) / ppc::util::GetPPCNumThreads())) * width_;

  LabelingRasterScan(eqs, ordinals);
  GlobalizeLabels(ordinals);

  CalculateReplacements(replacements, eqs, ordinals);
  PerformReplacements(replacements);
  return true;
}

bool Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<std::uint16_t*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}