#include "tbb/muradov_m_rect_int/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_reduce.h"

bool muradov_m_rect_int_tbb::RectIntTaskTBBPar::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] > 0 && task_data->outputs_count[0] == 1;
}

bool muradov_m_rect_int_tbb::RectIntTaskTBBPar::PreProcessingImpl() {
  auto* p_grains = reinterpret_cast<int*>(task_data->inputs[0]);
  grains_ = *p_grains;

  auto* p_bounds = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[1]);
  bounds_.assign(p_bounds, p_bounds + task_data->inputs_count[1]);

  return true;
}

bool muradov_m_rect_int_tbb::RectIntTaskTBBPar::RunImpl() {
  res_ = 0;

  const auto dims = static_cast<std::size_t>(bounds_.size());

  double hh = 1.0;
  for (std::size_t i = 0; i < dims; i++) {
    hh *= (bounds_[i].second - bounds_[i].first) / grains_;
  }
  int pts = static_cast<int>(std::pow(grains_, dims));

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  res_ = arena.execute([&] {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range(0, pts, pts / arena.max_concurrency()), 0.0,
        [&](const tbb::blocked_range<int>& rng, double s) {
          FunArgs args(dims);
          for (int i = rng.begin(); i < rng.end(); i++) {
            auto j = i;
            for (size_t k = 0; k < dims; k++) {
              args[k] = bounds_[k].first;
              args[k] += (j % grains_) * (bounds_[k].second - bounds_[k].first) / grains_;
              j /= grains_;
            }
            s += fun_(args);
          }
          return s;
        },
        std::plus<>());
  });

  res_ *= hh;

  return true;
}

bool muradov_m_rect_int_tbb::RectIntTaskTBBPar::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = res_;
  return true;
}
