#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
enum VectorClass : std::uint8_t { kRandom, kReverse };

template <VectorClass cl>
std::vector<int> VecGen(size_t size);

template <>
std::vector<int> VecGen<kReverse>(size_t size) {
  std::vector<int> v(size);
  std::iota(v.rbegin(), v.rend(), -2);
  return v;
}

void PerformPerfTest(bool test_task_run) {
  auto in = VecGen<kReverse>(25'000'000);
  decltype(in) out(in.size());

  auto data = std::make_shared<ppc::core::TaskData>();
  if (boost::mpi::communicator{}.rank() == 0) {
    data->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
    data->inputs_count.push_back(in.size());
    data->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.push_back(out.size());
  }

  auto task = std::make_shared<koshkin_m_radix_int_simple_merge::AllT>(data);

  auto attr = std::make_shared<ppc::core::PerfAttr>();
  attr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  attr->current_timer = [&] {
    const auto current_time_point = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  if (test_task_run) {
    analyzer->TaskRun(attr, results);
  } else {
    analyzer->PipelineRun(attr, results);
  }

  if (boost::mpi::communicator{}.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(results);

    EXPECT_TRUE(std::ranges::is_sorted(out));
  }
}
}  // namespace

TEST(koshkin_m_radix_int_simple_merge_all, test_pipeline_run) { PerformPerfTest(false); }
TEST(koshkin_m_radix_int_simple_merge_all, test_task_run) { PerformPerfTest(true); }
