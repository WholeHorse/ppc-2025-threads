#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "all/vershinina_a_hoare_sort/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
struct TestConf {
  int size;
  std::function<bool(const std::vector<int>& out)> chk;
} conf = {.size = 300000, .chk = [](const auto& out) { return std::ranges::is_sorted(out); }};
}  // namespace

namespace {
std::vector<int> GetRandomVector(int len) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<int> vec(len);
  size_t vec_size = vec.size();
  for (size_t i = 0; i < vec_size; i++) {
    vec[i] = distr(gen);
  }
  return vec;
}
}  // namespace

TEST(vershinina_a_hoare_sort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  if (world.rank() == 0) {
    in = GetRandomVector(conf.size);
    out.resize(in.size());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<vershinina_a_hoare_sort_mpi::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<int>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(conf.chk(out));
  }
}

TEST(vershinina_a_hoare_sort_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  if (world.rank() == 0) {
    in = GetRandomVector(conf.size);
    out.resize(in.size());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<vershinina_a_hoare_sort_mpi::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<int>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(conf.chk(out));
  }
}
