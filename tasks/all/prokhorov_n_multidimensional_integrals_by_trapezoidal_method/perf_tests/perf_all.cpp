#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "all/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all {

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all, test_pipeline_run) {
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<double> lower = {0.0, 0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {200, 200, 200};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  auto test_task = std::make_shared<TestTaskALL>(task_data);
  test_task->SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; });

  if (rank == 0) {
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  } else {
    for (int i = 0; i < 10; ++i) {
      test_task->PreProcessing();
      test_task->Run();
      test_task->PostProcessing();
    }
  }
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all, test_task_run) {
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<double> lower = {0.0, 0.0, 0.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {200, 200, 200};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  auto test_task = std::make_shared<TestTaskALL>(task_data);
  test_task->SetFunction([](const std::vector<double>& point) { return point[0] + point[1] + point[2]; });

  if (rank == 0) {
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
    perf_analyzer->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    double expected = 1.5;
    ASSERT_NEAR(result, expected, 0.01);
  } else {
    for (int i = 0; i < 10; ++i) {
      test_task->PreProcessing();
      test_task->Run();
      test_task->PostProcessing();
    }
  }
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_all