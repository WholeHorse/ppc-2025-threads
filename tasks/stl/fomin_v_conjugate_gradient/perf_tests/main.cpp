#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

namespace {

void VerifySolution(const std::vector<double>& input, const double* solution, int size) {
  // Извлекаем матрицу A и вектор b из входных данных
  std::vector<double> a(input.begin(), input.begin() + size * size);
  std::vector<double> b_vec(input.begin() + size * size, input.end());

  // Вычисляем невязку: r = b - A*x
  std::vector<double> residual(size, 0.0);
  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < size; ++j) {
      sum += a[(i * size) + j] * solution[j];
    }
    residual[i] = b_vec[i] - sum;
  }

  double residual_norm = 0.0;
  for (double r : residual) {
    residual_norm += r * r;
  }
  residual_norm = sqrt(residual_norm);

  EXPECT_LT(residual_norm, 1e-5);
}

}  // namespace

TEST(fomin_v_conjugate_gradient_stl, test_pipeline_run) {
  constexpr int kCount = 990;

  // Создаем трехдиагональную матрицу с диагональным преобладанием
  std::vector<double> input((kCount * kCount) + kCount, 0.0);
  for (int i = 0; i < kCount; ++i) {
    input[(i * kCount) + i] = 4.0;
    if (i > 0) {
      input[(i * kCount) + (i - 1)] = -1.0;
    }
    if (i < kCount - 1) {
      input[(i * kCount) + (i + 1)] = -1.0;
    }
    input[(kCount * kCount) + i] = 1.0;
  }

  // Используем умный указатель для автоматического управления памятью
  std::unique_ptr<double[]> output_buffer(new double[kCount]);

  // Создаем task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_buffer.get()));
  task_data_seq->outputs_count.emplace_back(kCount);

  auto test_task_sequential = std::make_shared<fomin_v_conjugate_gradient::FominVConjugateGradientStl>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты perf теста
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  VerifySolution(input, output_buffer.get(), kCount);
}

TEST(fomin_v_conjugate_gradient_stl, test_task_run) {
  constexpr int kCount = 990;

  // Создаем трехдиагональную матрицу с диагональным преобладанием
  std::vector<double> input((kCount * kCount) + kCount, 0.0);
  for (int i = 0; i < kCount; ++i) {
    input[(i * kCount) + i] = 4.0;
    if (i > 0) {
      input[(i * kCount) + (i - 1)] = -1.0;
    }
    if (i < kCount - 1) {
      input[(i * kCount) + (i + 1)] = -1.0;
    }
    input[(kCount * kCount) + i] = 1.0;
  }

  // Используем умный указатель для автоматического управления памятью
  std::unique_ptr<double[]> output_buffer(new double[kCount]);

  // Создаем task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_buffer.get()));
  task_data_seq->outputs_count.emplace_back(kCount);

  // Создаем задачу
  auto test_task_sequential = std::make_shared<fomin_v_conjugate_gradient::FominVConjugateGradientStl>(task_data_seq);

  // Создаем атрибуты для perf теста
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты perf теста
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Создаем perf анализатор
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  VerifySolution(input, output_buffer.get(), kCount);
}