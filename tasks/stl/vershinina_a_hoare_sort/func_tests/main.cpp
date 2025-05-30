#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/vershinina_a_hoare_sort/include/ops_stl.hpp"

namespace {
std::vector<double> GetRandomVector(int len) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(-100, 100);
  std::vector<double> vec(len);
  size_t vec_size = vec.size();
  for (size_t i = 0; i < vec_size; i++) {
    vec[i] = distr(gen);
  }
  return vec;
}
}  // namespace

TEST(vershinina_a_hoare_sort_stl, test_empty) {
  std::vector<double> in;
  std::vector<double> out;
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_not_random_reverse_order) {
  std::vector<double> in{8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_not_random_len_10) {
  std::vector<double> in{56, 39, 11, 98, 5, 73, 40, 51, 83, 22};
  std::vector<double> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_not_random_len_100) {
  std::vector<double> in{13,  -68, 8,   33,  87,  31,  20,  86,   -78, 20,  -71, -89, 4,   -44, 94,  21,  80,
                         31,  -54, 43,  -48, -60, -45, 24,  70,   -70, -88, 82,  -41, 73,  -77, -35, -54, 81,
                         -2,  0,   25,  -51, 57,  -9,  -34, -57,  87,  22,  39,  -42, -90, 27,  -57, 65,  -48,
                         68,  100, -95, -1,  46,  -16, 32,  -100, -48, -49, 34,  -91, -68, -72, -95, -19, 51,
                         -23, -48, 21,  -84, -89, 18,  33,  -49,  14,  80,  21,  0,   -91, -62, 47,  25,  20,
                         -92, -98, -17, 2,   -66, -71, 64,  71,   -41, -53, -99, -98, -93, -48, 64};
  std::vector<double> out(100);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_random_even_len_35) {
  std::vector<double> in;
  std::vector<double> out(35);
  in = GetRandomVector(35);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_random_odd_len_200) {
  std::vector<double> in;
  std::vector<double> out(200);
  in = GetRandomVector(200);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(vershinina_a_hoare_sort_stl, test_random_even_len_333) {
  std::vector<double> in;
  std::vector<double> out(333);
  in = GetRandomVector(333);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort_stl::TestTaskSTL test_task_omp(task_data_stl);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}