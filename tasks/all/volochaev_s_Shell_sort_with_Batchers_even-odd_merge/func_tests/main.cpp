#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "all/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_all.hpp"  // NOLINT(misc-include-cleaner)
#include "core/task/include/task.hpp"

namespace {
void GetRandomVector(std::vector<long long int> &v, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dis(gen);
  }
}
}  // namespace

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_error_in_val) {
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  constexpr size_t kSizeOfVector = 0;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  // Create data
  if (world.rank() == 0) {
    in.resize(kSizeOfVector, 0);
    out.resize(kSizeOfVector, 0);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_sequential.Validation(), false);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_error_in_generate) {
  constexpr size_t kSizeOfVector = 100;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;

  // Create data
  if (world.rank() == 0) {
    in.resize(kSizeOfVector, 0);
    ASSERT_ANY_THROW(GetRandomVector(in, 1000, -1000));
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_small_vector) {
  constexpr size_t kSizeOfVector = 100;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_small_vector2) {
  constexpr size_t kSizeOfVector = 200;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  // Create data
  if (world.rank() == 0) {
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_small_vector3) {
  constexpr size_t kSizeOfVector = 300;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_small_vector4) {
  constexpr size_t kSizeOfVector = 400;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_medium_vector) {
  constexpr size_t kSizeOfVector = 500;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_medium_vector2) {
  constexpr size_t kSizeOfVector = 600;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_medium_vector3) {
  constexpr size_t kSizeOfVector = 700;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_medium_vector4) {
  constexpr size_t kSizeOfVector = 800;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_medium_vector5) {
  constexpr size_t kSizeOfVector = 900;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_big_vector) {
  constexpr size_t kSizeOfVector = 1000;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_big_vector2) {
  constexpr size_t kSizeOfVector = 2000;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_big_vector3) {
  constexpr size_t kSizeOfVector = 3000;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_big_vector4) {
  constexpr size_t kSizeOfVector = 4000;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_extra_big_vector) {
  constexpr size_t kSizeOfVector = 10000;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_prime_size_vector) {
  constexpr size_t kSizeOfVector = 7;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_prime_size_vector1) {
  constexpr size_t kSizeOfVector = 13;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_prime_size_vector2) {
  constexpr size_t kSizeOfVector = 17;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_prime_size_vector3) {
  constexpr size_t kSizeOfVector = 23;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_prime_size_vector4) {
  constexpr size_t kSizeOfVector = 29;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements) {
  constexpr size_t kSizeOfVector = 101;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements1) {
  constexpr size_t kSizeOfVector = 99;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements2) {
  constexpr size_t kSizeOfVector = 201;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements3) {
  constexpr size_t kSizeOfVector = 199;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements4) {
  constexpr size_t kSizeOfVector = 301;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements5) {
  constexpr size_t kSizeOfVector = 299;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements6) {
  constexpr size_t kSizeOfVector = 401;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_odd_number_of_elements7) {
  constexpr size_t kSizeOfVector = 399;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_reverse) {
  constexpr size_t kSizeOfVector = 399;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    std::ranges::sort(in);
    std::ranges::reverse(in);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Fermats_1) {
  constexpr size_t kSizeOfVector = 3;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Fermats_2) {
  constexpr size_t kSizeOfVector = 5;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Fermats_3) {
  constexpr size_t kSizeOfVector = 17;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Fermats_4) {
  constexpr size_t kSizeOfVector = 257;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Fermats_5) {
  constexpr size_t kSizeOfVector = 65537;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_1) {
  constexpr size_t kSizeOfVector = 561;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_2) {
  constexpr size_t kSizeOfVector = 1105;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_3) {
  constexpr size_t kSizeOfVector = 1729;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_4) {
  constexpr size_t kSizeOfVector = 1905;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_5) {
  constexpr size_t kSizeOfVector = 2047;

  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  in.resize(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  out.resize(kSizeOfVector, 0);
  answer = in;
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_6) {
  constexpr size_t kSizeOfVector = 2465;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_7) {
  constexpr size_t kSizeOfVector = 3277;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_8) {
  constexpr size_t kSizeOfVector = 4033;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_9) {
  constexpr size_t kSizeOfVector = 4681;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_2_10) {
  constexpr size_t kSizeOfVector = 6601;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_1) {
  constexpr size_t kSizeOfVector = 121;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_2) {
  constexpr size_t kSizeOfVector = 703;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_3) {
  constexpr size_t kSizeOfVector = 1729;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_4) {
  constexpr size_t kSizeOfVector = 2821;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_5) {
  constexpr size_t kSizeOfVector = 3281;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_6) {
  constexpr size_t kSizeOfVector = 7381;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_7) {
  constexpr size_t kSizeOfVector = 8401;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_8) {
  constexpr size_t kSizeOfVector = 8911;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_9) {
  constexpr size_t kSizeOfVector = 10585;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Euler_base_3_10) {
  constexpr size_t kSizeOfVector = 12403;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Mersenne_1) {
  constexpr size_t kSizeOfVector = 16383;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Mersenne_2) {
  constexpr size_t kSizeOfVector = 32767;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Mersenne_3) {
  constexpr size_t kSizeOfVector = 65535;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Mersenne_4) {
  constexpr size_t kSizeOfVector = 131071;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Mersenne_5) {
  constexpr size_t kSizeOfVector = 524287;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_1) {
  constexpr size_t kSizeOfVector = 1;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_2) {
  constexpr size_t kSizeOfVector = 2;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_3) {
  constexpr size_t kSizeOfVector = 3;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_4) {
  constexpr size_t kSizeOfVector = 4;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_5) {
  constexpr size_t kSizeOfVector = 7;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_6) {
  constexpr size_t kSizeOfVector = 11;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_7) {
  constexpr size_t kSizeOfVector = 18;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_8) {
  constexpr size_t kSizeOfVector = 29;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_9) {
  constexpr size_t kSizeOfVector = 47;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_10) {
  constexpr size_t kSizeOfVector = 76;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_11) {
  constexpr size_t kSizeOfVector = 123;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_12) {
  constexpr size_t kSizeOfVector = 199;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Lucas_13) {
  constexpr size_t kSizeOfVector = 322;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_1) {
  constexpr size_t kSizeOfVector = 1729;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_2) {
  constexpr size_t kSizeOfVector = 4104;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_3) {
  constexpr size_t kSizeOfVector = 13832;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_4) {
  constexpr size_t kSizeOfVector = 20683;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_5) {
  constexpr size_t kSizeOfVector = 32832;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_6) {
  constexpr size_t kSizeOfVector = 39312;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_7) {
  constexpr size_t kSizeOfVector = 40033;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_8) {
  constexpr size_t kSizeOfVector = 46683;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_9) {
  constexpr size_t kSizeOfVector = 64232;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_10) {
  constexpr size_t kSizeOfVector = 65728;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_11) {
  constexpr size_t kSizeOfVector = 110656;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_12) {
  constexpr size_t kSizeOfVector = 110808;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_13) {
  constexpr size_t kSizeOfVector = 134379;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_14) {
  constexpr size_t kSizeOfVector = 149389;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_15) {
  constexpr size_t kSizeOfVector = 165464;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_16) {
  constexpr size_t kSizeOfVector = 171288;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_17) {
  constexpr size_t kSizeOfVector = 195841;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_18) {
  constexpr size_t kSizeOfVector = 216027;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_19) {
  constexpr size_t kSizeOfVector = 216125;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_20) {
  constexpr size_t kSizeOfVector = 262656;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_21) {
  constexpr size_t kSizeOfVector = 314496;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_22) {
  constexpr size_t kSizeOfVector = 320264;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Ramanujan_23) {
  constexpr size_t kSizeOfVector = 327763;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Carmichael_1) {
  constexpr size_t kSizeOfVector = 561;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Carmichael_2) {
  constexpr size_t kSizeOfVector = 41041;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_all, test_with_len_Carmichael_3) {
  constexpr size_t kSizeOfVector = 825265;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);
    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_1) {
  constexpr size_t kSizeOfVector = 1;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_2) {
  constexpr size_t kSizeOfVector = 2;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_3) {
  constexpr size_t kSizeOfVector = 6;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_4) {
  constexpr size_t kSizeOfVector = 24;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_5) {
  constexpr size_t kSizeOfVector = 120;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_6) {
  constexpr size_t kSizeOfVector = 720;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_7) {
  constexpr size_t kSizeOfVector = 5040;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_8) {
  constexpr size_t kSizeOfVector = 40320;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_1) {
  constexpr size_t kSizeOfVector = 40320;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_2) {
  constexpr size_t kSizeOfVector = 109584;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_3) {
  constexpr size_t kSizeOfVector = 118124;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_4) {
  constexpr size_t kSizeOfVector = 67284;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_5) {
  constexpr size_t kSizeOfVector = 22449;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_6) {
  constexpr size_t kSizeOfVector = 4536;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_7) {
  constexpr size_t kSizeOfVector = 546;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_9) {
  constexpr size_t kSizeOfVector = 1;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Stirling_size_n_k_9_8) {
  constexpr size_t kSizeOfVector = 36;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_1) {
  constexpr size_t kSizeOfVector = 1;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_2) {
  constexpr size_t kSizeOfVector = 2;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_3) {
  constexpr size_t kSizeOfVector = 5;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_4) {
  constexpr size_t kSizeOfVector = 14;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_5) {
  constexpr size_t kSizeOfVector = 42;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_6) {
  constexpr size_t kSizeOfVector = 132;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_7) {
  constexpr size_t kSizeOfVector = 429;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_8) {
  constexpr size_t kSizeOfVector = 1430;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_9) {
  constexpr size_t kSizeOfVector = 4862;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Katalan_size_10) {
  constexpr size_t kSizeOfVector = 16796;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Fibonacci_size_1) {
  constexpr size_t kSizeOfVector = 6765;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Fibonacci_size_2) {
  constexpr size_t kSizeOfVector = 10946;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_all, test_with_Fibonacci_size_3) {
  constexpr size_t kSizeOfVector = 17711;
  boost::mpi::communicator world;
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::vector<long long int> answer;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    in.resize(kSizeOfVector, 0);
    GetRandomVector(in, -100, 100);
    out.resize(kSizeOfVector, 0);
    answer = in;
    std::ranges::sort(answer);

    // Create task_data
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());
  }
  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortAll test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  if (world.rank() == 0) {
    EXPECT_EQ(answer, out);
  }
}