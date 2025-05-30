#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "all/kovalev_k_radix_sort_batcher_merge/include/header.hpp"
#include "core/task/include/task.hpp"

const long long int kMinLl = std::numeric_limits<long long>::lowest(), kMaxLl = std::numeric_limits<long long>::max();

TEST(kovalev_k_radix_sort_batcher_merge_all, zero_length) {
  std::vector<long long int> in;
  std::vector<long long int> out;
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    if (world.rank() == 0) {
      task_data_all->inputs_count.emplace_back(in.size());
      task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
      task_data_all->outputs_count.emplace_back(out.size());
    }
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task_all.ValidationImpl());
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, not_equal_lengths) {
  const unsigned int length = 10;
  std::vector<long long int> in(length);
  std::vector<long long int> out(2 * length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task_all.ValidationImpl());
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_No_viol_10_int) {
  const unsigned int length = 10;
  std::srand(std::time(nullptr));
  const long long int alpha = rand();
  std::vector<long long int> in(length, alpha);
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_2_int) {
  const unsigned int length = 2;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_5_int) {
  const unsigned int length = 5;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_10_int) {
  const unsigned int length = 10;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_12_int) {
  const unsigned int length = 12;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_13_int) {
  const unsigned int length = 13;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_17_int) {
  const unsigned int length = 17;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_24_int) {
  const unsigned int length = 24;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_49_int) {
  const unsigned int length = 49;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_83_int) {
  const unsigned int length = 83;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_137_int) {
  const unsigned int length = 137;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_274_int) {
  const unsigned int length = 274;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_524_int) {
  const unsigned int length = 524;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_793_int) {
  const unsigned int length = 793;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_1000_int) {
  const unsigned int length = 1000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_1762_int) {
  const unsigned int length = 1762;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_2158_int) {
  const unsigned int length = 2158;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_4763_int) {
  const unsigned int length = 4763;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_5000_int) {
  const unsigned int length = 5000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_15762_int) {
  const unsigned int length = 15762;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_27423_int) {
  const unsigned int length = 27423;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_76832_int) {
  const unsigned int length = 76832;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_128904_int) {
  const unsigned int length = 128904;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_164204_int) {
  const unsigned int length = 164204;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_178892_int) {
  const unsigned int length = 178892;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_215718_int) {
  const unsigned int length = 215718;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_244852_int) {
  const unsigned int length = 244852;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_398720_int) {
  const unsigned int length = 398720;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_601257_int) {
  const unsigned int length = 601257;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_875014_int) {
  const unsigned int length = 875014;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_radix_sort_batcher_merge_all, Test_1024789_int) {
  const unsigned int length = 1024789;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  kovalev_k_radix_sort_batcher_merge_all::TestTaskAll test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
    int count_viol = 0;
    for (unsigned int i = 0; i < length; i++) {
      if (out[i] != in[i]) {
        count_viol++;
      }
    }
    ASSERT_EQ(count_viol, 0);
  }
}