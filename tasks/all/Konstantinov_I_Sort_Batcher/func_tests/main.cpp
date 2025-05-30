#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace mpi = boost::mpi;
namespace konstantinov_i_sort_batcher_all {
namespace {
void VerifyNanPresence(const std::vector<double> &out) {
  bool has_nan = std::ranges::any_of(out, [](double val) { return std::isnan(val); });
  EXPECT_TRUE(has_nan);
}

void VerifySortingOrder(const std::vector<double> &out) {
  bool ordered = true;
  for (size_t i = 1; i < out.size(); ++i) {
    if (!std::isnan(out[i]) && !std::isnan(out[i - 1]) && out[i] < out[i - 1]) {
      ordered = false;
      break;
    }
  }
  EXPECT_TRUE(ordered);
}

void VerifySpecialValuesHandling(const std::vector<double> &out) {
  konstantinov_i_sort_batcher_all::VerifyNanPresence(out);
  konstantinov_i_sort_batcher_all::VerifySortingOrder(out);
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

TEST(Konstantinov_I_Sort_Batcher_all, invalid_input) {
  mpi::communicator world;
  std::vector<double> in{1.0};
  std::vector<double> out(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  if (world.rank() == 0) {
    EXPECT_EQ(test_task.ValidationImpl(), false);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, negative_values) {
  mpi::communicator world;
  std::vector<double> in{-3.14, -1.0, -104.5, -0.1, -990.90};
  std::vector<double> exp_out{-990.90, -104.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, special_floating_values) {
  mpi::communicator world;
  std::vector<double> in{std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         3.14,
                         -2.5,
                         0.0};
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  if (world.rank() == 0) {
    konstantinov_i_sort_batcher_all::VerifySpecialValuesHandling(out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, nearly_sorted_input) {
  mpi::communicator world;
  std::vector<double> in{1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 9.0, 8.0, 10.0};
  std::vector<double> exp_out{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, positive_values) {
  std::vector<double> in{3.14, 1.0, 104.5, 0.1, 990.90};
  std::vector<double> exp_out{0.1, 1.0, 3.14, 104.5, 990.90};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, mixed_values) {
  std::vector<double> in{0.0, -2.4, 3.4, -1.1, 2.2};
  std::vector<double> exp_out{-2.4, -1.1, 0.0, 2.2, 3.4};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, duplicate_values) {
  std::vector<double> in{6.5, 1.2, 6.5, 3.3, 1.2};
  std::vector<double> exp_out{1.2, 1.2, 3.3, 6.5, 6.5};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, sorted_input) {
  std::vector<double> in{-6.6, -3.3, 0.0, 4.4, 6.6};
  std::vector<double> exp_out{-6.6, -3.3, 0.0, 4.4, 6.6};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, large_array) {
  constexpr size_t kSize = 100000;
  std::vector<double> in(kSize);
  std::vector<double> exp_out(kSize);

  for (size_t i = 0; i < kSize; ++i) {
    in[i] = static_cast<double>(kSize - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kSize);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}