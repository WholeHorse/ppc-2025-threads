#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "all/vedernikova_k_gauss/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

using TaskVars = std::tuple<uint32_t, uint32_t, uint32_t, Image, Image>;
namespace {
class vedernikova_k_gauss_test_all  // NOLINT(readability-identifier-naming)
    : public ::testing::TestWithParam<TaskVars> {
 protected:
};

TEST_F(vedernikova_k_gauss_test_all, validation_fails_not_enough_params) {
  if (boost::mpi::communicator{}.rank() != 0) {
    return;
  }

  Image in(15, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_all::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_all, validation_fails_no_input_image) {
  if (boost::mpi::communicator{}.rank() != 0) {
    return;
  }

  Image in(13, 100);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_all::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_all, validation_fails_no_output_buffer) {
  if (boost::mpi::communicator{}.rank() != 0) {
    return;
  }

  Image in(17, 137);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_all::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_all, validation_fails_in_and_out_sizes_are_different) {
  if (boost::mpi::communicator{}.rank() != 0) {
    return;
  }

  Image in(37, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() + 1);

  vedernikova_k_gauss_all::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_P(vedernikova_k_gauss_test_all, returns_correct_blurred_image) {
  boost::mpi::communicator world;
  const auto &[width, height, channels, in, exp] = GetParam();
  Image out(in.size(), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
    task_data->inputs_count.emplace_back(width);
    task_data->inputs_count.emplace_back(height);
    task_data->inputs_count.emplace_back(channels);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  // Create Task
  vedernikova_k_gauss_all::Gauss task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, exp);
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(vedernikova_k_gauss_test_all, vedernikova_k_gauss_test_all, ::testing::Values(
    TaskVars(
      1, 1, 1,
      {255},
      {255}
    ),
    TaskVars(
      3, 3, 1, 
      {
        15, 15, 15,
        15, 15, 15,
        15, 15, 15,
      },
      {
        15, 15, 15,
        15, 15, 15,
        15, 15, 15,
      }
    ),
    TaskVars(
      5, 5, 1, 
      {
        255, 0, 255, 0, 255, 
        0, 255, 0, 255, 0, 
        255, 0, 255, 0, 255, 
        0, 255, 0, 255, 0, 
        255, 0, 255, 0, 255, 
      }, 
      {
        231, 37, 219, 37, 231,
        37, 209, 47, 209, 37,
        219, 47, 209, 47, 219,
        37, 209, 47, 209, 37,
        231, 37, 219, 37, 231,
      }
    ),
    TaskVars(
      5, 3, 1, 
      {
        255, 0, 255, 0, 255, 
        0, 255, 0, 255, 0, 
        255, 0, 255, 0, 255, 
        
      }, 
      {
        231, 37, 219, 37, 231,
        37, 209, 47, 209, 37,
        231, 37, 219, 37, 231,
        
      }
    ),
    TaskVars(
      6, 5, 1, 
      {
        25,45,56,67,78,13,
        24,35,46,57,14,25,
        36,47,58,69,123,243,
        76,95,73,56,234,134,
        245,197,72,185,136,73
      }, 
      {
        26, 45, 56, 67, 72,17, 
        26, 37, 48, 56,26, 35, 
        38, 49, 59,71,126,222,
        84,96,74,72,211,141,
        235,189,84,172,140,80
      }
    ),
    TaskVars(
      6, 7, 1, 
      {
        25,45,56,67,78,56,
        13,24,35,46,57,86,
        14,25,36,47,58,96,
        69,123,243,76,95,34,
        73,56,234,134,245,57,
        197,72,185,136,73,87,
        190,80,170,120,70,90
      }, 
      {
        26, 44, 55, 66, 76,59, 
        15, 26, 37, 48,59, 84, 
        18, 31, 46,49,61,91,
        69,119,219,87,97,42,
        79,70,219,142,216,67,
        185,84,179,135,85,86,
        185,90,164,121,74,89
      }
    ),
    TaskVars(
      4, 6, 1, 
      {
        25, 50, 60, 80,
        40, 65, 70, 90,
        58, 79, 96, 120,
        70, 94, 100, 130,
        97, 112, 122, 145,
        111, 132, 168, 168
        
      }, 
      {
        28, 50, 62, 80, 
        42, 64, 72, 90, 
        59, 79,96, 118, 
        72, 94, 103,129,
        98, 112, 124, 145,
        112, 132, 164, 167
        
      }
    ),
    TaskVars(
      100, 100, 1,
      Image(10000, (uint8_t)128),
      Image(10000, (uint8_t)128)
    )
    
  )
);
}  // namespace