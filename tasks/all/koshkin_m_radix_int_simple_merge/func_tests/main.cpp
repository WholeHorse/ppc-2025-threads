#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {
enum VectorClass : std::uint8_t { kRandom, kReverse };

template <VectorClass cl>
std::vector<int> VecGen(size_t size);

template <>
std::vector<int> VecGen<kRandom>(size_t size) {
  constexpr int kK = 100;
  std::vector<int> v(size);

  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-kK, kK);
  std::ranges::generate(v, [&] { return dist(gen); });

  return v;
}
template <>
std::vector<int> VecGen<kReverse>(size_t size) {
  std::vector<int> v(size);
  std::iota(v.rbegin(), v.rend(), -2);
  return v;
}

template <VectorClass CL>
void PerformFuncTest(size_t size) {
  auto in = VecGen<CL>(size);
  decltype(in) out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (boost::mpi::communicator{}.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.push_back(in.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.push_back(out.size());
  }

  auto task = koshkin_m_radix_int_simple_merge::AllT(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (boost::mpi::communicator{}.rank() == 0) {
    EXPECT_TRUE(std::ranges::is_sorted(out));
  }
}
}  // namespace

#define RANDOM_RND_TEST(size) \
  TEST(koshkin_m_radix_int_simple_merge_all, random_##size) { PerformFuncTest<kRandom>(size); }
#define RANDOM_REV_TEST(size) \
  TEST(koshkin_m_radix_int_simple_merge_all, reversed_##size) { PerformFuncTest<kReverse>(size); }

RANDOM_RND_TEST(0)
RANDOM_RND_TEST(1)
RANDOM_RND_TEST(2)
RANDOM_RND_TEST(3)
RANDOM_RND_TEST(4)
RANDOM_RND_TEST(5)
RANDOM_RND_TEST(6)
RANDOM_RND_TEST(7)
RANDOM_RND_TEST(8)
RANDOM_RND_TEST(9)
RANDOM_RND_TEST(10)
RANDOM_RND_TEST(11)
RANDOM_RND_TEST(13)
RANDOM_RND_TEST(17)
RANDOM_RND_TEST(19)
RANDOM_RND_TEST(21)
RANDOM_RND_TEST(23)
RANDOM_RND_TEST(27)
RANDOM_RND_TEST(33)
RANDOM_RND_TEST(41)
RANDOM_RND_TEST(43)
RANDOM_RND_TEST(55)
RANDOM_RND_TEST(63)
RANDOM_RND_TEST(64)
RANDOM_RND_TEST(67)
RANDOM_RND_TEST(73)
RANDOM_RND_TEST(85)

RANDOM_REV_TEST(0)
RANDOM_REV_TEST(1)
RANDOM_REV_TEST(2)
RANDOM_REV_TEST(3)
RANDOM_REV_TEST(4)
RANDOM_REV_TEST(5)
RANDOM_REV_TEST(6)
RANDOM_REV_TEST(7)
RANDOM_REV_TEST(8)
RANDOM_REV_TEST(9)
RANDOM_REV_TEST(10)
RANDOM_REV_TEST(11)
RANDOM_REV_TEST(13)
RANDOM_REV_TEST(17)
RANDOM_REV_TEST(19)
RANDOM_REV_TEST(21)
RANDOM_REV_TEST(23)
RANDOM_REV_TEST(27)
RANDOM_REV_TEST(33)
RANDOM_REV_TEST(41)
RANDOM_REV_TEST(43)
RANDOM_REV_TEST(55)
RANDOM_REV_TEST(63)
RANDOM_REV_TEST(64)
RANDOM_REV_TEST(67)
RANDOM_REV_TEST(73)
RANDOM_REV_TEST(85)

#undef RANDOM_REV_TEST
#undef RANDOM_RND_TEST