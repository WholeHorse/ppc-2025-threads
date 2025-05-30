#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> RandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-5000, 8000);
  std::vector<double> vec(size);
  std::ranges::generate(vec, [&dist, &gen] { return dist(gen); });
  return vec;
}

void STest(std::vector<double> in) {
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_all::TestTaskParallelOmpMpi(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

void STest(size_t size) { STest(RandomVector(size)); }
}  // namespace

TEST(petrov_a_radix_double_batcher_all, test_0) { STest(0); }

TEST(petrov_a_radix_double_batcher_all, test_1) { STest(1); }
TEST(petrov_a_radix_double_batcher_all, test_2) { STest(2); }
TEST(petrov_a_radix_double_batcher_all, test_3) { STest(3); }
TEST(petrov_a_radix_double_batcher_all, test_4) { STest(4); }
TEST(petrov_a_radix_double_batcher_all, test_5) { STest(5); }
TEST(petrov_a_radix_double_batcher_all, test_6) { STest(6); }
TEST(petrov_a_radix_double_batcher_all, test_7) { STest(7); }
TEST(petrov_a_radix_double_batcher_all, test_8) { STest(8); }
TEST(petrov_a_radix_double_batcher_all, test_9) { STest(9); }
TEST(petrov_a_radix_double_batcher_all, test_10) { STest(10); }
TEST(petrov_a_radix_double_batcher_all, test_11) { STest(11); }
TEST(petrov_a_radix_double_batcher_all, test_12) { STest(12); }
TEST(petrov_a_radix_double_batcher_all, test_13) { STest(13); }
TEST(petrov_a_radix_double_batcher_all, test_14) { STest(14); }
TEST(petrov_a_radix_double_batcher_all, test_15) { STest(15); }
TEST(petrov_a_radix_double_batcher_all, test_16) { STest(16); }
TEST(petrov_a_radix_double_batcher_all, test_17) { STest(17); }
TEST(petrov_a_radix_double_batcher_all, test_18) { STest(18); }
TEST(petrov_a_radix_double_batcher_all, test_19) { STest(19); }
TEST(petrov_a_radix_double_batcher_all, test_20) { STest(20); }
TEST(petrov_a_radix_double_batcher_all, test_21) { STest(21); }
TEST(petrov_a_radix_double_batcher_all, test_22) { STest(22); }
TEST(petrov_a_radix_double_batcher_all, test_23) { STest(23); }
TEST(petrov_a_radix_double_batcher_all, test_24) { STest(24); }
TEST(petrov_a_radix_double_batcher_all, test_25) { STest(25); }
TEST(petrov_a_radix_double_batcher_all, test_26) { STest(26); }
TEST(petrov_a_radix_double_batcher_all, test_27) { STest(27); }
TEST(petrov_a_radix_double_batcher_all, test_28) { STest(28); }
TEST(petrov_a_radix_double_batcher_all, test_29) { STest(29); }
TEST(petrov_a_radix_double_batcher_all, test_30) { STest(30); }
TEST(petrov_a_radix_double_batcher_all, test_31) { STest(31); }
TEST(petrov_a_radix_double_batcher_all, test_32) { STest(32); }
TEST(petrov_a_radix_double_batcher_all, test_33) { STest(33); }
TEST(petrov_a_radix_double_batcher_all, test_34) { STest(34); }
TEST(petrov_a_radix_double_batcher_all, test_35) { STest(35); }
TEST(petrov_a_radix_double_batcher_all, test_36) { STest(36); }
TEST(petrov_a_radix_double_batcher_all, test_37) { STest(37); }
TEST(petrov_a_radix_double_batcher_all, test_38) { STest(38); }
TEST(petrov_a_radix_double_batcher_all, test_39) { STest(39); }
TEST(petrov_a_radix_double_batcher_all, test_40) { STest(40); }
TEST(petrov_a_radix_double_batcher_all, test_41) { STest(41); }
TEST(petrov_a_radix_double_batcher_all, test_42) { STest(42); }
TEST(petrov_a_radix_double_batcher_all, test_43) { STest(43); }
TEST(petrov_a_radix_double_batcher_all, test_44) { STest(44); }
TEST(petrov_a_radix_double_batcher_all, test_45) { STest(45); }
TEST(petrov_a_radix_double_batcher_all, test_46) { STest(46); }
TEST(petrov_a_radix_double_batcher_all, test_47) { STest(47); }
TEST(petrov_a_radix_double_batcher_all, test_48) { STest(48); }
TEST(petrov_a_radix_double_batcher_all, test_49) { STest(49); }
TEST(petrov_a_radix_double_batcher_all, test_50) { STest(50); }
TEST(petrov_a_radix_double_batcher_all, test_51) { STest(51); }
TEST(petrov_a_radix_double_batcher_all, test_52) { STest(52); }
TEST(petrov_a_radix_double_batcher_all, test_53) { STest(53); }
TEST(petrov_a_radix_double_batcher_all, test_54) { STest(54); }
TEST(petrov_a_radix_double_batcher_all, test_55) { STest(55); }
TEST(petrov_a_radix_double_batcher_all, test_56) { STest(56); }
TEST(petrov_a_radix_double_batcher_all, test_57) { STest(57); }
TEST(petrov_a_radix_double_batcher_all, test_58) { STest(58); }
TEST(petrov_a_radix_double_batcher_all, test_59) { STest(59); }
TEST(petrov_a_radix_double_batcher_all, test_60) { STest(60); }
TEST(petrov_a_radix_double_batcher_all, test_61) { STest(61); }
TEST(petrov_a_radix_double_batcher_all, test_62) { STest(62); }
TEST(petrov_a_radix_double_batcher_all, test_63) { STest(63); }
TEST(petrov_a_radix_double_batcher_all, test_64) { STest(64); }
TEST(petrov_a_radix_double_batcher_all, test_65) { STest(65); }
TEST(petrov_a_radix_double_batcher_all, test_66) { STest(66); }
TEST(petrov_a_radix_double_batcher_all, test_67) { STest(67); }
TEST(petrov_a_radix_double_batcher_all, test_68) { STest(68); }
TEST(petrov_a_radix_double_batcher_all, test_69) { STest(69); }
TEST(petrov_a_radix_double_batcher_all, test_70) { STest(70); }
TEST(petrov_a_radix_double_batcher_all, test_71) { STest(71); }
TEST(petrov_a_radix_double_batcher_all, test_72) { STest(72); }
TEST(petrov_a_radix_double_batcher_all, test_73) { STest(73); }
TEST(petrov_a_radix_double_batcher_all, test_74) { STest(74); }
TEST(petrov_a_radix_double_batcher_all, test_75) { STest(75); }
TEST(petrov_a_radix_double_batcher_all, test_76) { STest(76); }
TEST(petrov_a_radix_double_batcher_all, test_77) { STest(77); }
TEST(petrov_a_radix_double_batcher_all, test_78) { STest(78); }
TEST(petrov_a_radix_double_batcher_all, test_79) { STest(79); }
TEST(petrov_a_radix_double_batcher_all, test_80) { STest(80); }
TEST(petrov_a_radix_double_batcher_all, test_81) { STest(81); }
TEST(petrov_a_radix_double_batcher_all, test_82) { STest(82); }
TEST(petrov_a_radix_double_batcher_all, test_83) { STest(83); }
TEST(petrov_a_radix_double_batcher_all, test_84) { STest(84); }
TEST(petrov_a_radix_double_batcher_all, test_85) { STest(85); }
TEST(petrov_a_radix_double_batcher_all, test_86) { STest(86); }
TEST(petrov_a_radix_double_batcher_all, test_87) { STest(87); }
TEST(petrov_a_radix_double_batcher_all, test_88) { STest(88); }
TEST(petrov_a_radix_double_batcher_all, test_89) { STest(89); }
TEST(petrov_a_radix_double_batcher_all, test_90) { STest(90); }
TEST(petrov_a_radix_double_batcher_all, test_91) { STest(91); }
TEST(petrov_a_radix_double_batcher_all, test_92) { STest(92); }
TEST(petrov_a_radix_double_batcher_all, test_93) { STest(93); }
TEST(petrov_a_radix_double_batcher_all, test_94) { STest(94); }
TEST(petrov_a_radix_double_batcher_all, test_95) { STest(95); }
TEST(petrov_a_radix_double_batcher_all, test_96) { STest(96); }
TEST(petrov_a_radix_double_batcher_all, test_97) { STest(97); }
TEST(petrov_a_radix_double_batcher_all, test_98) { STest(98); }
TEST(petrov_a_radix_double_batcher_all, test_99) { STest(99); }
TEST(petrov_a_radix_double_batcher_all, test_111) { STest(111); }
TEST(petrov_a_radix_double_batcher_all, test_213) { STest(213); }

TEST(petrov_a_radix_double_batcher_all, test_already_sorted) { STest({1, 2, 3, 4, 5, 10, 15, 16, 100}); }

TEST(petrov_a_radix_double_batcher_all, test_same) { STest(std::vector<double>(81, 555)); }