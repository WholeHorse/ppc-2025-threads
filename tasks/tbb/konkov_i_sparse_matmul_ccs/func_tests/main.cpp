#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/konkov_i_sparse_matmul_ccs/include/ops_tbb.hpp"

TEST(konkov_i_SparseMatmulTest_tbb, SimpleTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  task.A_values = {5.0, 7.0, 9.0};
  task.A_row_indices = {0, 1, 2};
  task.A_col_ptr = {0, 1, 2, 3};
  task.rowsA = 3;
  task.colsA = 3;

  task.B_values = {3.0, 4.0, 2.0};
  task.B_row_indices = {0, 1, 2};
  task.B_col_ptr = {0, 1, 2, 3};
  task.rowsB = 3;
  task.colsB = 3;

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  std::vector<double> expected_values = {15.0, 28.0, 18.0};
  EXPECT_EQ(task.C_values, expected_values);
}

TEST(konkov_i_SparseMatmulTest_tbb, EmptyMatrixTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  task.A_col_ptr = {0};
  task.B_col_ptr = {0};
  task.rowsA = 0;
  task.colsA = 0;
  task.rowsB = 0;
  task.colsB = 0;

  EXPECT_FALSE(task.ValidationImpl());
}

TEST(konkov_i_SparseMatmulTest_tbb, ComplexTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  task.A_values = {1.0, 2.0};
  task.A_row_indices = {0, 2};
  task.A_col_ptr = {0, 1, 1, 2};
  task.rowsA = 3;
  task.colsA = 3;

  task.B_values = {3.0, 4.0};
  task.B_row_indices = {1, 2};
  task.B_col_ptr = {0, 0, 1, 2};
  task.rowsB = 3;
  task.colsB = 3;

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  std::vector<double> expected_values = {8.0};
  EXPECT_EQ(task.C_values, expected_values);
}

TEST(konkov_i_SparseMatmulTest_tbb, IdentityMatrixTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  task.A_values = {1.0, 1.0, 1.0};
  task.A_row_indices = {0, 1, 2};
  task.A_col_ptr = {0, 1, 2, 3};
  task.rowsA = 3;
  task.colsA = 3;

  task.B_values = {1.0, 2.0, 3.0};
  task.B_row_indices = {0, 1, 2};
  task.B_col_ptr = {0, 1, 2, 3};
  task.rowsB = 3;
  task.colsB = 3;

  EXPECT_TRUE(task.RunImpl());

  EXPECT_EQ(task.C_values, task.B_values);
  EXPECT_EQ(task.C_row_indices, task.B_row_indices);
  EXPECT_EQ(task.C_col_ptr, task.B_col_ptr);
}
