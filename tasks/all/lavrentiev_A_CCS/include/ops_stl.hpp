#pragma once

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lavrentiev_a_ccs_all {
struct Sparse {
  std::pair<int, int> size;
  std::vector<double> elements;
  std::vector<int> rows;
  std::vector<int> columnsSum;
  template <typename Archive>
  // NOLINTNEXTLINE(readability-identifier-naming)
  void serialize(Archive& archive, const unsigned int) {
    archive & size.first;
    archive & size.second;
    archive & elements;
    archive & rows;
    archive & columnsSum;
  }
};
class CCSALL : public ppc::core::Task {
 private:
  [[nodiscard]] bool IsEmpty() const;
  static Sparse ConvertToSparse(std::pair<int, int> size, const std::vector<double>& values);
  static Sparse Transpose(const Sparse& sparse);
  static Sparse MatMul(const Sparse& matrix1, const Sparse& matrix2, int interval_begin, int interval_end);
  static int GetElementsCount(int index, const std::vector<int>& columns_sum);
  static std::vector<double> ConvertFromSparse(const Sparse& matrix);
  static int CalculateStartIndex(int index, const std::vector<int>& columns_sum);
  void GetDisplacements();
  void CollectSizes();
  void CollectData();
  void AddData(const std::vector<double>& data, int past_data, int index);

  Sparse A_;
  Sparse B_;
  Sparse Answer_;
  Sparse Process_data_;
  int resize_data_;
  boost::mpi::communicator world_;
  std::vector<int> displ_;
  std::vector<double> sending_data_;
  std::vector<int> sum_sizes_;

 public:
  explicit CCSALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lavrentiev_a_ccs_all