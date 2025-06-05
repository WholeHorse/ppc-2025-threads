#pragma once

#include <boost/mpi/collectives.hpp>
#include <map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_marc_on_bin_image_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::map<int, int> BuildParentMap(const std::vector<int>& global_output,
                                           const std::vector<int>& all_equivalences);
  static int FindRoot(std::map<int, int>& parent, int x);
  static void ProcessEquivalences(std::map<int, int>& parent, const std::vector<int>& all_equivalences);
  static void RenumberLabels(std::vector<int>& global_output);
  void ProcessPixel(int row, int col);
  void AssignNewLabel(int row, int col);
  void AssignMinLabel(int row, int col, const std::vector<int>& neighbors);
  std::vector<int> FindAdjacentLabels(int row, int col);
  int FindRoot(int label);
  void UnionLabels(int label1, int label2);
  void LocalLabeling();
  void MergeLabelsBetweenProcesses();
  void UpdateGlobalLabels();

  int rows_{}, cols_{};
  int rank_{}, num_procs_{};
  int base_rows_{}, remainder_{};
  int local_start_row_{}, local_rows_{};
  int base_label_{};

  std::vector<int> input_image_;
  std::vector<int> local_image_;
  std::vector<int> local_output_;
  std::vector<int> local_label_parent_;
  std::vector<int> global_output_;
  std::vector<int> global_label_parent_;
  std::vector<int> all_equivalences_;

  int current_label_ = 0;
};

}  // namespace naumov_b_marc_on_bin_image_all