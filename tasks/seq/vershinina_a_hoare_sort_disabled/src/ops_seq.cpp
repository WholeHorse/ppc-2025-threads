#include "seq/vershinina_a_hoare_sort/include/ops_seq.hpp"

#include <algorithm>

int vershinina_a_hoare_sort_seq::TestTaskSequential::Partition(int *s_vec, int first, int last) {
  int i = first - 1;
  value_ = s_vec[last];

  for (int j = first; j <= last - 1; j++) {
    if (s_vec[j] <= value_) {
      i++;
      std::swap(s_vec[i], s_vec[j]);
    }
  }
  std::swap(s_vec[i + 1], s_vec[last]);
  return i + 1;
}

void vershinina_a_hoare_sort_seq::TestTaskSequential::HoareSort(int *s_vec, int first, int last) {
  if (first < last) {
    int iter = Partition(s_vec, first, last);
    HoareSort(s_vec, first, iter - 1);
    HoareSort(s_vec, iter + 1, last);
  }
}

bool vershinina_a_hoare_sort_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = reinterpret_cast<int *>(task_data->inputs[0]);
  n_ = (int)(task_data->inputs_count[0]);
  return true;
}

bool vershinina_a_hoare_sort_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 1;
}

bool vershinina_a_hoare_sort_seq::TestTaskSequential::RunImpl() {
  if (n_ <= 1) {
    return true;
  }
  HoareSort(input_, 0, n_ - 1);
  return true;
}

bool vershinina_a_hoare_sort_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(input_, input_ + n_, reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
