#include "tbb/gnitienko_k_strassen_algorithm/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_1_ = std::vector<double>(in_ptr, in_ptr + input_size);

  in_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  input_2_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  size_ = static_cast<int>(std::sqrt(input_size));

  if ((input_size <= 0) || (input_size & (input_size - 1)) != 0) {
    int new_size = static_cast<int>(std::pow(2, std::ceil(std::log2(size_))));
    std::vector<double> extended_input_1(new_size * new_size, 0.0);
    std::vector<double> extended_input_2(new_size * new_size, 0.0);
    output_.resize(new_size * new_size);
    extend_ = new_size - size_;

    for (int i = 0; i < size_; ++i) {
      for (int j = 0; j < size_; ++j) {
        extended_input_1[(i * new_size) + j] = input_1_[(i * size_) + j];
        extended_input_2[(i * new_size) + j] = input_2_[(i * size_) + j];
      }
    }

    input_1_ = std::move(extended_input_1);
    input_2_ = std::move(extended_input_2);

    size_ = new_size;
  }
  return true;
}

bool gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::AddMatrix(const std::vector<double>& a,
                                                                   const std::vector<double>& b, std::vector<double>& c,
                                                                   int size) {
  for (int i = 0; i < size * size; ++i) {
    c[i] = a[i] + b[i];
  }
}

void gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::SubMatrix(const std::vector<double>& a,
                                                                   const std::vector<double>& b, std::vector<double>& c,
                                                                   int size) {
  for (int i = 0; i < size * size; ++i) {
    c[i] = a[i] - b[i];
  }
}

void gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::TrivialMultiply(const std::vector<double>& a,
                                                                         const std::vector<double>& b,
                                                                         std::vector<double>& c, int size) {
  tbb::parallel_for(0, size, [&](int i) {
    for (int j = 0; j < size; ++j) {
      double sum = 0.0;
      for (int k = 0; k < size; ++k) {
        sum += a[(i * size) + k] * b[(k * size) + j];
      }
      c[(i * size) + j] = sum;
    }
  });
}

void gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::StrassenMultiply(const std::vector<double>& a,
                                                                          const std::vector<double>& b,
                                                                          std::vector<double>& c, int size) {
  if (size <= TRIVIAL_MULTIPLICATION_BOUND_) {
    TrivialMultiply(a, b, c, size);
    return;
  }

  int half_size = size / 2;

  std::vector<double> a11(half_size * half_size);
  std::vector<double> a12(half_size * half_size);
  std::vector<double> a21(half_size * half_size);
  std::vector<double> a22(half_size * half_size);
  std::vector<double> b11(half_size * half_size);
  std::vector<double> b12(half_size * half_size);
  std::vector<double> b21(half_size * half_size);
  std::vector<double> b22(half_size * half_size);

  tbb::parallel_for(0, half_size, [&](int i) {
    for (int j = 0; j < half_size; ++j) {
      a11[(i * half_size) + j] = a[(i * size) + j];
      a12[(i * half_size) + j] = a[(i * size) + j + half_size];
      a21[(i * half_size) + j] = a[((i + half_size) * size) + j];
      a22[(i * half_size) + j] = a[((i + half_size) * size) + j + half_size];

      b11[(i * half_size) + j] = b[(i * size) + j];
      b12[(i * half_size) + j] = b[(i * size) + j + half_size];
      b21[(i * half_size) + j] = b[((i + half_size) * size) + j];
      b22[(i * half_size) + j] = b[((i + half_size) * size) + j + half_size];
    }
  });

  std::vector<double> d(half_size * half_size);
  std::vector<double> d1(half_size * half_size);
  std::vector<double> d2(half_size * half_size);
  std::vector<double> h1(half_size * half_size);
  std::vector<double> h2(half_size * half_size);
  std::vector<double> v1(half_size * half_size);
  std::vector<double> v2(half_size * half_size);

  tbb::task_group tg;

  tg.run([&]() {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    // d = (a11 + a22) * (b11 + b22)
    AddMatrix(a11, a22, temp_a, half_size);
    AddMatrix(b11, b22, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    // d1 = (a12 - a22) * (b21 + b22)
    SubMatrix(a12, a22, temp_a, half_size);
    AddMatrix(b21, b22, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d1, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    // d2 = (a21 - a11) * (b11 + b12)
    SubMatrix(a21, a11, temp_a, half_size);
    AddMatrix(b11, b12, temp_b, half_size);
    StrassenMultiply(temp_a, temp_b, d2, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_a(half_size * half_size);
    // h1 = (a11 + a12) * b22
    AddMatrix(a11, a12, temp_a, half_size);
    StrassenMultiply(temp_a, b22, h1, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_a(half_size * half_size);
    std::vector<double> temp_b(half_size * half_size);
    // h2 = (a21 + a22) * b11
    AddMatrix(a21, a22, temp_a, half_size);
    StrassenMultiply(temp_a, b11, h2, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_b(half_size * half_size);
    // v1 = a22 * (b21 - b11)
    SubMatrix(b21, b11, temp_b, half_size);
    StrassenMultiply(a22, temp_b, v1, half_size);
  });

  tg.run([&]() {
    std::vector<double> temp_b(half_size * half_size);
    // v2 = a11 * (b12 - b22)
    SubMatrix(b12, b22, temp_b, half_size);
    StrassenMultiply(a11, temp_b, v2, half_size);
  });

  tg.wait();

  std::vector<double> c11(half_size * half_size);
  std::vector<double> c12(half_size * half_size);
  std::vector<double> c21(half_size * half_size);
  std::vector<double> c22(half_size * half_size);

  AddMatrix(d, d1, c11, half_size);
  AddMatrix(c11, v1, c11, half_size);
  SubMatrix(c11, h1, c11, half_size);
  AddMatrix(v2, h1, c12, half_size);
  AddMatrix(v1, h2, c21, half_size);
  AddMatrix(d, d2, c22, half_size);
  AddMatrix(c22, v2, c22, half_size);
  SubMatrix(c22, h2, c22, half_size);

  tbb::parallel_for(0, half_size, [&](int i) {
    for (int j = 0; j < half_size; ++j) {
      c[(i * size) + j] = c11[(i * half_size) + j];
      c[(i * size) + j + half_size] = c12[(i * half_size) + j];
      c[((i + half_size) * size) + j] = c21[(i * half_size) + j];
      c[((i + half_size) * size) + j + half_size] = c22[(i * half_size) + j];
    }
  });
}

bool gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] { StrassenMultiply(input_1_, input_2_, output_, size_); });
  if (extend_ != 0) {
    int original_size = size_ - extend_;
    std::vector<double> res(original_size * original_size);

    for (int i = 0; i < original_size; ++i) {
      for (int j = 0; j < original_size; ++j) {
        res[(i * original_size) + j] = output_[(i * size_) + j];
      }
    }

    output_ = std::move(res);
  }
  return true;
}

bool gnitienko_k_strassen_algorithm_tbb::StrassenAlgTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    ;
  }
  return true;
}