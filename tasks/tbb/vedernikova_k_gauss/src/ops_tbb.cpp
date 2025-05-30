#include "tbb/vedernikova_k_gauss/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

bool vedernikova_k_gauss_tbb::Gauss::ValidationImpl() {
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.empty()) {
    return false;
  }

  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  channels_ = task_data->inputs_count[2];
  size_ = width_ * height_ * channels_;
  return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->outputs_count[0] == size_;
}

bool vedernikova_k_gauss_tbb::Gauss::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  channels_ = task_data->inputs_count[2];
  kernel_.resize(9);
  input_.resize(size_);
  output_.resize(size_);
  ComputeKernel();
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + (width_ * height_ * channels_), input_.begin());
  return true;
}

bool vedernikova_k_gauss_tbb::Gauss::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<uint8_t*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void vedernikova_k_gauss_tbb::Gauss::ComputeKernel(double sigma) {
  // For 3x3 kernel sigma from [1/3; 1/2] is required
  for (int i = 0; i < 9; i++) {
    int ik = (i % 3) - 1;
    int jk = (i / 3) - 1;
    kernel_[i] = std::exp(-1.0 * (ik * ik + jk * jk) / (2 * sigma * sigma)) / (2 * std::numbers::pi * sigma * sigma);
  }
  double amount = std::accumulate(kernel_.begin(), kernel_.end(), 0.0);
  for (auto&& it : kernel_) {
    it /= amount;
  }
}

uint8_t vedernikova_k_gauss_tbb::Gauss::GetPixel(uint32_t x, uint32_t y, uint32_t channel) {
  return input_[(y * width_ * channels_) + (x * channels_) + channel];
}

void vedernikova_k_gauss_tbb::Gauss::SetPixel(uint8_t value, uint32_t x, uint32_t y, uint32_t channel) {
  output_[(y * width_ * channels_) + (x * channels_) + channel] = value;
}

double vedernikova_k_gauss_tbb::Gauss::GetMultiplier(int i, int j) { return kernel_[(3 * (j + 1)) + (i + 1)]; }

void vedernikova_k_gauss_tbb::Gauss::ComputePixel(uint32_t x, uint32_t y) {
  for (uint32_t channel = 0; channel < channels_; channel++) {
    double brightness = 0;
    for (int shift_x = -1; shift_x <= 1; shift_x++) {
      for (int shift_y = -1; shift_y <= 1; shift_y++) {
        int xn = std::clamp((int)x + shift_x, 0, (int)width_ - 1);
        int yn = std::clamp((int)y + shift_y, 0, (int)height_ - 1);
        brightness += GetPixel(xn, yn, channel) * GetMultiplier(shift_x, shift_y);
      }
    }

    SetPixel(std::ceil(brightness), x, y, channel);
  }
}

bool vedernikova_k_gauss_tbb::Gauss::RunImpl() {
  if (height_ == 1) {
    SetPixel(GetPixel(0, 0, channels_ - 1), 0, 0, channels_ - 1);
    return true;
  }
  oneapi::tbb::task_arena arena((ppc::util::GetPPCNumThreads()));
  arena.execute([&] {
    oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, int(height_)), [&](const tbb::blocked_range<int>&) {
      for (int j = 0; j < (int)height_; ++j) {
        for (int i = 0; i < (int)width_; ++i) {
          ComputePixel(i, j);
        }
      }
    });
  });

  return true;
}