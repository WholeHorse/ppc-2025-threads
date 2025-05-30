#include "all/vedernikova_k_gauss/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numbers>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool vedernikova_k_gauss_all::Gauss::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.empty()) {
    return false;
  }
  width_ = task_data->inputs_count[0];
  g_height_ = task_data->inputs_count[1];
  channels_ = task_data->inputs_count[2];
  size_ = width_ * g_height_ * channels_;
  return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->outputs_count[0] == size_;
}

bool vedernikova_k_gauss_all::Gauss::PreProcessingImpl() {
  ComputeKernel();
  if (world_.rank() == 0) {
    width_ = task_data->inputs_count[0];
    g_height_ = task_data->inputs_count[1];
    channels_ = task_data->inputs_count[2];
    input_.resize(size_);
    output_.resize(size_);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + (width_ * g_height_ * channels_), input_.begin());
  }
  return true;
}

void vedernikova_k_gauss_all::Gauss::ComputeKernel(double sigma) {
  kernel_.resize(9);
  // For 3x3 kernel sigma from [1/3; 1/2] is required
  for (int i = 0; i < 9; i++) {
    int ik = (i % 3) - 1;
    int jk = (i / 3) - 1;
    kernel_[i] = std::exp(-1.0 * (ik * ik + jk * jk) / (2 * sigma * sigma)) / (2 * std::numbers::pi * sigma * sigma);
    ;
  }
  double amount = std::accumulate(kernel_.begin(), kernel_.end(), 0.0);
  for (auto&& it : kernel_) {
    it /= amount;
  }
}

uint8_t vedernikova_k_gauss_all::Gauss::GetPixel(uint32_t x, uint32_t y, uint32_t channel) {
  return local_input_[(y * width_ * channels_) + (x * channels_) + channel];
}

void vedernikova_k_gauss_all::Gauss::SetPixel(uint8_t value, uint32_t x, uint32_t y, uint32_t channel) {
  local_output_[(y * width_ * channels_) + (x * channels_) + channel] = value;
}

double vedernikova_k_gauss_all::Gauss::GetMultiplier(int i, int j) { return kernel_[(3 * (j + 1)) + (i + 1)]; }

void vedernikova_k_gauss_all::Gauss::ComputePixel(uint32_t x, uint32_t y) {
  for (uint32_t channel = 0; channel < channels_; channel++) {
    double brightness = 0;
    for (int shift_x = -1; shift_x <= 1; shift_x++) {
      for (int shift_y = -1; shift_y <= 1; shift_y++) {
        // if _x or _y out of image bounds, aproximating them with the nearest valid orthogonally adjacent pixels
        int xn = std::clamp((int)x + shift_x, 0, (int)width_ - 1);
        int yn = std::clamp((int)y + shift_y, 0, (int)l_height_ - 1);
        brightness += GetPixel(xn, yn, channel) * GetMultiplier(shift_x, shift_y);
      }
    }
    SetPixel(std::ceil(brightness), x, y, channel);
  }
}

namespace {
void FillAux(int participants, uint32_t cells_per_row, int overall_h, std::vector<int>& sendcounts,
             std::vector<int>& senddispls, std::vector<int>& recvcounts, std::vector<int>& recvdispls) {
  sendcounts.resize(participants, 0);
  senddispls.resize(participants, 0);
  recvcounts.resize(participants, 0);
  recvdispls.resize(participants, 0);

  const int base_rows = overall_h / participants;
  const int extra_rows = overall_h % participants;

  std::fill(sendcounts.begin(), sendcounts.begin() + extra_rows, base_rows + 1);
  std::fill(recvcounts.begin(), recvcounts.begin() + extra_rows, base_rows + 1);
  //
  std::fill(sendcounts.begin() + extra_rows, sendcounts.end(), base_rows);
  std::fill(recvcounts.begin() + extra_rows, recvcounts.end(), base_rows);
  if (participants > 1) {
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, senddispls.begin() + 1, std::plus<>());
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, recvdispls.begin() + 1, std::plus<>());
  }
  for (int i = 1; i < participants - 1; i++) {
    sendcounts[i] += 2;
  }
  if (participants > 1) {
    sendcounts.front() += 1;
    sendcounts.back() += 1;
  }
  for (int i = 1; i < participants; i++) {
    senddispls[i] -= 1;
  }
  for (int i = 0; i < participants; i++) {
    sendcounts[i] *= static_cast<int>(cells_per_row);
    senddispls[i] *= static_cast<int>(cells_per_row);
    recvcounts[i] *= static_cast<int>(cells_per_row);
    recvdispls[i] *= static_cast<int>(cells_per_row);
  }
}
}  // namespace

bool vedernikova_k_gauss_all::Gauss::RunImpl() {
  const int world_size = world_.size();
  const int rank = world_.rank();

  int overall_h{};
  if (rank == 0) {
    overall_h = static_cast<int>(g_height_);
  }
  boost::mpi::broadcast(world_, overall_h, 0);
  boost::mpi::broadcast(world_, width_, 0);
  boost::mpi::broadcast(world_, channels_, 0);

  if (overall_h == 1) {
    if (rank == 0) {
      local_input_ = input_;
      local_output_ = output_;
      SetPixel(GetPixel(0, 0, channels_ - 1), 0, 0, channels_ - 1);
      output_ = local_output_;
    }
    return true;
  }

  const uint32_t cells_per_row = width_ * channels_;

  const int participants = std::min(overall_h, world_size);
  if (world_.rank() >= participants) {
    world_.split(0);
    return true;
  }
  auto party = world_.split(1);

  std::vector<int> sendcounts;
  std::vector<int> senddispls;
  std::vector<int> recvcounts;
  std::vector<int> recvdispls;

  FillAux(participants, cells_per_row, overall_h, sendcounts, senddispls, recvcounts, recvdispls);

  l_height_ = sendcounts[rank] / cells_per_row;

  local_input_.resize(sendcounts[rank]);
  local_output_.resize(local_input_.size());
  boost::mpi::scatterv(party, input_.data(), sendcounts, senddispls, local_input_.data(), int(local_input_.size()), 0);

  const uint32_t tnum = std::min<uint32_t>(l_height_, ppc::util::GetPPCNumThreads());
  const uint32_t base = l_height_ / tnum;
  const uint32_t extra = l_height_ % tnum;

  std::vector<std::thread> threads(tnum);
  uint32_t cur = 0;
  for (uint32_t k = 0; k < tnum; ++k) {
    const uint32_t cnt = base + (k < extra ? 1 : 0);
    threads[k] = std::thread(
        [&](int rbegin, int rcnt) {
          for (int y = rbegin; y < rbegin + rcnt; ++y) {
            for (int x = 0; x < int(width_); ++x) {
              ComputePixel(x, y);
            }
          }
        },
        cur, cnt);
    cur += cnt;
  }
  for (auto& t : threads) {
    t.join();
  }

  boost::mpi::gatherv(party, local_output_.data() + (rank == 0 ? 0 : cells_per_row), recvcounts[rank], output_.data(),
                      recvcounts, recvdispls, 0);

  return true;
}

bool vedernikova_k_gauss_all::Gauss::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, reinterpret_cast<uint8_t*>(task_data->outputs[0]));
  }
  return true;
}