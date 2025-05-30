#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

const std::vector<double> kErnel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

int ComputeBlockSizes(std::vector<int> &block_size, int width, int height, const boost::mpi::communicator &world) {
  int delta = width / world.size();
  if (width % world.size() != 0) {
    delta++;
  }
  if (world.rank() >= world.size() - (world.size() * delta) + width) {
    delta--;
  }
  delta += 2;

  gather(world, delta, block_size.data(), 0);
  return delta;
}

std::vector<double> CopySingleProcessImage(const std::vector<double> &image, int width, int height) {
  std::vector<double> local_image((width + 2) * height, 0);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      local_image[(i * (width + 2)) + j] = image[(i * width) + j - 1];
    }
  }
  return local_image;
}

void SendBlocksToOtherProcesses(const std::vector<double> &image, int delta, const std::vector<int> &block_size,
                                int width, int height, const boost::mpi::communicator &world,
                                std::vector<double> &local_image) {
  std::vector<double> send_image(delta * height);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < delta - 1; ++j) {
      local_image[(i * delta) + j + 1] = image[(i * width) + j];
    }
  }

  int idx = delta - 2;
  for (int proc = 1; proc < world.size(); ++proc) {
    send_image.assign(delta * height, 0);
    int block = block_size[proc];

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < height; ++i) {
      for (int j = -1; j < block - (proc == world.size() - 1 ? 2 : 1); ++j) {
        send_image[(i * block) + j + 1] = image[(i * width) + j + idx];
      }
    }

    if (proc != world.size() - 1) {
      idx += block - 2;
    }

    world.send(proc, 0, send_image.data(), block * height);
  }
}

std::vector<double> DistributeImage(const std::vector<double> &image, int delta, const std::vector<int> &block_size,
                                    int width, int height, const boost::mpi::communicator &world) {
  std::vector<double> local_image(delta * height, 0);

  if (world.size() == 1) {
    return CopySingleProcessImage(image, width, height);
  }

  if (world.rank() == 0) {
    SendBlocksToOtherProcesses(image, delta, block_size, width, height, world, local_image);
  } else {
    world.recv(0, 0, local_image.data(), delta * height);
  }

  return local_image;
}

std::vector<double> FilterLocalImage(const std::vector<double> &local_image, int delta, int height) {
  std::vector<double> local_filtered_image(delta * height, 0);

#pragma omp parallel for schedule(static)
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += local_image[((i - l) * delta) + j - k] * kErnel[((l + 1) * 3) + k + 1];
        }
      }
      local_filtered_image[(i * delta) + j] = sum;
    }
  }

  return local_filtered_image;
}

std::vector<double> ExtractFilteredData(const std::vector<double> &filtered, int delta, int height) {
  std::vector<double> back_image((delta - 2) * height);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      back_image[(i * (delta - 2)) + j - 1] = filtered[(i * delta) + j];
    }
  }

  return back_image;
}

std::vector<double> AssembleFinalImage(const std::vector<double> &gathered_image, const std::vector<int> &block_size,
                                       int width, int height, const boost::mpi::communicator &world) {
  std::vector<double> filtered_image(width * height);

  int idx = 0;
  for (int proc = 0; proc < world.size(); ++proc) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < block_size[proc] - 2; ++j) {
        filtered_image[(i * width) + j + idx] = gathered_image[(i * (block_size[proc] - 2)) + j + (idx * height)];
      }
    }
    idx += block_size[proc] - 2;
  }

  return filtered_image;
}

}  // namespace

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    // Init image
    image_ = std::vector<double>(task_data->inputs_count[0]);
    auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

    width_ = static_cast<int>(task_data->inputs_count[1]);
    height_ = static_cast<int>(task_data->inputs_count[2]);

    // Init filtered image
    filtered_image_ = std::vector<double>(width_ * height_, 0);
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    // Init image
    image_ = std::vector<double>(task_data->inputs_count[0]);
    auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

    size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

    // Check pixels range from 0 to 255
    for (size_t i = 0; i < img_size; ++i) {
      if (image_[i] < 0 || image_[i] > 255) {
        return false;
      }
    }

    // Check size of image
    return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == img_size &&
           task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 &&
           task_data->inputs_count[2] >= 3;
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::RunImpl() {
  broadcast(world_, width_, 0);
  broadcast(world_, height_, 0);

  std::vector<int> block_size(world_.size());
  int delta = ComputeBlockSizes(block_size, width_, height_, world_);

  std::vector<double> local_image = DistributeImage(image_, delta, block_size, width_, height_, world_);
  std::vector<double> local_filtered_image = FilterLocalImage(local_image, delta, height_);
  std::vector<double> back_image = ExtractFilteredData(local_filtered_image, delta, height_);

  std::vector<int> recv_counts(world_.size());
  if (world_.rank() == 0) {
    for (int i = 0; i < world_.size(); ++i) {
      recv_counts[i] = (block_size[i] - 2) * height_;
    }
  }

  std::vector<double> gathered_image(width_ * height_);
  gatherv(world_, back_image, gathered_image.data(), recv_counts, 0);

  if (world_.rank() == 0) {
    filtered_image_ = AssembleFinalImage(gathered_image, block_size, width_, height_, world_);
  }

  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  }
  return true;
}
