#include "../include/chc_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "../include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_all;

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    int height = *ptr;
    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    int width = *ptr;
    int pixels_size = static_cast<int>(task_data->inputs_count[0]);
    return height > 0 && width > 0 && (height * width) == pixels_size;
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    int height = *ptr;
    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    int width = *ptr;

    std::vector<int> pixels(task_data->inputs_count[0]);
    ptr = reinterpret_cast<int *>(task_data->inputs[2]);
    std::ranges::copy(ptr, ptr + task_data->inputs_count[0], pixels.begin());

    Image image(height, width, pixels);
    imageIn_ = image;
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::RunImpl() {
  std::vector<Component> components;

  if (world_.rank() == 0) {
    components = FindComponentsOMP(imageIn_);
  }

  if (world_.size() <= 1) {
    hullsOut_ = QuickHullAllOMP(components);
  } else {
    hullsOut_ = QuickHullAllMPIOMP(components, imageIn_.width);
  }

  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int *hulls_indxs = reinterpret_cast<int *>(task_data->outputs[0]);
    int *pixels_indxs = reinterpret_cast<int *>(task_data->outputs[1]);
    PackHulls(hullsOut_, imageIn_.width, imageIn_.height, hulls_indxs, pixels_indxs);
    task_data->outputs_count[0] = hullsOut_.size();
  }
  return true;
}
