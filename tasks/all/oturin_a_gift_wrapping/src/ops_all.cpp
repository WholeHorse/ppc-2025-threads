#include "all/oturin_a_gift_wrapping/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <vector>

double oturin_a_gift_wrapping_all::ABTP(Coord a, Coord b, Coord c) {
  Coord ab = {.x = b.x - a.x, .y = b.y - a.y};
  Coord cb = {.x = b.x - c.x, .y = b.y - c.y};
  double dot = ((ab.x * cb.x) + (ab.y * cb.y));
  double cross = ((ab.x * cb.y) - (ab.y * cb.x));
  return fabs(atan2(cross, dot));
}

double oturin_a_gift_wrapping_all::ABTP(Coord a, Coord c) {
  Coord b{.x = a.x, .y = (a.y - 1)};
  return ABTP(b, a, c);
}

double oturin_a_gift_wrapping_all::Distance(Coord a, Coord b) {
  int t1 = a.x - b.x;
  int t2 = a.y - b.y;
  return double((t1 * t1) + (t2 * t2));
}

bool oturin_a_gift_wrapping_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    // Init value for input and output
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<Coord *>(task_data->inputs[0]);
    input_ = std::vector<Coord>(in_ptr, in_ptr + input_size);

    for (int i = 1; i < world_.size(); i++) {
      world_.send(i, 1, input_.data(), (int)input_size);
    }

  } else {
    input_ = std::vector<Coord>(task_data->inputs_count[0]);
    world_.recv(0, 1, input_.data(), (int)input_.size());
  }

  n_ = int(input_.size());
  output_ = std::vector<Coord>();
  output_.reserve(n_);

  // check if all points are same
  auto are_same = [&](const auto &p) { return p == input_[0]; };
  return !std::ranges::all_of(input_.begin(), input_.end(), are_same);
}

bool oturin_a_gift_wrapping_all::TestTaskALL::ValidationImpl() {
  return task_data->inputs_count[0] >= 3;  // task requires 3 or more points to wrap
}

void oturin_a_gift_wrapping_all::TestTaskALL::PointSearcher::operator()(const tbb::blocked_range<int> &r) {
  int begin = r.begin();
  int end = r.end();
  const Coord penultimate_element = p_->output_[p_->output_.size() - 2];
  const Coord last_element = p_->output_.back();

  for (int i = begin; i < end; i++) {
    double t = ABTP(penultimate_element, last_element, p_->input_[i]);
    p_->PointSearch(t, line_angle_, search_index, i);
  }
}

void oturin_a_gift_wrapping_all::TestTaskALL::PointSearcher::join(const PointSearcher &x) {
  if (line_angle_ <= x.line_angle_ &&
      (line_angle_ != x.line_angle_ || Distance(p_->output_.back(), p_->input_[x.search_index]) <
                                           Distance(p_->output_.back(), p_->input_[search_index]))) {
    search_index = x.search_index;
    line_angle_ = x.line_angle_;
  }
}

bool oturin_a_gift_wrapping_all::TestTaskALL::RunImpl() {
  double line_angle = -5;
  int search_index = 0;
  int start_index = 0;

  int world_size = world_.size();

  if (world_size > (int)input_.size()) {
    if (world_.rank() + 1 > (int)input_.size()) {
      return true;
    }
    world_size = (int)input_.size();
  }
  int thread_block = (int)input_.size() / world_size;

  if (!output_.empty()) {
    output_.clear();
  }
  // this .clear() used ONLY for perftest TaskRun. for some reason output_ has something in it

  // find most left point (priority to top)
  start_index = FindMostLeft();
  output_.push_back(input_[start_index]);

  SearchSecondPoint(start_index, search_index);

  // main loop
  do {
    output_.push_back(input_[search_index]);

    PointSearcher ps(this, search_index);
    if (world_.rank() == world_.size() - 1) {
      tbb::parallel_reduce(tbb::blocked_range<int>(world_.rank() * thread_block, n_), ps);
    } else {
      tbb::parallel_reduce(tbb::blocked_range<int>(world_.rank() * thread_block, (world_.rank() + 1) * thread_block),
                           ps);
    }
    search_index = ps.search_index;

    if (world_.rank() == 0) {
      line_angle = ABTP(output_[output_.size() - 2], output_.back(), input_[search_index]);

      for (int i = 1; i < world_size; i++) {
        int temp_index = 0;
        world_.recv(i, 111, &temp_index, 1);

        double t = ABTP(output_[output_.size() - 2], output_.back(), input_[temp_index]);
        PointSearch(t, line_angle, search_index, temp_index);
      }

      for (int i = 1; i < world_size; i++) {
        world_.send(i, 222, &search_index, 1);
      }

    } else {
      world_.send(0, 111, &search_index, 1);
      world_.recv(0, 222, &search_index, 1);
    }

  } while (search_index != start_index);

  return true;
}

bool oturin_a_gift_wrapping_all::TestTaskALL::PostProcessingImpl() {
  auto *result_ptr = reinterpret_cast<Coord *>(task_data->outputs[0]);
  std::ranges::copy(output_.begin(), output_.end(), result_ptr);
  world_.barrier();
  return true;
}

int oturin_a_gift_wrapping_all::TestTaskALL::FindMostLeft() {
  Coord most_left = input_[0];
  int start_index = 0;
  for (int i = 1; i < n_; i++) {
    if (input_[i].x < most_left.x || (input_[i].x == most_left.x && input_[i].y > most_left.y)) {
      start_index = i;
      most_left = input_[i];
    }
  }
  return start_index;
}

void oturin_a_gift_wrapping_all::TestTaskALL::PointSearch(const double t, double &line_angle, int &search_index,
                                                          const int i) {
  if (t < line_angle) {
    return;
  }
  if (output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
    if (t == line_angle && Distance(output_.back(), input_[i]) >= Distance(output_.back(), input_[search_index])) {
      return;
    }
    search_index = i;
    line_angle = t;
  }
}

void oturin_a_gift_wrapping_all::TestTaskALL::SearchSecondPoint(int start_index, int &search_index) {
  double line_angle = -5;
  for (int i = 0; i < n_; i++) {
    if (i == start_index) {
      continue;
    }
    double t = ABTP(input_[start_index], input_[i]);
    if (t > line_angle) {
      line_angle = t;
      search_index = i;
    } else if (t == line_angle) {
      if (Distance(input_[start_index], input_[i]) < Distance(input_[start_index], input_[search_index])) {
        search_index = i;
        line_angle = t;
      }
    }
  }
}