#include "all/burykin_m_radix/include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};

#pragma omp parallel
  {
    std::array<int, 256> local_count = {};

#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < a.size(); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80U;  // Обработка знакового бита
      }
      ++local_count[key];
    }

    // Используем reduction вместо critical section
#pragma omp critical
    {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_count[i];
      }
    }
  }

  return count;
}

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {};
  // Вычисление префиксных сумм
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_all::RadixALL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  // Исправленная версия без гонок данных
  // Сначала собираем элементы для каждого bucket
  std::vector<std::vector<int>> buckets(256);

#pragma omp parallel
  {
    std::vector<std::vector<int>> local_buckets(256);

#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < a.size(); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80U;
      }
      local_buckets[key].push_back(v);
    }

    // Объединяем локальные buckets в глобальные (критическая секция)
#pragma omp critical
    {
      for (int k = 0; k < 256; ++k) {
        if (!local_buckets[k].empty()) {
          buckets[k].insert(buckets[k].end(), local_buckets[k].begin(), local_buckets[k].end());
        }
      }
    }
  }

  // Записываем результат в правильном порядке
  int pos = 0;
  for (int k = 0; k < 256; ++k) {
    for (int val : buckets[k]) {
      b[pos++] = val;
    }
  }
}

std::vector<int> burykin_m_radix_all::RadixALL::PerformKWayMerge(const std::vector<int>& all_data,
                                                                 const std::vector<int>& sizes,
                                                                 const std::vector<int>& displs) {
  std::vector<int> result;
  if (all_data.empty()) return result;

  result.reserve(all_data.size());

  // Указатели на текущие позиции в каждом массиве
  std::vector<int> indices(sizes.size(), 0);

  using Element = std::pair<int, int>;  // {value, array_index}
  std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;

  // Инициализация очереди первыми элементами каждого массива
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 0) {
      pq.push({all_data[displs[i]], static_cast<int>(i)});
    }
  }

  // K-way merge
  while (!pq.empty()) {
    auto current = pq.top();
    pq.pop();

    int value = current.first;
    int array_idx = current.second;

    result.push_back(value);
    indices[array_idx]++;

    // Добавляем следующий элемент из того же массива
    if (indices[array_idx] < sizes[array_idx]) {
      int next_pos = displs[array_idx] + indices[array_idx];
      pq.push({all_data[next_pos], array_idx});
    }
  }

  return result;
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  original_size_ = static_cast<int>(input_size);

  // Only rank 0 has the original data
  std::vector<int> original_data;
  if (world_.rank() == 0 && input_size > 0) {
    original_data.assign(in_ptr, in_ptr + input_size);
  }

  // Single process optimization
  if (world_.size() == 1) {
    input_ = std::move(original_data);
    output_.resize(input_.size());
    return true;
  }

  // Broadcast size to all processes
  boost::mpi::broadcast(world_, original_size_, 0);

  if (original_size_ == 0) {
    input_.clear();
    output_.clear();
    return true;
  }

  // Calculate distribution
  int base_size = original_size_ / world_.size();
  int remainder = original_size_ % world_.size();

  std::vector<int> send_counts(world_.size());
  std::vector<int> displs(world_.size());

  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  int local_size = send_counts[world_.rank()];
  input_.resize(local_size);
  output_.resize(local_size);

  // Scatter data using boost::mpi
  if (local_size > 0) {
    boost::mpi::scatterv(world_, world_.rank() == 0 ? original_data.data() : static_cast<int*>(nullptr), send_counts,
                         input_.data(), 0);
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  // Handle empty input
  if (original_size_ == 0) {
    output_.clear();
    return true;
  }

  // Single process - just radix sort locally
  if (world_.size() == 1) {
    std::vector<int> a = input_;
    std::vector<int> b(a.size());

    // Radix sort для каждого байта (0, 8, 16, 24 бита)
    for (int shift = 0; shift < 32; shift += 8) {
      auto count = ComputeFrequency(a, shift);
      auto index = ComputeIndices(count);
      DistributeElements(a, b, index, shift);  // передаем по значению
      std::swap(a, b);
    }

    output_ = a;
    return true;
  }

  // Multiple processes: каждый процесс делает локальную радиксную сортировку
  std::vector<int> a = input_;
  std::vector<int> b(a.size());

  // Локальная радиксная сортировка на каждом процессе
  for (int shift = 0; shift < 32; shift += 8) {
    if (a.empty()) break;

    auto count = ComputeFrequency(a, shift);
    auto index = ComputeIndices(count);
    DistributeElements(a, b, index, shift);  // передаем по значению
    std::swap(a, b);
  }

  // Теперь у каждого процесса есть локально отсортированный массив
  output_ = a;

  // Безопасное слияние с использованием коллективных операций MPI
  std::vector<int> all_sizes;
  int local_size = static_cast<int>(output_.size());

  // Собираем размеры от всех процессов
  boost::mpi::gather(world_, local_size, all_sizes, 0);

  if (world_.rank() == 0) {
    // Подготавливаем буферы для получения данных
    std::vector<int> displs(world_.size(), 0);
    int total_size = 0;

    for (int i = 0; i < world_.size(); ++i) {
      if (i > 0) {
        displs[i] = displs[i - 1] + all_sizes[i - 1];
      }
      total_size += all_sizes[i];
    }

    std::vector<int> all_data(total_size);

    // Используем gatherv для безопасного сбора данных
    boost::mpi::gatherv(world_, output_.data(), output_.size(), all_data.data(), all_sizes, 0);

    // K-way merge отсортированных кусков
    if (total_size > 0) {
      output_ = PerformKWayMerge(all_data, all_sizes, displs);
    } else {
      output_.clear();
    }

  } else {
    // Отправляем данные процессу 0
    boost::mpi::gatherv(world_, output_.data(), output_.size(), 0);
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  // Only root process writes output
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    const auto output_size = static_cast<int>(output_.size());

    // Безопасное копирование с проверкой размеров
    if (output_size > 0 && output_ptr != nullptr) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < output_size; ++i) {
        output_ptr[i] = output_[i];
      }
    }
  }

  return true;
}