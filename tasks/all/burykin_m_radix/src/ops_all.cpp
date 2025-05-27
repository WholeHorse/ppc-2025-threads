#include "all/burykin_m_radix/include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

namespace {
std::vector<std::span<int>> Distribute(std::span<int> arr, std::size_t n) {
  std::vector<std::span<int>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto* cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}

// Функция для получения ключа с правильной обработкой знака
inline unsigned int GetRadixKey(int value, int shift) {
  unsigned int uval = static_cast<unsigned int>(value);
  // Для старшего разряда инвертируем знаковый бит
  if (shift == 24) {
    uval ^= 0x80000000U;
  }
  return (uval >> shift) & 0xFFU;
}
}  // namespace

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};

#pragma omp parallel default(none) shared(a, count, shift)
  {
    // Каждый поток ведет свой локальный счетчик
    std::array<int, 256> local_count = {};

#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
      const unsigned int key = GetRadixKey(a[i], shift);
      ++local_count[key];
    }

    // Объединяем локальные счетчики в общий
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
  std::array<int, 256> index = {0};
  // Последовательное вычисление префиксных сумм
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_all::RadixALL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       const std::array<int, 256>& base_index, const int shift) {
  // Создаем атомарные счетчики для каждого bucket'а
  std::array<std::atomic<int>, 256> atomic_indices;
  for (int i = 0; i < 256; ++i) {
    atomic_indices[i].store(base_index[i], std::memory_order_relaxed);
  }

#pragma omp parallel for default(none) shared(a, b, atomic_indices, shift)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    const int value = a[i];
    const unsigned int key = GetRadixKey(value, shift);

    // Атомарно получаем и увеличиваем позицию
    const int pos = atomic_indices[key].fetch_add(1, std::memory_order_relaxed);
    b[pos] = value;
  }
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  // Для одного процесса или главного процесса проверяем размеры
  if (world_.size() == 1 || world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }
  return true;
}

void burykin_m_radix_all::RadixALL::Squash(boost::mpi::communicator& group) {
  if (group.rank() == 0) {
    // Ранг 0 принимает данные от всех остальных процессов
    for (int i = 1; i < group.size(); ++i) {
      int partner_size = 0;
      group.recv(i, 0, partner_size);  // Получаем размер данных от процесса i
      if (partner_size > 0) {
        std::vector<int> partner_data(partner_size);
        group.recv(i, 0, partner_data.data(), partner_size);  // Получаем данные
        // Сливаем текущий procchunk_ с полученными данными
        std::vector<int> temp;
        std::ranges::merge(procchunk_, partner_data, std::back_inserter(temp));
        procchunk_ = std::move(temp);
      }
    }
  } else {
    // Все остальные процессы отправляют свои данные рангу 0
    const int size = static_cast<int>(procchunk_.size());
    group.send(0, 0, size);  // Отправляем размер
    if (size > 0) {
      group.send(0, 0, procchunk_.data(), size);  // Отправляем данные
    }
  }
}

void burykin_m_radix_all::RadixALL::LocalRadixSort() {
  if (procchunk_.empty()) {
    return;
  }

  std::vector<int> a = std::move(procchunk_);
  std::vector<int> b(a.size());

  // Выполняем поразрядную сортировку по 8 бит за раз
  for (int shift = 0; shift < 32; shift += 8) {
    // 1. Подсчет частот (параллельно)
    auto count = ComputeFrequency(a, shift);

    // 2. Вычисление индексов (последовательно)
    const auto index = ComputeIndices(count);

    // 3. Распределение элементов (параллельно)
    DistributeElements(a, b, index, shift);

    // Меняем местами массивы для следующей итерации
    a.swap(b);
  }

  procchunk_ = std::move(a);
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  // 1. Определяем общий размер данных
  std::size_t totalsize = 0;
  if (world_.rank() == 0) {
    totalsize = input_.size();
  }
  boost::mpi::broadcast(world_, totalsize, 0);

  // Обработка пустого ввода
  if (totalsize == 0) {
    return true;
  }

  // 2. Определяем количество активных процессов
  const auto numprocs = std::min<std::size_t>(totalsize, world_.size());

  // Если процесс не нужен, исключаем его
  if (world_.rank() >= static_cast<int>(numprocs)) {
    world_.split(1);  // Неактивная группа
    return true;
  }

  auto group = world_.split(0);  // Активная группа

  // 3. Распределение данных между процессами (только MPI)
  if (group.rank() == 0) {
    // Главный процесс распределяет данные
    std::span<int> input_span{input_.data(), input_.size()};
    std::vector<std::span<int>> procchunks = Distribute(input_span, numprocs);

    // Сохраняем свою часть
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());

    // Отправляем части другим процессам
    for (std::size_t i = 1; i < procchunks.size(); ++i) {
      const auto& chunk = procchunks[i];
      const int chunksize = static_cast<int>(chunk.size());
      group.send(static_cast<int>(i), 0, chunksize);
      if (chunksize > 0) {
        group.send(static_cast<int>(i), 0, chunk.data(), chunksize);
      }
    }
  } else {
    // Остальные процессы получают свои части
    int chunksize = 0;
    group.recv(0, 0, chunksize);
    if (chunksize > 0) {
      procchunk_.resize(chunksize);
      group.recv(0, 0, procchunk_.data(), chunksize);
    }
  }

  // 4. Локальная сортировка каждым процессом (OpenMP)
  LocalRadixSort();

  // 5. Сбор и объединение результатов (только MPI)
  Squash(group);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    const auto output_size = static_cast<int>(procchunk_.size());

    // Параллельное копирование результатов
#pragma omp parallel for default(none) shared(output_ptr, output_size)
    for (int i = 0; i < output_size; ++i) {
      output_ptr[i] = procchunk_[i];
    }
  }
  return true;
}