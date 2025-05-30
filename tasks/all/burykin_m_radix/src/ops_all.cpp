#include "../include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

namespace {
template <typename T>
constexpr size_t Bytes() {
  return sizeof(T);
}

template <typename T>
constexpr size_t Bits() {
  return Bytes<T>() * CHAR_BIT;
}

class Bitutil {
 public:
  static constexpr uint32_t AsU32(int x) {
    const uint32_t ux = static_cast<uint32_t>(x);
    return ux ^ (1U << 31);
  }

  template <typename T>
    requires std::is_integral_v<T>
  static constexpr uint8_t ByteAt(const T &val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

void RadixSort(std::span<int> v) {
  if (v.empty()) return;

  constexpr size_t kBase = 1 << CHAR_BIT;

  std::vector<int> aux_buf(v.size());
  std::span<int> aux{aux_buf};

  std::array<std::size_t, kBase> count;

  // Указатели на текущий и вспомогательный массивы
  std::span<int> current = v;
  std::span<int> temp = aux;

  for (std::size_t ib = 0; ib < Bytes<int>(); ++ib) {
    std::ranges::fill(count, 0);

    // Подсчет элементов для текущего байта
    std::ranges::for_each(current, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]; });

    // Преобразование в префиксные суммы
    std::partial_sum(count.begin(), count.end(), count.begin());

    // Распределение элементов во временный массив (в обратном порядке для стабильности)
    std::ranges::for_each(std::ranges::reverse_view(current),
                          [&](auto el) { temp[--count[Bitutil::ByteAt(Bitutil::AsU32(el), ib)]] = el; });

    // Меняем местами указатели
    std::swap(current, temp);
  }

  // Если результат находится не в исходном массиве, копируем его обратно
  if (current.data() != v.data()) {
    std::ranges::copy(current, v.begin());
  }
}

std::vector<std::span<int>> Distribute(std::span<int> arr, std::size_t n) {
  std::vector<std::span<int>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto *cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}
}  // namespace

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    std::span<int> src = {reinterpret_cast<int *>(task_data->inputs[0]), task_data->inputs_count[0]};
    input_.assign(src.begin(), src.end());
    output_.reserve(input_.size());
  }
  return true;
}

void burykin_m_radix_all::RadixALL::Squash(boost::mpi::communicator &group) {
  const auto numprocs = static_cast<std::size_t>(group.size());

  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (group.rank() % (2 * i) == 0) {
      const int slave = group.rank() + static_cast<int>(i);
      if (slave < static_cast<int>(numprocs)) {
        int size{};
        group.recv(int(slave), 0, size);

        if (size > 0) {
          const std::size_t threshold = procchunk_.size();
          procchunk_.resize(threshold + size);
          group.recv(int(slave), 0, procchunk_.data() + threshold, size);

          // Слияние отсортированных частей
          std::ranges::inplace_merge(procchunk_, procchunk_.begin() + std::int64_t(threshold));
        }
      }
    } else if ((group.rank() % i) == 0) {
      const int size = static_cast<int>(procchunk_.size());
      const int master = group.rank() - static_cast<int>(i);
      group.send(master, 0, size);
      if (size > 0) {
        group.send(master, 0, procchunk_.data(), size);
      }
      break;
    }
  }
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  std::size_t totalsize{};
  if (world_.rank() == 0) {
    totalsize = input_.size();
  }
  boost::mpi::broadcast(world_, totalsize, 0);

  if (totalsize == 0) {
    return true;
  }

  const auto numprocs = std::min<std::size_t>(totalsize, world_.size());

  if (world_.rank() >= int(numprocs)) {
    world_.split(1);
    return true;
  }

  auto group = world_.split(0);

  // Распределение данных между процессами
  if (group.rank() == 0) {
    std::vector<std::span<int>> procchunks = Distribute(input_, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());

    for (int i = 1; i < int(procchunks.size()); i++) {
      const auto &chunk = procchunks[i];
      const int chunksize = int(chunk.size());
      group.send(i, 0, chunksize);
      if (chunksize > 0) {
        group.send(i, 0, chunk.data(), chunksize);
      }
    }
  } else {
    int chunksize{};
    group.recv(0, 0, chunksize);
    procchunk_.resize(chunksize);
    if (chunksize > 0) {
      group.recv(0, 0, procchunk_.data(), chunksize);
    }
  }

  // Локальная сортировка с использованием OpenMP
  if (!procchunk_.empty()) {
    const auto numthreads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
    std::vector<std::span<int>> chunks = Distribute(procchunk_, numthreads);

    // Параллельная сортировка каждого chunk'а
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numthreads); i++) {
      RadixSort(chunks[i]);
    }

    // Слияние отсортированных chunk'ов
    for (std::size_t step = 1; step < numthreads; step *= 2) {
      const auto multithreaded = chunks.front().size() > 48;

#pragma omp parallel for if (multithreaded)
      for (int j = 0; j < static_cast<int>(numthreads); j += 2 * static_cast<int>(step)) {
        const auto right_idx = j + static_cast<int>(step);
        if (static_cast<std::size_t>(right_idx) < chunks.size()) {
          auto &left = chunks[j];
          auto &right = chunks[right_idx];

          // Создаем временный буфер для слияния
          std::vector<int> merged;
          merged.reserve(left.size() + right.size());

          // Сливаем два отсортированных диапазона
          std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(merged));

          // Копируем результат обратно в левый chunk
          std::copy(merged.begin(), merged.end(), left.begin());

          // Обновляем размер левого chunk'а
          left = std::span{left.begin(), left.begin() + merged.size()};
        }
      }
    }

    // Копируем окончательно отсортированный результат
    if (!chunks.empty()) {
      procchunk_.assign(chunks[0].begin(), chunks[0].end());
    }
  }

  // Сбор результатов от всех процессов
  Squash(group);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(procchunk_, reinterpret_cast<int *>(task_data->outputs[0]));
  }
  return true;
}