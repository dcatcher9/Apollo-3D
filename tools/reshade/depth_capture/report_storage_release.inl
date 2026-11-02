// SPDX-License-Identifier: GPL-3.0-only
// Verbatim body fragment (913fef6f:2096-2117), included inside
// release_storage_job; its local throttle, release and timing keep their scope.
      LARGE_INTEGER start{}, end{}, frequency{};
      QueryPerformanceCounter(&start);
      std::uint64_t bytes = 0;
      const auto count = static_cast<unsigned>(job.values.size());
      for (const auto &value : job.values) if (value.use_count() == 1) bytes += value->allocation_bytes;
      // During the add-on's detach, devices may be gone: nothing is released
      // then (the process is ending or the add-on is unloading for good).
      if (sunshine_addon_lifetime::stopping()) (void) new (std::nothrow) storage_list(std::move(job.values));
      job.values.clear();
      QueryPerformanceCounter(&end);
      QueryPerformanceFrequency(&frequency);
      const double ms = double(end.QuadPart - start.QuadPart) * 1000.0 / double(frequency.QuadPart);
      static std::atomic<std::uint64_t> next_line{};
      const auto now = sunshine_game3d::clock::game_now();
      auto due = next_line.load(std::memory_order_relaxed);
      if (pooled && ms >= 8.0 && now >= due && next_line.compare_exchange_strong(due, now + 1000)) {
        const sunshine_game3d::vram::latest_suffix memory;
        char text[256];
        std::snprintf(text, sizeof(text), "Sunshine Game 3D: depth capture storage release took %.1f ms off the present thread "
          "(%u textures, %.1f MiB)%s", ms, count, double(bytes) / 1048576.0, memory.text);
        sunshine_log::message(reshade::log::level::info, text);
      }
