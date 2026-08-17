/**
 * @file src/platform/windows/nvprefs/driver_settings.h
 * @brief Declarations for nvidia driver settings.
 */
#pragma once

// nvapi headers
// disable clang-format header reordering
// as <NvApiDriverSettings.h> needs types from <nvapi.h>
// clang-format off
#if defined(__MINGW32__) && !defined(__NVAPI_EMPTY_SAL)
// R560's split lite headers nest their SAL start/end shims. Defer their
// cleanup so an inner header cannot remove annotations still used by nvapi.h.
#define SUNSHINE_NVAPI_END_EMPTY_SAL
#define __NVAPI_EMPTY_SAL
#endif
#include <nvapi.h>
#include <NvApiDriverSettings.h>
#ifdef SUNSHINE_NVAPI_END_EMPTY_SAL
#undef __NVAPI_EMPTY_SAL
#include <nvapi_lite_salend.h>
#undef SUNSHINE_NVAPI_END_EMPTY_SAL
#endif
// clang-format on

// local includes
#include "undo_data.h"

namespace nvprefs {

  class driver_settings_t {
  public:
    ~driver_settings_t();

    bool init();

    void destroy();

    bool load_settings();

    bool save_settings();

    bool restore_global_profile_to_undo(const undo_data_t &undo_data);

    bool check_and_modify_global_profile(std::optional<undo_data_t> &undo_data);

    bool check_and_modify_application_profile(bool &modified);

  private:
    NvDRSSessionHandle session_handle = nullptr;
  };

}  // namespace nvprefs
