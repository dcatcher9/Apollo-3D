/**
 * @file src/platform/windows/nvprefs/nvprefs_interface.h
 * @brief Declarations for nvidia preferences interface.
 */
#pragma once

// standard includes
#include <memory>

namespace nvprefs {

  class nvprefs_interface {
  public:
    nvprefs_interface();
    ~nvprefs_interface();

    bool load();

    void unload();

    bool restore_from_and_delete_undo_file_if_exists();

    bool modify_application_profile();

    // Whether the application profile leaves variable refresh to the driver's default. Takes
    // effect with the next modify_application_profile().
    void allow_variable_refresh(bool allowed);

    bool modify_global_profile();

    bool owning_undo_file();

    bool restore_global_profile();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace nvprefs
