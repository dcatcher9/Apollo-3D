/**
 * @file src/platform/windows/process_environment.h
 * @brief Windows process environment block construction.
 */
#pragma once

#include <boost/process/v1/environment.hpp>
#include <string>

namespace platf {
  // Retain the environment's order and a final block NUL inside the mutable string.
  std::wstring create_environment_block(boost::process::v1::environment &env);
}  // namespace platf
