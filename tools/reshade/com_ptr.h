// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The add-on's one owner of a COM reference (IUnknown::AddRef/Release). It
// holds at most one reference and releases it when reset, replaced or
// destroyed; it moves but never copies. p is the held pointer, readable
// directly; it changes only through put(), reset(), retain() and release().

#include <utility>

namespace sunshine_game3d {
  template<class T> class com_ptr {
  public:
    com_ptr() = default;
    ~com_ptr() { reset(); }
    com_ptr(const com_ptr &) = delete;
    com_ptr &operator=(const com_ptr &) = delete;
    com_ptr(com_ptr &&other) noexcept: p(other.release()) {}
    com_ptr &operator=(com_ptr &&other) noexcept {
      if (this != &other) reset(other.release());
      return *this;
    }

    T *get() const { return p; }
    T *operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
    // Releases the held reference and returns the slot for an out parameter.
    T **put() {
      reset();
      return &p;
    }
    // Gives up the held reference without releasing it.
    T *release() { return std::exchange(p, nullptr); }
    // Takes over owned (a reference already counted) and releases the old one.
    void reset(T *owned = nullptr) {
      if (auto *old = std::exchange(p, owned)) old->Release();
    }
    // Holds a new reference to value.
    void retain(T *value) {
      if (value) value->AddRef();
      reset(value);
    }

    T *p = nullptr;
  };
}
