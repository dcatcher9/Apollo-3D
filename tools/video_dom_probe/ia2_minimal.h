/**
 * Minimal IAccessible2 ABI declarations used by the video DOM helper.
 *
 * The method order and signatures below are derived from Accessible2.idl in the
 * Linux Foundation IAccessible2 1.3 specification. Only IAccessible2 itself is
 * declared because the probe reads get_uniqueID() and get_attributes() only.
 *
 * Copyright (c) 2007, 2013 Linux Foundation
 * Copyright (c) 2006 IBM Corporation
 * Copyright (c) 2000, 2006 Sun Microsystems, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the Linux Foundation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <oleacc.h>

namespace video_dom_probe::ia2 {
  struct IAccessibleRelation;

  using AccessibleStates = LONG;

  enum IA2ScrollType : int {
    IA2_SCROLL_TYPE_TOP_LEFT,
    IA2_SCROLL_TYPE_BOTTOM_RIGHT,
    IA2_SCROLL_TYPE_TOP_EDGE,
    IA2_SCROLL_TYPE_BOTTOM_EDGE,
    IA2_SCROLL_TYPE_LEFT_EDGE,
    IA2_SCROLL_TYPE_RIGHT_EDGE,
    IA2_SCROLL_TYPE_ANYWHERE,
  };

  enum IA2CoordinateType : int {
    IA2_COORDTYPE_SCREEN_RELATIVE,
    IA2_COORDTYPE_PARENT_RELATIVE,
  };

  struct IA2Locale {
    BSTR language;
    BSTR country;
    BSTR variant;
  };

  struct IAccessible2: IAccessible {
    virtual HRESULT STDMETHODCALLTYPE get_nRelations(LONG *relation_count) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_relation(
      LONG relation_index,
      IAccessibleRelation **relation
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_relations(
      LONG maximum_relations,
      IAccessibleRelation **relations,
      LONG *relation_count
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE role(LONG *role) = 0;
    virtual HRESULT STDMETHODCALLTYPE scrollTo(IA2ScrollType scroll_type) = 0;
    virtual HRESULT STDMETHODCALLTYPE scrollToPoint(
      IA2CoordinateType coordinate_type,
      LONG x,
      LONG y
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_groupPosition(
      LONG *group_level,
      LONG *similar_items_in_group,
      LONG *position_in_group
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_states(AccessibleStates *states) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_extendedRole(BSTR *extended_role) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_localizedExtendedRole(BSTR *localized_extended_role) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_nExtendedStates(LONG *extended_state_count) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_extendedStates(
      LONG maximum_extended_states,
      BSTR **extended_states,
      LONG *extended_state_count
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_localizedExtendedStates(
      LONG maximum_localized_extended_states,
      BSTR **localized_extended_states,
      LONG *localized_extended_state_count
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_uniqueID(LONG *unique_id) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_windowHandle(HWND *window_handle) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_indexInParent(LONG *index_in_parent) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_locale(IA2Locale *locale) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_attributes(BSTR *attributes) = 0;
  };

  inline constexpr IID IID_IAccessible2 {
    0xE89F726E,
    0xC4F4,
    0x4C19,
    {0xBB, 0x19, 0xB6, 0x47, 0xD7, 0xFA, 0x84, 0x78},
  };
}  // namespace video_dom_probe::ia2
