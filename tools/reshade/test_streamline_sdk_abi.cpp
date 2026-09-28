// SPDX-License-Identifier: GPL-3.0-only
// Opt-in, compile against an externally supplied official SDK (no DLL loading).
// Example: g++ -std=c++17 -I<Streamline/include> -DSUNSHINE_SL_MAJOR=2
//   -DSUNSHINE_SL_MINOR=14 -DSUNSHINE_SL_PATCH=1 test_streamline_sdk_abi.cpp -o abi.exe
// Historical GCC-incompatible SDK headers need the documented compile-only
// portability copies; define SUNSHINE_SL_DECLARE_VK_RESULT for their missing type.
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdio>
#include <type_traits>
#include "streamline_camera_data.h"

#if !defined(SUNSHINE_SL_MAJOR) || !defined(SUNSHINE_SL_MINOR) || !defined(SUNSHINE_SL_PATCH)
#error Define the version of the external official SDK headers.
#endif
#ifdef SUNSHINE_SL_DECLARE_VK_RESULT
enum VkResult : int { VkResultHeaderCompileOnly = 0 };
#endif
#include <sl.h>
#if SUNSHINE_SL_MAJOR == 2
#include <sl_dlss_g.h>
#if SUNSHINE_SL_MINOR >= 4
#include <sl_pcl.h>
#endif
#endif

namespace observed = sunshine_streamline;
#if SUNSHINE_SL_MAJOR == 1
using constants = observed::abi_v1::constants;
#if SUNSHINE_SL_MINOR == 0 && SUNSHINE_SL_PATCH < 3
using resource = observed::abi_v1::resource_without_state;
#else
using resource = observed::abi_v1::resource;
#endif
namespace api = observed::abi_v1;
#elif SUNSHINE_SL_MAJOR == 2
using constants = observed::abi_v2::constants;
using resource = observed::abi_v2::resource;
namespace api = observed::abi_v2;
#else
#error This observer has no adapter for this SDK major.
#endif

#define FIELD(ours, our_field, sdk, sdk_field) \
  static_assert(offsetof(ours, our_field) == offsetof(sdk, sdk_field), #ours "." #our_field " offset drift"); \
  static_assert(sizeof(ours::our_field) == sizeof(sdk::sdk_field), #ours "." #our_field " size drift")
#define COMMON(our_field, sdk_field) \
  static_assert(offsetof(constants, common) + offsetof(observed::common_constants, our_field) == \
    offsetof(sl::Constants, sdk_field), "Constants." #sdk_field " offset drift"); \
  static_assert(sizeof(observed::common_constants::our_field) == sizeof(sl::Constants::sdk_field), \
    "Constants." #sdk_field " size drift")

static_assert(sizeof(void *) == 8);
static_assert(sizeof(constants) == sizeof(sl::Constants));
static_assert(sizeof(resource) == sizeof(sl::Resource));
static_assert(sizeof(observed::extent) == sizeof(sl::Extent));
FIELD(observed::extent, left, sl::Extent, left);
FIELD(observed::extent, top, sl::Extent, top);
FIELD(observed::extent, width, sl::Extent, width);
FIELD(observed::extent, height, sl::Extent, height);
COMMON(camera_view_to_clip, cameraViewToClip);
COMMON(clip_to_camera_view, clipToCameraView);
COMMON(clip_to_lens_clip, clipToLensClip);
COMMON(clip_to_prev_clip, clipToPrevClip);
COMMON(prev_clip_to_clip, prevClipToClip);
COMMON(jitter_offset, jitterOffset);
COMMON(motion_vector_scale, mvecScale);
COMMON(camera_pinhole_offset, cameraPinholeOffset);
COMMON(camera_position, cameraPos);
COMMON(camera_up, cameraUp);
COMMON(camera_right, cameraRight);
COMMON(camera_forward, cameraFwd);
COMMON(camera_near, cameraNear);
COMMON(camera_far, cameraFar);
COMMON(camera_fov, cameraFOV);
COMMON(camera_aspect, cameraAspectRatio);
COMMON(invalid_motion_vector, motionVectorsInvalidValue);
FIELD(constants, depth_inverted, sl::Constants, depthInverted);
FIELD(constants, camera_motion_included, sl::Constants, cameraMotionIncluded);
FIELD(constants, motion_vectors_3d, sl::Constants, motionVectors3D);
FIELD(constants, reset, sl::Constants, reset);
FIELD(constants, orthographic_projection, sl::Constants, orthographicProjection);
FIELD(constants, motion_vectors_dilated, sl::Constants, motionVectorsDilated);
FIELD(constants, motion_vectors_jittered, sl::Constants, motionVectorsJittered);
FIELD(resource, type, sl::Resource, type);
FIELD(resource, native, sl::Resource, native);
FIELD(resource, memory, sl::Resource, memory);
FIELD(resource, view, sl::Resource, view);
static_assert(static_cast<unsigned>(sl::Boolean::eFalse) == 0 &&
  static_cast<unsigned>(sl::Boolean::eTrue) == 1 && static_cast<unsigned>(sl::Boolean::eInvalid) == 2);

#if SUNSHINE_SL_MAJOR == 1
static_assert(sl::eResourceTypeTex2d == 0 && sl::eResourceTypeBuffer == 1);
static_assert(sl::eFeatureDLSS == 0 && sl::eBufferTypeDepth == 0 && sl::eBufferTypeMVec == 1 &&
  sl::eBufferTypeHUDLessColor == 2);
#if SUNSHINE_SL_MINOR == 0 && SUNSHINE_SL_PATCH < 2
static_assert(sl::eBufferTypeDLSSInputColor == 3 && sl::eBufferTypeDLSSOutputColor == 4);
#else
static_assert(sl::eBufferTypeScalingInputColor == 3 && sl::eBufferTypeScalingOutputColor == 4);
#endif
#if SUNSHINE_SL_MINOR >= 1
static_assert(sl::eFeatureReflex == 3);
#endif
FIELD(constants, not_rendering_game_frames, sl::Constants, notRenderingGameFrames);
FIELD(constants, ext, sl::Constants, ext);
FIELD(resource, ext, sl::Resource, ext);
#if SUNSHINE_SL_MINOR != 0 || SUNSHINE_SL_PATCH >= 3
FIELD(resource, state, sl::Resource, state);
#endif
#else
static_assert(static_cast<unsigned>(sl::ResourceType::eTex2d) == 0 &&
  static_cast<unsigned>(sl::ResourceType::eBuffer) == 1);
static_assert(sl::kBufferTypeDepth == 0 && sl::kBufferTypeMotionVectors == 1 && sl::kBufferTypeHUDLessColor == 2 &&
  sl::kBufferTypeScalingInputColor == 3 && sl::kBufferTypeScalingOutputColor == 4 && sl::kBufferTypeUIColorAndAlpha == 23);
static_assert(sl::ResourceLifecycle::eOnlyValidNow == 0 && sl::ResourceLifecycle::eValidUntilPresent == 1 &&
  sl::ResourceLifecycle::eValidUntilEvaluate == 2);
static_assert(static_cast<unsigned>(sl::Result::eOk) == 0 && sl::kFeatureDLSS_G == 1000);
static_assert(static_cast<unsigned>(sl::DLSSGMode::eOff) == 0 && static_cast<unsigned>(sl::DLSSGMode::eOn) == 1);
#if SUNSHINE_SL_MINOR >= 2
static_assert(sl::kFeatureDLSS == 0);
#endif
#if SUNSHINE_SL_MINOR > 2 || (SUNSHINE_SL_MINOR == 2 && SUNSHINE_SL_PATCH >= 1)
static_assert(sl::kBufferTypeHiResDepth == 48 && sl::kBufferTypeLinearDepth == 49);
static_assert(static_cast<unsigned>(sl::DLSSGMode::eAuto) == 2);
#endif
#if SUNSHINE_SL_MINOR >= 4
static_assert(sl::kBufferTypeBackbuffer == 53 && sl::kFeaturePCL == api::feature_pcl);
static_assert(static_cast<unsigned>(sl::PCLMarker::ePresentStart) == 4 &&
  static_cast<unsigned>(sl::PCLMarker::ePresentEnd) == 5);
#endif
#if SUNSHINE_SL_MINOR >= 7
// eUnknown means an IUnknown-compatible resource, not a texture or buffer.
// Production still requires the native resource/device/description QI gates.
static_assert(static_cast<unsigned>(sl::ResourceType::eUnknown) == 8);
#endif
#if SUNSHINE_SL_MINOR >= 11
static_assert(static_cast<unsigned>(sl::DLSSGMode::eDynamic) == 3);
#endif
#if SUNSHINE_SL_MINOR >= 12
static_assert(sl::kBufferTypeUIAlpha == 69);
#elif SUNSHINE_SL_MINOR == 11
// UIAlpha has a different public tag value in the 2.11 SDK.
static_assert(sl::kBufferTypeUIAlpha == 68);
#endif
static_assert(sizeof(observed::base_structure) == sizeof(sl::BaseStructure));
FIELD(observed::base_structure, next, sl::BaseStructure, next);
FIELD(observed::base_structure, type, sl::BaseStructure, structType);
FIELD(observed::base_structure, version, sl::BaseStructure, structVersion);
FIELD(resource, state, sl::Resource, state);
FIELD(resource, width, sl::Resource, width);
FIELD(resource, height, sl::Resource, height);
FIELD(resource, native_format, sl::Resource, nativeFormat);
FIELD(resource, mip_levels, sl::Resource, mipLevels);
FIELD(resource, array_layers, sl::Resource, arrayLayers);
FIELD(resource, gpu_virtual_address, sl::Resource, gpuVirtualAddress);
FIELD(resource, flags, sl::Resource, flags);
FIELD(resource, usage, sl::Resource, usage);
static_assert(sizeof(api::resource_tag) == sizeof(sl::ResourceTag));
FIELD(api::resource_tag, resource_ptr, sl::ResourceTag, resource);
FIELD(api::resource_tag, type, sl::ResourceTag, type);
FIELD(api::resource_tag, lifecycle, sl::ResourceTag, lifecycle);
FIELD(api::resource_tag, area, sl::ResourceTag, extent);
static_assert(sizeof(api::viewport) == sizeof(sl::ViewportHandle));
#if SUNSHINE_SL_MINOR > 2 || (SUNSHINE_SL_MINOR == 2 && SUNSHINE_SL_PATCH >= 1)
FIELD(constants, min_relative_linear_depth_object_separation, sl::Constants, minRelativeLinearDepthObjectSeparation);
#endif
FIELD(api::fg_options_prefix, mode, sl::DLSSGOptions, mode);
FIELD(api::fg_options_prefix, generated_frames, sl::DLSSGOptions, numFramesToGenerate);
#endif

// Compare the actual adapter aliases with official declarations, including
// argument ordering, pointer/reference indirection and constness. SDK enums
// and observer integer storage use their common Windows x64 scalar width.
template<std::size_t Bytes> struct integer_wire {};
template<class T, bool Scalar = std::is_integral_v<T> || std::is_enum_v<T>>
struct scalar_wire { using type = T; };
template<class T> struct scalar_wire<T, true> { using type = integer_wire<sizeof(T)>; };
template<class T> struct signature : scalar_wire<T> {};
template<class T> using wire = typename signature<T>::type;
template<class T> struct signature<const T> { using type = const wire<T>; };
template<class T> struct signature<T *> { using type = wire<T> *; };
template<class T> struct signature<T &> { using type = wire<T> &; };
template<class R, class... Args> struct signature<R (*)(Args...)> {
  using type = wire<R> (*)(wire<Args>...);
};
template<> struct signature<sl::Constants> { using type = constants; };
// The early pointer shares the call ABI; its different allocation is checked above.
template<> struct signature<sl::Resource> { using type = api::resource; };
template<> struct signature<sl::Extent> { using type = observed::extent; };
#if SUNSHINE_SL_MAJOR == 2
template<> struct signature<sl::BaseStructure> { using type = observed::base_structure; };
template<> struct signature<sl::FrameToken> { using type = api::frame_token; };
template<> struct signature<sl::ViewportHandle> { using type = api::viewport; };
template<> struct signature<sl::ResourceTag> { using type = api::resource_tag; };
template<> struct signature<sl::DLSSGOptions> { using type = api::fg_options_prefix; };
#endif
#define FUNCTION(ours, sdk) static_assert(std::is_same_v<wire<ours>, wire<decltype(&sdk)>>, #sdk " signature drift")
#if SUNSHINE_SL_MAJOR == 1 && SUNSHINE_SL_MINOR == 0 && SUNSHINE_SL_PATCH < 2
FUNCTION(api::set_constants, sl::setConstants);
FUNCTION(api::set_tag, sl::setTag);
FUNCTION(api::evaluate_feature, sl::evaluateFeature);
#else
FUNCTION(api::set_constants, slSetConstants);
FUNCTION(api::set_tag, slSetTag);
FUNCTION(api::evaluate_feature, slEvaluateFeature);
#endif
#if SUNSHINE_SL_MAJOR == 2
FUNCTION(api::get_new_frame_token, slGetNewFrameToken);
FUNCTION(api::get_feature_function, slGetFeatureFunction);
FUNCTION(api::fg_set_options, slDLSSGSetOptions);
#if SUNSHINE_SL_MINOR >= 4
FUNCTION(api::pcl_set_marker, slPCLSetMarker);
#endif
#if SUNSHINE_SL_MINOR > 7 || (SUNSHINE_SL_MINOR == 7 && SUNSHINE_SL_PATCH >= 30)
FUNCTION(api::set_tag_for_frame, slSetTagForFrame);
#endif
#endif

int main() {
#if SUNSHINE_SL_MAJOR == 2
  const auto compatible_base = [](const sl::BaseStructure &sdk, const observed::guid &type,
      std::uint64_t minimum_version, std::uint64_t maximum_version) {
    observed::base_structure copy{};
    std::memcpy(&copy, &sdk, sizeof(copy));
    return copy.next == nullptr && copy.version >= minimum_version && copy.version <= maximum_version &&
      std::memcmp(&copy.type, &type, sizeof(type)) == 0;
  };
  const sl::Constants sdk_constants{};
  sl::Resource sdk_resource{sl::ResourceType::eTex2d, nullptr};
  const sl::ResourceTag sdk_tag{&sdk_resource, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent};
  if (!compatible_base(sdk_constants, observed::constants_guid, 1, 2) ||
      !compatible_base(sdk_resource, observed::resource_guid, 1, 1) ||
      !compatible_base(sdk_tag, observed::tag_guid, 1, 1)) {
    std::fputs("FAIL official typed GUID/version disagrees with production admission\n", stderr);
    return 1;
  }
  // ViewportHandle::value is private. Construct its public value and inspect
  // our actual wire offset instead of altering the official access controls.
  const sl::ViewportHandle sdk_viewport{0x12345678u};
  api::viewport copied_viewport{};
  std::memcpy(&copied_viewport, &sdk_viewport, sizeof(copied_viewport));
  if (copied_viewport.value != 0x12345678u || !compatible_base(sdk_viewport, observed::viewport_guid, 1, 1)) {
    std::fputs("FAIL public ViewportHandle constructor disagrees with production wire layout\n", stderr);
    return 1;
  }
#endif
  std::printf("PASS SDK %u.%u.%u: production constants=%zu resource=%zu; fields and hook signatures match\n",
    SUNSHINE_SL_MAJOR, SUNSHINE_SL_MINOR, SUNSHINE_SL_PATCH, sizeof(constants), sizeof(resource));
  return 0;
}
