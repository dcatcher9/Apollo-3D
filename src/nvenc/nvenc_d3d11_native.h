/**
 * @file src/nvenc/nvenc_d3d11_native.h
 * @brief Declarations for native Direct3D11 NVENC encoder.
 */
#pragma once
#ifdef _WIN32
  // local includes
  #include "nvenc_d3d11.h"

namespace nvenc {

  /**
   * @brief Native Direct3D11 NVENC encoder.
   */
  class nvenc_d3d11_native final: public nvenc_d3d11 {
  public:
    /**
     * @param d3d_device Direct3D11 device used for encoding.
     */
    explicit nvenc_d3d11_native(ID3D11Device *d3d_device);
    ~nvenc_d3d11_native();

    ID3D11Texture2D *get_input_texture() override;

  #ifdef SUNSHINE_TESTS
    /** Commands issued at the actual per-picture CopyResource site; submitting thread only. */
    std::array<std::uint64_t, max_pipeline_depth> input_copy_counts_for_test() const noexcept {
      return input_copy_counts_;
    }
  #endif

  private:
    bool create_and_register_input_buffer() override;
    void prepare_input(unsigned slot, std::uint64_t input_revision) override;

    const ID3D11DevicePtr d3d_device;
    ID3D11Texture2DPtr d3d_input_texture;  ///< Conversion target; registered itself at depth 1.
    std::array<ID3D11Texture2DPtr, max_pipeline_depth> picture_inputs;  ///< Each picture's copy at depth 2.
    // Proof belongs to each actual picture texture and advances only when its copy is issued.
    std::array<std::uint64_t, max_pipeline_depth> picture_input_revisions {};
    ID3D11DeviceContextPtr copy_context;
  #ifdef SUNSHINE_TESTS
    std::array<std::uint64_t, max_pipeline_depth> input_copy_counts_ {};
#endif
  };

}  // namespace nvenc
#endif
