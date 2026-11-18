/**
 * @file src/nvenc/nvenc_d3d11_native.h
 * @brief Declarations for native Direct3D11 NVENC encoder.
 */
#pragma once
#ifdef _WIN32
  // local includes
  #include "nvenc_d3d11.h"

  #include <optional>

namespace nvenc {

  /**
   * @brief Native Direct3D11 NVENC encoder.
   */
  class nvenc_d3d11_native final: public nvenc_d3d11 {
  public:
    /**
     * @param d3d_device Direct3D11 device used for encoding.
     * @param allow_p010_chroma_store Request unordered access for P010 inputs when available.
     */
    explicit nvenc_d3d11_native(ID3D11Device *d3d_device, bool allow_p010_chroma_store = false);
    ~nvenc_d3d11_native();

    ID3D11Texture2D *get_input_texture() override;

    /** Registration belongs to these surfaces, independently of the output picture slots. */
    unsigned input_surface_count() const noexcept;
    ID3D11Texture2D *get_input_texture(unsigned surface);

    /** Submitting thread: select an unmapped output, then publish it after conversion writes. */
    std::optional<unsigned> begin_input_conversion() noexcept;
    void set_converted_input(unsigned surface) noexcept;
    bool input_waits_for_retirement() const noexcept override;

  #ifdef SUNSHINE_TESTS
    std::optional<unsigned> input_surface_for_picture_for_test(unsigned slot) const noexcept;
  #endif

  private:
    bool create_and_register_input_buffer() override;
    NV_ENC_REGISTERED_PTR prepare_input(unsigned slot) override;

    const ID3D11DevicePtr d3d_device;
    const bool allow_p010_chroma_store;
    // With pictures in flight, one spare keeps the next conversion independent of retrieval.
    std::array<ID3D11Texture2DPtr, max_pipeline_depth + 1> input_surfaces;
    unsigned converted_input = 0;
  };

}  // namespace nvenc
#endif
