#include <gtest/gtest.h>

#ifdef _WIN32
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <wrl/client.h>

  #include <array>
  #include <cstdint>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <string>
  #include <vector>

  #include <src/cuda_conditional_graph.h>
  #include <src/generated/depth_coordinate_v2_contract.h>
  #include <src/host_sbs_gpu_trace.h>
  #include <src/host_sbs_shader_cache.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace trace = models::host_sbs_gpu_trace;
  namespace v2 = models::depth_coordinate_v2;

  TEST(HostSbsGpuTraceContractTest, HlslAbiAndAuthenticatedSourceMatchNative) {
    const auto shader_root = std::filesystem::path {SUNSHINE_SHADERS_DIR};
    const auto sources = models::host_sbs_shader_cache::snapshot_sources(
      shader_root, models::host_sbs_shader_cache::gpu_trace_specs
    );
    ASSERT_TRUE(sources);
    EXPECT_EQ(
      models::host_sbs_shader_cache::source_closure_sha256(sources),
      models::host_sbs_shader_cache::gpu_trace_source_closure_sha256
    );
    std::ifstream input(shader_root / "host_sbs_gpu_trace_cs.hlsl", std::ios::binary);
    const std::string shader {
      std::istreambuf_iterator<char> {input}, std::istreambuf_iterator<char> {}
    };
    ASSERT_FALSE(shader.empty());
    static_assert(trace::constant_word_count == 20u);
    EXPECT_NE(shader.find("uint trace_observation_timestamp_low;"), std::string::npos);
    EXPECT_NE(shader.find("uint trace_observation_timestamp_high;"), std::string::npos);
    EXPECT_NE(shader.find("uint trace_padding1;"), std::string::npos);
    const std::array definitions {
      "#define GPU_TRACE_RING_SCHEMA 3u",
      "#define GPU_TRACE_RING_TAG 0x48525447u",
      "#define GPU_TRACE_RECORD_TAG 0x31525447u",
      "#define GPU_TRACE_CAPACITY 300u",
      "#define GPU_TRACE_HEADER_WORDS 16u",
      "#define GPU_TRACE_RECORD_WORDS 176u",
      "#define GPU_TRACE_TRANSACTION_WORDS 64u",
      "#define GPU_TRACE_LOCATOR_WORDS 80u",
      "#define GPU_TRACE_CONDITION_WORDS 6u",
      "#define GPU_TRACE_HEADER_NEXT_SEQUENCE_LOW 4u",
      "#define GPU_TRACE_HEADER_NEXT_SEQUENCE_HIGH 5u",
      "#define GPU_TRACE_HEADER_NEXT_SLOT 6u",
      "#define GPU_TRACE_HEADER_COMMITTED_COUNT 7u",
      "#define GPU_TRACE_HEADER_RESERVED_BEGIN 8u",
      "#define GPU_TRACE_RECORD_SEQUENCE_LOW 2u",
      "#define GPU_TRACE_RECORD_SEQUENCE_HIGH 3u",
      "#define GPU_TRACE_RECORD_FRAME_LOW 4u",
      "#define GPU_TRACE_RECORD_ANALYSIS_GENERATION_LOW 6u",
      "#define GPU_TRACE_RECORD_DOMAIN_TAG_LOW 8u",
      "#define GPU_TRACE_RECORD_TRANSACTION_TOKEN_LOW 10u",
      "#define GPU_TRACE_RECORD_SUBMISSION_CLASS 12u",
      "#define GPU_TRACE_RECORD_DEPTH_DISPOSITION 13u",
      "#define GPU_TRACE_RECORD_EXPECTED_WORK 14u",
      "#define GPU_TRACE_RECORD_SUBTITLE_DISPOSITION 15u",
      "#define GPU_TRACE_RECORD_FLAGS 16u",
      "#define GPU_TRACE_RECORD_HOST_SUBTITLE_OUTCOME 17u",
      "#define GPU_TRACE_RECORD_SOURCE_WIDTH 18u",
      "#define GPU_TRACE_RECORD_FIELD_WIDTH 20u",
      "#define GPU_TRACE_RECORD_TRANSACTION_WORD_COUNT 22u",
      "#define GPU_TRACE_RECORD_TRANSACTION_BEGIN 24u",
      "#define GPU_TRACE_RECORD_LOCATOR_BEGIN 88u",
      "#define GPU_TRACE_RECORD_CONDITION_BEGIN 168u",
      "#define GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_LOW 174u",
      "#define GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_HIGH 175u",
      "#define GPU_TRACE_CLASS_FORCE_INFER 1u",
      "#define GPU_TRACE_CLASS_GPU_UNDECIDED 2u",
      "#define GPU_TRACE_DEPTH_INVALID 0u",
      "#define GPU_TRACE_DEPTH_REUSE 1u",
      "#define GPU_TRACE_DEPTH_INFER 2u",
      "#define GPU_TRACE_SUBTITLE_SUPPRESSED 0u",
      "#define GPU_TRACE_SUBTITLE_OPTIONAL_OCR 1u",
      "#define GPU_TRACE_SUBTITLE_ABSTENTION 2u",
      "#define GPU_TRACE_SUBTITLE_HELD_WITH_DEPTH 5u",
      "#define GPU_TRACE_SUBTITLE_INVALID 6u",
      "#define GPU_TRACE_HOST_SUPPRESSED 0u",
      "#define GPU_TRACE_HOST_ORDINARY_RECORD 1u",
      "#define GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED (1u << 5u)",
      "#define GPU_TRACE_KNOWN_FLAGS 0x3fu",
      "#define GPU_TRACE_DECISION_COOKIE 0xD1EC15A5u",
      "#define GPU_TRACE_TOKEN_LOW_COOKIE 0xA3756C91u",
      "#define GPU_TRACE_TOKEN_HIGH_COOKIE 0x5C8A936Eu",
      "#define GPU_TRACE_RECEIPT_MAGIC 0x47524243u",
      "#define GPU_TRACE_REQUEST_MAGIC 0x54535152u",
      "#define GPU_TRACE_OPTIONAL_RECEIPT_MAGIC 0x52434F4Fu",
      "#define GPU_TRACE_WORK_FLAGS_COOKIE 0x6F435257u",
      "#define GPU_TRACE_WORK_OPTIONAL_OCR (1u << 0u)",
      "#define GPU_TRACE_WORK_SUBTITLE_OBSERVATION (1u << 1u)",
      "#define GPU_TRACE_WORK_OPTIONAL_OCR_DUE (1u << 3u)",
      "#define GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE (1u << 4u)",
    };
    for (const auto *definition : definitions) {
      EXPECT_NE(shader.find(definition), std::string::npos) << definition;
    }
  }

  bool create_structured_buffer(
    ID3D11Device *device,
    const std::uint32_t *initial,
    const std::uint32_t word_count,
    const UINT bind_flags,
    ComPtr<ID3D11Buffer> &buffer,
    ComPtr<ID3D11ShaderResourceView> *srv = nullptr,
    ComPtr<ID3D11UnorderedAccessView> *uav = nullptr
  ) {
    D3D11_BUFFER_DESC desc {};
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.ByteWidth = word_count * sizeof(std::uint32_t);
    desc.BindFlags = bind_flags;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(std::uint32_t);
    D3D11_SUBRESOURCE_DATA data {initial, 0u, 0u};
    if (FAILED(device->CreateBuffer(
          &desc, initial ? &data : nullptr, buffer.ReleaseAndGetAddressOf()
        ))) {
      return false;
    }
    if (srv && FAILED(device->CreateShaderResourceView(
          buffer.Get(), nullptr, srv->ReleaseAndGetAddressOf()
        ))) {
      return false;
    }
    return !uav || SUCCEEDED(device->CreateUnorderedAccessView(
      buffer.Get(), nullptr, uav->ReleaseAndGetAddressOf()
    ));
  }

  std::array<std::uint32_t, trace::transaction_word_count> transaction(
    const std::uint64_t token,
    const cuda_conditional_graph::branch_e branch,
    const cuda_conditional_graph::work_flag_e work,
    const bool optional_child
  ) {
    std::array<std::uint32_t, trace::transaction_word_count> words {};
    const auto request = cuda_conditional_graph::make_request(token, work);
    const auto receipt = cuda_conditional_graph::resolve_proposal(
      cuda_conditional_graph::make_proposal(branch, token),
      request,
      optional_child
    );
    std::memcpy(words.data(), &receipt, sizeof(receipt));
    std::memcpy(words.data() + 8u, &request, sizeof(request));
    return words;
  }

  TEST(HostSbsGpuTraceContractTest, AuthenticatedReuseClassifiesOrdinaryHoldAndDuePublication) {
    static_assert(static_cast<std::uint32_t>(
                    trace::subtitle_disposition_e::held_with_depth) == 5u);
    static_assert(static_cast<std::uint32_t>(trace::subtitle_disposition_e::invalid) == 6u);
    constexpr std::uint64_t token = 0x1020304050607080ull;
    struct case_t {
      cuda_conditional_graph::work_flag_e work;
      trace::host_subtitle_outcome_e host_outcome;
      bool optional_child;
      trace::subtitle_disposition_e expected;
    };
    constexpr std::array cases {
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr,
        trace::host_subtitle_outcome_e::ordinary_record,
        true,
        trace::subtitle_disposition_e::held_with_depth,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::held_with_depth,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::subtitle_observation,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::held_with_depth,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr_due,
        trace::host_subtitle_outcome_e::ordinary_record,
        true,
        trace::subtitle_disposition_e::optional_ocr,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr_due,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::abstention,
      },
    };
    constexpr auto flags = trace::subtitle_branch_gated;
    for (const auto &test : cases) {
      SCOPED_TRACE(cuda_conditional_graph::work_flags_value(test.work));
      const auto words = transaction(
        token, cuda_conditional_graph::branch_e::reuse,
        test.work, test.optional_child
      );
      const auto receipt = trace::authenticate_receipt(
        words,
        token,
        cuda_conditional_graph::work_flags_value(test.work),
        trace::submission_class_e::gpu_undecided
      );
      ASSERT_TRUE(receipt.receipt_valid);
      ASSERT_EQ(receipt.depth, trace::depth_disposition_e::reuse);
      EXPECT_EQ(
        receipt.optional_ocr_executed,
        test.work == cuda_conditional_graph::work_flag_e::optional_ocr_due &&
          test.optional_child
      );
      EXPECT_EQ(
        trace::classify_subtitle_disposition(
          cuda_conditional_graph::work_flags_value(test.work),
          test.host_outcome,
          receipt,
          flags
        ),
        test.expected
      );
    }
  }

  TEST(HostSbsGpuTraceContractTest, AuthenticatedOpaqueInferOwnsSubtitleExecutionProof) {
    constexpr std::uint64_t token = 0x1020304050607080ull;
    struct case_t {
      cuda_conditional_graph::work_flag_e work;
      trace::host_subtitle_outcome_e host_outcome;
      bool optional_child;
      trace::subtitle_disposition_e expected;
    };
    constexpr std::array cases {
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr,
        trace::host_subtitle_outcome_e::ordinary_record,
        true,
        trace::subtitle_disposition_e::optional_ocr,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::abstention,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::subtitle_observation,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::abstention,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr_due,
        trace::host_subtitle_outcome_e::ordinary_record,
        true,
        trace::subtitle_disposition_e::optional_ocr,
      },
      case_t {
        cuda_conditional_graph::work_flag_e::optional_ocr_due,
        trace::host_subtitle_outcome_e::ordinary_record,
        false,
        trace::subtitle_disposition_e::abstention,
      },
    };
    for (const auto &test : cases) {
      SCOPED_TRACE(cuda_conditional_graph::work_flags_value(test.work));
      const auto words = transaction(
        token, cuda_conditional_graph::branch_e::infer,
        test.work, test.optional_child
      );
      const auto receipt = trace::authenticate_receipt(
        words,
        token,
        cuda_conditional_graph::work_flags_value(test.work),
        trace::submission_class_e::gpu_undecided
      );
      ASSERT_TRUE(receipt.receipt_valid);
      ASSERT_EQ(receipt.depth, trace::depth_disposition_e::infer);
      EXPECT_EQ(
        trace::classify_subtitle_disposition(
          cuda_conditional_graph::work_flags_value(test.work),
          test.host_outcome,
          receipt,
          trace::subtitle_branch_gated
        ),
        test.expected
      );
      EXPECT_EQ(
        trace::classify_subtitle_disposition(
          cuda_conditional_graph::work_flags_value(test.work),
          test.host_outcome,
          receipt,
          trace::subtitle_branch_gated | trace::condition_executed
        ),
        trace::subtitle_disposition_e::invalid
      ) << "opaque execution proof must come from the authenticated receipt";
    }
  }

  TEST(HostSbsGpuTraceContractTest, SuppressionAndDepthReceiptValidityRemainDistinct) {
    constexpr std::uint64_t token = 0x8877665544332211ull;
    const auto suppressed_words = transaction(
      token,
      cuda_conditional_graph::branch_e::reuse,
      cuda_conditional_graph::work_flag_e::none,
      false
    );
    const auto suppressed_receipt = trace::authenticate_receipt(
      suppressed_words,
      token,
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::none
      ),
      trace::submission_class_e::gpu_undecided
    );
    ASSERT_TRUE(suppressed_receipt.receipt_valid);
    ASSERT_EQ(suppressed_receipt.depth, trace::depth_disposition_e::reuse);
    EXPECT_EQ(
      trace::classify_subtitle_disposition(
        cuda_conditional_graph::work_flags_value(
          cuda_conditional_graph::work_flag_e::none
        ),
        trace::host_subtitle_outcome_e::suppressed,
        suppressed_receipt,
        trace::subtitle_suppressed
      ),
      trace::subtitle_disposition_e::suppressed
    );

    auto invalid_words = transaction(
      token,
      cuda_conditional_graph::branch_e::infer,
      cuda_conditional_graph::work_flag_e::subtitle_observation,
      false
    );
    invalid_words[1u] ^= 1u;
    const auto invalid_receipt = trace::authenticate_receipt(
      invalid_words,
      token,
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::subtitle_observation
      ),
      trace::submission_class_e::gpu_undecided
    );
    ASSERT_FALSE(invalid_receipt.receipt_valid);
    constexpr auto published_flags =
      trace::ocr_record_submitted | trace::condition_executed;
    EXPECT_EQ(
      trace::classify_subtitle_disposition(
        cuda_conditional_graph::work_flags_value(
          cuda_conditional_graph::work_flag_e::subtitle_observation
        ),
        trace::host_subtitle_outcome_e::ordinary_record,
        invalid_receipt,
        published_flags
      ),
      trace::subtitle_disposition_e::abstention
    );

    const auto forced_reuse_words = transaction(
      token,
      cuda_conditional_graph::branch_e::reuse,
      cuda_conditional_graph::work_flag_e::subtitle_observation,
      false
    );
    const auto forced_reuse = trace::authenticate_receipt(
      forced_reuse_words,
      token,
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::subtitle_observation
      ),
      trace::submission_class_e::force_infer
    );
    EXPECT_FALSE(forced_reuse.receipt_valid);
    EXPECT_EQ(
      trace::classify_subtitle_disposition(
        cuda_conditional_graph::work_flags_value(
          cuda_conditional_graph::work_flag_e::subtitle_observation
        ),
        trace::host_subtitle_outcome_e::ordinary_record,
        forced_reuse,
        published_flags
      ),
      trace::subtitle_disposition_e::abstention
    );

    const auto infer_words = transaction(
      token,
      cuda_conditional_graph::branch_e::infer,
      cuda_conditional_graph::work_flag_e::subtitle_observation,
      false
    );
    const auto infer_receipt = trace::authenticate_receipt(
      infer_words,
      token,
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::subtitle_observation
      ),
      trace::submission_class_e::force_infer
    );
    ASSERT_TRUE(infer_receipt.receipt_valid);
    EXPECT_EQ(infer_receipt.depth, trace::depth_disposition_e::infer);
    EXPECT_EQ(
      trace::classify_subtitle_disposition(
        cuda_conditional_graph::work_flags_value(
          cuda_conditional_graph::work_flag_e::subtitle_observation
        ),
        trace::host_subtitle_outcome_e::ordinary_record,
        infer_receipt,
        published_flags
      ),
      trace::subtitle_disposition_e::abstention
    );
  }

  TEST(HostSbsGpuTraceContractTest, ForgedOrdinaryReuseOptionalReceiptIsRejected) {
    constexpr std::uint64_t token = 0x8877665544332211ull;
    auto words = transaction(
      token,
      cuda_conditional_graph::branch_e::reuse,
      cuda_conditional_graph::work_flag_e::optional_ocr,
      false
    );
    words[7u] = cuda_conditional_graph::optional_ocr_receipt_magic;
    words[1u] = words[0u] ^ cuda_conditional_graph::decision_cookie ^ words[7u];
    const auto receipt = trace::authenticate_receipt(
      words,
      token,
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::optional_ocr
      ),
      trace::submission_class_e::gpu_undecided
    );
    EXPECT_FALSE(receipt.receipt_valid);
    EXPECT_FALSE(receipt.optional_ocr_executed);
  }

  TEST(HostSbsGpuTraceWarpTest, AppendsAuthenticatesWrapsAndRepairsCommittedRing) {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level {};
    if (FAILED(D3D11CreateDevice(
          nullptr,
          D3D_DRIVER_TYPE_WARP,
          nullptr,
          0u,
          nullptr,
          0u,
          D3D11_SDK_VERSION,
          device.ReleaseAndGetAddressOf(),
          &level,
          context.ReleaseAndGetAddressOf()
        ))) {
      GTEST_SKIP() << "D3D11 WARP is unavailable";
    }

    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;
    const auto shader_path = std::filesystem::path {SUNSHINE_SHADERS_DIR} /
                             "host_sbs_gpu_trace_cs.hlsl";
    const auto compiled = D3DCompileFromFile(
      shader_path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS,
      0u,
      bytecode.ReleaseAndGetAddressOf(),
      errors.ReleaseAndGetAddressOf()
    );
    ASSERT_TRUE(SUCCEEDED(compiled)) << (errors ?
      std::string {
        static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize()
      } : "D3DCompileFromFile failed without diagnostics");
    ComPtr<ID3D11ComputeShader> shader;
    ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
      bytecode->GetBufferPointer(),
      bytecode->GetBufferSize(),
      nullptr,
      shader.ReleaseAndGetAddressOf()
    )));

    constexpr std::uint64_t token = 0x1020304050607080ull;
    auto transaction_words = transaction(
      token,
      cuda_conditional_graph::branch_e::reuse,
      cuda_conditional_graph::work_flag_e::optional_ocr,
      true
    );
    D3D11_BUFFER_DESC transaction_desc {};
    transaction_desc.Usage = D3D11_USAGE_DEFAULT;
    transaction_desc.ByteWidth = sizeof(transaction_words);
    transaction_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    transaction_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    D3D11_SUBRESOURCE_DATA transaction_data {transaction_words.data(), 0u, 0u};
    ComPtr<ID3D11Buffer> transaction_buffer;
    ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
      &transaction_desc, &transaction_data, transaction_buffer.ReleaseAndGetAddressOf()
    )));
    D3D11_SHADER_RESOURCE_VIEW_DESC raw_srv_desc {};
    raw_srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    raw_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    raw_srv_desc.BufferEx.NumElements = trace::transaction_word_count;
    raw_srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    ComPtr<ID3D11ShaderResourceView> transaction_srv;
    ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
      transaction_buffer.Get(), &raw_srv_desc, transaction_srv.ReleaseAndGetAddressOf()
    )));

    std::array<std::uint32_t, trace::subtitle_locator_word_count> locator {};
    locator[0u] = v2::subtitle_locator_state_schema;
    locator[1u] = v2::subtitle_locator_state_tag;
    locator[2u] = 1u | 4u;
    locator[3u] = 1u;
    locator[4u] = 1u;
    locator[18u] = 0x3c23d70au;  // float32 0.01
    locator[19u] = 1u;
    locator[20u] = 1u;
    locator[21u] = 1u;
    locator[22u] = 41u;
    locator[24u] = 1u;
    locator[26u] = 3u;
    locator[27u] = 770u;
    locator[28u] = 434u;
    std::array<std::uint32_t, trace::subtitle_condition_word_count> condition {
      v2::subtitle_condition_param_schema,
      v2::subtitle_condition_param_tag,
      1u,
      0u,
      1u,
      locator[18u],
    };
    ComPtr<ID3D11Buffer> locator_buffer;
    ComPtr<ID3D11ShaderResourceView> locator_srv;
    ComPtr<ID3D11Buffer> condition_buffer;
    ComPtr<ID3D11ShaderResourceView> condition_srv;
    ASSERT_TRUE(create_structured_buffer(
      device.Get(), locator.data(), locator.size(), D3D11_BIND_SHADER_RESOURCE,
      locator_buffer, &locator_srv
    ));
    ASSERT_TRUE(create_structured_buffer(
      device.Get(), condition.data(), condition.size(), D3D11_BIND_SHADER_RESOURCE,
      condition_buffer, &condition_srv
    ));

    std::vector<std::uint32_t> initial_ring(trace::ring_word_count, 0u);
    initial_ring[0u] = trace::ring_schema;
    initial_ring[1u] = trace::ring_tag;
    initial_ring[2u] = trace::capacity;
    initial_ring[3u] = trace::record_word_count;
    initial_ring[4u] = 1u;
    ComPtr<ID3D11Buffer> ring_buffer;
    ComPtr<ID3D11UnorderedAccessView> ring_uav;
    ASSERT_TRUE(create_structured_buffer(
      device.Get(), initial_ring.data(), initial_ring.size(),
      D3D11_BIND_UNORDERED_ACCESS, ring_buffer, nullptr, &ring_uav
    ));
    D3D11_BUFFER_DESC staging_desc {};
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.ByteWidth = trace::ring_byte_count;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    staging_desc.StructureByteStride = sizeof(std::uint32_t);
    ComPtr<ID3D11Buffer> staging;
    ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
      &staging_desc, nullptr, staging.ReleaseAndGetAddressOf()
    )));
    D3D11_BUFFER_DESC constants_desc {};
    constants_desc.Usage = D3D11_USAGE_DEFAULT;
    constants_desc.ByteWidth = trace::constant_word_count * sizeof(std::uint32_t);
    constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> constants_buffer;
    ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
      &constants_desc, nullptr, constants_buffer.ReleaseAndGetAddressOf()
    )));

    std::array<std::uint32_t, 20> constants {
      41u, 0u, 0u, 0u, 0x89abcdefu, 0x01234567u,
      static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(token >> 32u),
      cuda_conditional_graph::work_flags_value(
        cuda_conditional_graph::work_flag_e::optional_ocr),
      static_cast<std::uint32_t>(trace::submission_class_e::gpu_undecided),
      trace::subtitle_branch_gated,
      static_cast<std::uint32_t>(trace::host_subtitle_outcome_e::ordinary_record),
      1920u, 1080u, 770u, 434u,
      0x89abcdefu, 0x01234567u, 0u, 0u,
    };
    const auto dispatch = [&] {
      context->UpdateSubresource(
        constants_buffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      ID3D11ShaderResourceView *srvs[] = {
        transaction_srv.Get(), locator_srv.Get(), condition_srv.Get()
      };
      context->CSSetShader(shader.Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, 3u, srvs);
      context->CSSetConstantBuffers(0u, 1u, constants_buffer.GetAddressOf());
      context->CSSetUnorderedAccessViews(0u, 1u, ring_uav.GetAddressOf(), nullptr);
      context->Dispatch(1u, 1u, 1u);
      ID3D11ShaderResourceView *null_srvs[3] = {};
      ID3D11Buffer *null_constant = nullptr;
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context->CSSetShaderResources(0u, 3u, null_srvs);
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
      context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context->CSSetShader(nullptr, nullptr, 0u);
    };
    const auto read_ring = [&] {
      std::vector<std::uint32_t> words(trace::ring_word_count, 0u);
      context->CopyResource(staging.Get(), ring_buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      EXPECT_TRUE(SUCCEEDED(context->Map(
        staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
      )));
      if (mapped.pData) {
        std::memcpy(words.data(), mapped.pData, trace::ring_byte_count);
        context->Unmap(staging.Get(), 0u);
      }
      return words;
    };

    dispatch();
    auto ring = read_ring();
    ASSERT_EQ(ring[1u], trace::ring_tag);
    ASSERT_EQ(ring[4u], 2u);
    ASSERT_EQ(ring[6u], 1u);
    ASSERT_EQ(ring[7u], 1u);
    const auto first = trace::record_base(0u);
    EXPECT_EQ(ring[first + trace::word_index(trace::record_word_e::commit_tag)],
              trace::record_tag);
    EXPECT_EQ(ring[first + trace::word_index(trace::record_word_e::depth_disposition)],
              static_cast<std::uint32_t>(trace::depth_disposition_e::reuse));
    EXPECT_EQ(ring[first + trace::word_index(trace::record_word_e::subtitle_disposition)],
              static_cast<std::uint32_t>(
                trace::subtitle_disposition_e::held_with_depth));
    EXPECT_EQ(ring[first + trace::word_index(
                          trace::record_word_e::observation_timestamp_low)],
              0x89abcdefu);
    EXPECT_EQ(ring[first + trace::word_index(
                          trace::record_word_e::observation_timestamp_high)],
              0x01234567u);

    transaction_words = transaction(
      token,
      cuda_conditional_graph::branch_e::infer,
      cuda_conditional_graph::work_flag_e::subtitle_observation,
      false
    );
    context->UpdateSubresource(
      transaction_buffer.Get(), 0u, nullptr, transaction_words.data(), 0u, 0u
    );
    constants[8u] = cuda_conditional_graph::work_flags_value(
      cuda_conditional_graph::work_flag_e::subtitle_observation
    );
    constants[9u] = static_cast<std::uint32_t>(trace::submission_class_e::force_infer);
    constants[10u] = trace::ocr_record_submitted | trace::condition_executed;
    for (std::uint32_t index = 0u; index < trace::capacity; ++index) {
      dispatch();
    }
    ring = read_ring();
    ASSERT_EQ(ring[1u], trace::ring_tag);
    EXPECT_EQ(ring[4u], trace::capacity + 2u);
    EXPECT_EQ(ring[6u], 1u);
    EXPECT_EQ(ring[7u], trace::capacity);
    const auto oldest = trace::record_base(1u);
    const auto newest = trace::record_base(0u);
    EXPECT_EQ(ring[oldest + trace::word_index(trace::record_word_e::sequence_low)], 2u);
    EXPECT_EQ(ring[newest + trace::word_index(trace::record_word_e::sequence_low)],
              trace::capacity + 1u);
    EXPECT_EQ(ring[newest + trace::word_index(trace::record_word_e::depth_disposition)],
              static_cast<std::uint32_t>(trace::depth_disposition_e::infer));
    EXPECT_EQ(ring[newest + trace::word_index(trace::record_word_e::subtitle_disposition)],
              static_cast<std::uint32_t>(trace::subtitle_disposition_e::abstention));

    // A torn/unknown header is repaired as an empty ring; record and header tags still publish
    // last around the fresh sequence-1 append.
    ring[1u] = 0u;
    ring[6u] = trace::capacity + 7u;
    context->UpdateSubresource(ring_buffer.Get(), 0u, nullptr, ring.data(), 0u, 0u);
    dispatch();
    ring = read_ring();
    EXPECT_EQ(ring[0u], trace::ring_schema);
    EXPECT_EQ(ring[1u], trace::ring_tag);
    EXPECT_EQ(ring[4u], 2u);
    EXPECT_EQ(ring[6u], 1u);
    EXPECT_EQ(ring[7u], 1u);
    EXPECT_EQ(ring[first + trace::word_index(trace::record_word_e::sequence_low)], 1u);
    EXPECT_EQ(ring[first + trace::word_index(trace::record_word_e::commit_tag)],
              trace::record_tag);
  }
}  // namespace

#else

TEST(HostSbsGpuTraceWarpTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
