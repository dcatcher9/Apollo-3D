/** @file Private startup helpers for the direct glyph TensorRT engine. */

// Included by video_depth_estimator.cpp inside namespace models after the shared
// TensorRT cache and serialization helpers.

static bool build_direct_glyph_engine(const direct_glyph_artifact_t &artifact, bool batched_tiles) {
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
  if (!builder) {
    return false;
  }
#if NV_TENSORRT_MAJOR < 11
  constexpr auto network_flags = 1u << static_cast<std::uint32_t>(
    nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
  );
#else
  constexpr auto network_flags = 0u;
#endif
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(network_flags));
  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!network || !config) {
    return false;
  }
  // The graph sets convolution precision explicitly. Keep its FP32 scoring operations at full precision.
  config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
  if (!parser || !parser->parseFromFile(artifact.source_path.string().c_str(),
                                       static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    BOOST_LOG(warning) << "Could not parse the authenticated direct glyph ONNX.";
    return false;
  }

  const auto matches = [](const nvinfer1::ITensor *tensor, const std::string_view name) {
    return tensor && tensor->getName() && name == tensor->getName() &&
           tensor->getType() == nvinfer1::DataType::kFLOAT;
  };
  if (network->getNbInputs() != 1 || network->getNbOutputs() != 1 ||
      !matches(network->getInput(0), direct_glyph_engine_input_name) ||
      !matches(network->getOutput(0), direct_glyph_engine_output_name)) {
    BOOST_LOG(warning) << "Direct glyph ONNX does not expose its fixed FP32 rgb/logits interface.";
    return false;
  }

  if (batched_tiles) {
    using namespace host_sbs_direct_glyph_gpu;
    static_assert(tile_capacity == 6u);
    auto *input = network->getInput(0);
    const auto shape = [](int tiles) {
      return nvinfer1::Dims4 {tiles, 3, int(tile_height), int(tile_width)};
    };
    input->setDimensions(shape(-1));
    auto *profile = builder->createOptimizationProfile();
    using point = nvinfer1::OptProfileSelector;
    if (!profile || !profile->setDimensions(input->getName(), point::kMIN, shape(1)) ||
        !profile->setDimensions(input->getName(), point::kOPT, shape(4)) ||
        !profile->setDimensions(input->getName(), point::kMAX, shape(int(tile_capacity))) ||
        config->addOptimizationProfile(profile) < 0) {
      return false;
    }
  }
  auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  return serialized && publish_serialized_engine(artifact.engine_path, *serialized, "direct glyph engine");
}

static bool validate_direct_glyph_io_locked(nvinfer1::ICudaEngine *engine, engine_slot &slot) {
  if (!engine) {
    return false;
  }
  if (slot.io_validated) {
    return slot.io_compatible;
  }
  const auto matches = [&](const std::string_view name, const nvinfer1::TensorIOMode mode) {
    return engine->getTensorIOMode(name.data()) == mode &&
           engine->getTensorDataType(name.data()) == nvinfer1::DataType::kFLOAT &&
           engine->getTensorLocation(name.data()) == nvinfer1::TensorLocation::kDEVICE &&
           engine->getTensorFormat(name.data()) == nvinfer1::TensorFormat::kLINEAR &&
           engine->getTensorVectorizedDim(name.data()) == -1 &&
           !engine->isShapeInferenceIO(name.data());
  };
  slot.io_compatible = engine->getNbIOTensors() == 2 && engine->getNbOptimizationProfiles() == 1 &&
                       matches(direct_glyph_engine_input_name, nvinfer1::TensorIOMode::kINPUT) &&
                       matches(direct_glyph_engine_output_name, nvinfer1::TensorIOMode::kOUTPUT);
  slot.io_validated = true;
  return slot.io_compatible;
}

static bool bind_direct_glyph_shape(
  nvinfer1::IExecutionContext &context, host_sbs_direct_glyph_gpu::tile_shape_t profile, bool batched_tiles
) {
  // Send Dims by reference only. Returning Dims by value from the MSVC DLL is
  // incompatible with the MinGW host ABI.
  const auto shape = make_input_dims(profile.height, profile.width);
  const auto bytes = static_cast<std::int64_t>(profile.output_words() * sizeof(float));
  if (batched_tiles) {
    auto largest = shape;
    largest.d[0] = host_sbs_direct_glyph_gpu::tile_capacity;
    if (!context.setInputShape(direct_glyph_engine_input_name.data(), largest) ||
        context.getMaxOutputSize(direct_glyph_engine_output_name.data()) != bytes * largest.d[0]) {
      return false;
    }
  }
  return context.setInputShape(direct_glyph_engine_input_name.data(), shape) &&
         (batched_tiles || context.getMaxOutputSize(direct_glyph_engine_output_name.data()) == bytes);
}

static bool warmup_direct_glyph_context(
  cuda_driver_api &cuda, nvinfer1::IExecutionContext &context, host_sbs_direct_glyph_gpu::tile_shape_t profile
) {
  const std::size_t input_words = profile.input_words();
  const std::size_t input_bytes = input_words * sizeof(float);
  const std::size_t output_bytes = profile.output_words() * sizeof(float);
  if (!cuda.cuStreamSynchronize || !cuda.cuMemcpyHtoD) {
    return false;
  }
  // Match the graphics binding alignment using one private, short-lived startup allocation.
  std::vector<float> zero_rgb(input_words, 0.0f);
  CUdeviceptr storage = 0u;
  if (cuda.cuMemAlloc(&storage, input_bytes + output_bytes) != CUDA_SUCCESS) {
    return false;
  }
  auto free_storage = util::fail_guard([&]() { cuda.cuMemFree(storage); });
  if (cuda.cuMemcpyHtoD(storage, zero_rgb.data(), input_bytes) != CUDA_SUCCESS) {
    return false;
  }
  CUstream stream = nullptr;
  if (cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || !stream) {
    return false;
  }
  auto destroy_stream = util::fail_guard([&]() { cuda.cuStreamDestroy(stream); });
  if (!context.setTensorAddress(direct_glyph_engine_input_name.data(), reinterpret_cast<void *>(storage)) ||
      !context.setTensorAddress(direct_glyph_engine_output_name.data(), reinterpret_cast<void *>(storage + input_bytes))) {
    return false;
  }

  bool enqueued;
  {
    std::lock_guard lock(g_trt_mutex);
    enqueued = context.enqueueV3(stream);
  }
  // A failed enqueue can still have submitted work. Retain uncertain operands
  // with the context until process teardown rather than racing the CUDA stream.
  if (cuda.cuStreamSynchronize(stream) != CUDA_SUCCESS) {
    free_storage.disable();
    destroy_stream.disable();
    BOOST_LOG(warning) << "Could not drain direct glyph warmup; retaining its CUDA operands.";
    return false;
  }
  return enqueued;
}

static bool acquire_direct_glyph_context(
  const std::filesystem::path &assets_dir, cuda_driver_api &cuda, CUdevice device,
  nvinfer1::ICudaEngine *&engine, nvinfer1::IExecutionContext *&context, std::string &engine_key,
  host_sbs_direct_glyph_gpu::tile_profile_e profile = host_sbs_direct_glyph_gpu::tile_profile_e::common,
  bool batched_tiles = false
) {
  if (engine || context || !engine_key.empty() || !cuda.is_valid()) {
    return false;
  }
  const auto cuda_ctx = primary_context(cuda, device);
  if (!cuda_ctx || cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    return false;
  }

  auto artifact = get_direct_glyph_artifact(assets_dir, engine_compatibility_tag(cuda, device), profile, batched_tiles);
  if (!artifact) {
    return false;
  }
  auto key = std::to_string(device) + ':' + artifact->name;
  nvinfer1::ICudaEngine *cached_engine = nullptr;
  nvinfer1::IExecutionContext *leased_context = nullptr;
  bool reserved = false;
  {
    std::lock_guard compile_lock(g_tensorrt_compile_mutex);
    const auto checkout = [&]() {
      cached_engine = acquire_engine_locked(key, artifact->engine_path, leased_context);
      return validate_direct_glyph_io_locked(cached_engine, g_engines[key]);
    };

    std::error_code file_error;
    const auto status = std::filesystem::status(artifact->engine_path, file_error);
    if (file_error && file_error != std::errc::no_such_file_or_directory) {
      return false;
    }
    const bool cache_exists = std::filesystem::is_regular_file(status);
    if (std::filesystem::exists(status) && !cache_exists) {
      return false;
    }
    {
      std::lock_guard lock(g_trt_mutex);
      const auto found = g_engines.find(key);
      const bool resident = found != g_engines.end() && found->second.engine;
      if ((cache_exists || resident) && !checkout()) {
        return_execution_context_locked(key, leased_context);
        if (!detach_incompatible_engine_locked(key)) {
          return false;
        }
        erase_empty_engine_slot_locked(key);
        cached_engine = nullptr;
      }
    }
    if (!cached_engine) {
      // Repair only the exact cache file and only after verifying it has no live owner.
      if (cache_exists) {
        std::filesystem::remove(artifact->engine_path, file_error);
        if (file_error) {
          return false;
        }
      }
      if (!build_direct_glyph_engine(*artifact, batched_tiles)) {
        return false;
      }
      std::lock_guard lock(g_trt_mutex);
      if (!checkout()) {
        return_execution_context_locked(key, leased_context);
        detach_incompatible_engine_locked(key);
        erase_empty_engine_slot_locked(key);
        return false;
      }
    }
    if (!leased_context) {
      std::lock_guard lock(g_trt_mutex);
      auto &slot = g_engines[key];
      reserved = slot.context_accounting.reserve(slot_context_limit(slot));
      if (!reserved) {
        return false;
      }
    }
  }

  if (reserved) {
    leased_context = cached_engine->createExecutionContext();
    if (!leased_context) {
      std::lock_guard lock(g_trt_mutex);
      release_context_reservation_locked(key);
      return false;
    }
  }
  const auto shape = host_sbs_direct_glyph_gpu::tile_shape(profile);
  if (!bind_direct_glyph_shape(*leased_context, shape, batched_tiles) ||
      (reserved && !warmup_direct_glyph_context(cuda, *leased_context, shape))) {
    std::lock_guard lock(g_trt_mutex);
    quarantine_execution_context_locked(key, leased_context, !reserved);
    BOOST_LOG(warning) << "Direct glyph TensorRT fixed shape validation or startup warmup failed.";
    return false;
  }
  if (reserved) {
    std::lock_guard lock(g_trt_mutex);
    mark_execution_context_warmed_locked(key);
  }

  // The caller owns the checked-out lease and rebinds its private mapped buffers
  // before graph capture. Recycle only after its interop and graph work drains.
  engine = cached_engine;
  context = leased_context;
  engine_key = std::move(key);
  return true;
}
