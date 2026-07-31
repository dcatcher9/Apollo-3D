#ifndef SBS_SCENE_RULES_MEDIA_HLSL
#define SBS_SCENE_RULES_MEDIA_HLSL

// A static-media weight is positive evidence for an image, not a negative "text" label.
// This distinction matters: text, diagrams, subtitles, and grayscale art cannot be separated
// semantically by the compact scene grid. Only sustained colour with local luminance structure
// may move a static crop boundary. Generic content between media extrema remains inside the
// rectangular hull automatically; achromatic video is handled by the temporal path.
float SbsRuleMediaRamp(float value, float low, float high) {
    return saturate((value - low) / max(high - low, 1.0e-5f));
}

// Keep the authoritative classifier resource-free. The evidence pass already has every scalar
// in registers, computes this exactly once, and publishes the result in dense lane
// PHOTOGRAPHIC_CONTENT_COLLAGE. Reducers consume that lane; they must not reconstruct a subtly
// different classifier from analysis/history resources.
float SbsRuleStaticMediaWeight(
    float viewport_valid,
    float advertisement_unsafe,
    float ui_like,
    float variance,
    float current_chroma,
    float photo,
    float chroma_ema,
    out bool inputs_valid)
{
    inputs_valid =
        all(isfinite(float4(
            viewport_valid,
            advertisement_unsafe,
            ui_like,
            variance))) &&
        all(isfinite(float3(current_chroma, photo, chroma_ema))) &&
        viewport_valid >= 0.0f;
    if (!inputs_valid) {
        return 0.0f;
    }
    if (viewport_valid <= 0.5f) {
        return 0.0f;
    }

    const float unsafe = max(advertisement_unsafe, ui_like);
    const float chroma = min(
        saturate(current_chroma),
        saturate(chroma_ema));
    // Colour is the strongest text-resistant cue available in the existing GPU contract.
    // Local variance prevents a flat saturated UI button from contributing its whole interior;
    // only textured colour regions accumulate enough mass to establish a boundary.
    const float colour_media =
        SbsRuleMediaRamp(chroma, 0.08f, 0.22f) *
        SbsRuleMediaRamp(saturate(variance), 0.04f, 0.16f) *
        SbsRuleMediaRamp(saturate(photo), 0.18f, 0.36f);

    // Unsafe/UI evidence is a soft penalty rather than a second hard gate. This keeps the rule
    // monotonic and lets a strong media region survive imperfect UI/ad classification.
    const float safe_weight = 1.0f - saturate(unsafe);
    return saturate(colour_media * safe_weight * safe_weight);
}

// The stored value is the classifier result, not another heuristic label. A saturated finite
// range is therefore part of the private producer/consumer contract.
float SbsRuleStoredMediaWeight(float stored_weight, out bool inputs_valid) {
    inputs_valid =
        isfinite(stored_weight) &&
        stored_weight >= 0.0f &&
        stored_weight <= 1.0f;
    return inputs_valid ? stored_weight : 0.0f;
}

#endif
