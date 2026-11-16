// One authority for deciding whether the current model-input/appearance/depth tuple becomes the
// next comparison baseline. Keep the fused map root's tuple-copy and near-identical owner paths on
// this exact predicate: an authenticated owner must never name pixels/depth that were not copied.
// Every valid depth frame advances the baseline; an invalid or collapsed one holds it.
bool DepthValidHistoryAdvances(const float current_depth_validity) {
    return current_depth_validity >= 0.5f;
}
