// One authority for deciding whether the current model-input/appearance/depth tuple becomes the
// next comparison baseline. Keep the fused map root's tuple-copy and near-identical owner paths on
// this exact predicate: an authenticated owner must never name pixels/depth that were not copied.
bool DepthValidHistoryAdvances(
    const float current_depth_validity,
    const float model_input_history_state) {
    bool hold_structureless_gap =
        model_input_history_state > 1.5f && model_input_history_state < 2.5f;
    bool hold_geometry_confirmation =
        model_input_history_state > 3.5f && model_input_history_state < 4.5f;
    return current_depth_validity >= 0.5f &&
        !hold_structureless_gap && !hold_geometry_confirmation;
}
