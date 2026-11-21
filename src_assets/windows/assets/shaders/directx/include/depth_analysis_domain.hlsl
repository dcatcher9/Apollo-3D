#ifndef DEPTH_ANALYSIS_DOMAIN_HLSL
#define DEPTH_ANALYSIS_DOMAIN_HLSL

// One structured element, stride 32. This is derived tensor geometry, not a pixel
// classifier or dispatch-coverage proof. Zero initialization is an invalid domain.
struct TensorAnalysisDomain {
    uint4 extent; // valid, target width, target height, reserved zero
    uint4 rect;   // left, top, right, bottom in integer tensor cells
};

bool TensorAnalysisRectangleValid(uint2 size, uint4 rectangle) {
    return all(size > 0u) && all(rectangle.xy < rectangle.zw) &&
        all(rectangle.zw <= size);
}

bool TensorAnalysisDomainValid(TensorAnalysisDomain domain, uint2 expected_size) {
    return domain.extent.x == 1u && all(domain.extent.yz == expected_size) &&
        domain.extent.w == 0u && TensorAnalysisRectangleValid(expected_size, domain.rect);
}

// Call after validating the descriptor and its join to the current depth constants.
bool TensorAnalysisDomainContainsCell(uint4 rectangle, uint2 position) {
    return all(position >= rectangle.xy) && all(position < rectangle.zw);
}

// The producer supplies actual bound-source validity, including its complete
// content-constant validation. Malformed source/content publishes a literal invalid
// record rather than retaining a prior valid rectangle at the same resource pointer.
TensorAnalysisDomain MakeTensorAnalysisDomain(uint2 size, uint4 content, bool source_valid) {
    bool valid = source_valid && TensorAnalysisRectangleValid(size, content);
    TensorAnalysisDomain domain;
    domain.extent = uint4(valid ? 1u : 0u, size, 0u);
    domain.rect = valid ? content : 0u;
    return domain;
}

#endif
