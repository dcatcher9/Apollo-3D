#ifndef DEPTH_ANALYSIS_DOMAIN_CONSUMER_HLSL
#define DEPTH_ANALYSIS_DOMAIN_CONSUMER_HLSL

// Each consumer declares TensorAnalysisDomainRead at its former mask slot. A compact
// record and b0 must describe the same current tensor; no stale rectangle is authority.
bool LoadTensorAnalysisDomain(out uint4 rectangle) {
    rectangle = 0u;
    uint count, stride;
    TensorAnalysisDomainRead.GetDimensions(count, stride);
    if (count != 1u || stride != 32u) return false;
    TensorAnalysisDomain domain = TensorAnalysisDomainRead[0u];
    if (!TensorAnalysisDomainValid(domain, uint2(target_w, target_h)) ||
        any(analysis_content_reserved != 0u) ||
        any(domain.rect != uint4(analysis_content_left, analysis_content_top,
            analysis_content_right, analysis_content_bottom))) return false;
    rectangle = domain.rect;
    return true;
}

bool TensorAnalysisCellAdmitted(uint2 position, bool use_domain,
    bool domain_valid, uint4 rectangle) {
    if (use_domain) return domain_valid && TensorAnalysisDomainContainsCell(rectangle, position);
    // Independently authored replay masks may contain arbitrary holes. Keep this
    // texture lookup in the legacy entry rather than replacing it with a rectangle.
    return TensorExclusion[position] == 0u;
}

uint2 TensorAnalysisClampCell(uint2 position, bool use_domain, uint4 rectangle) {
    if (use_domain) return clamp(position, rectangle.xy, rectangle.zw - 1u);
    return DepthAnalysisClampCell(position);
}

#endif
