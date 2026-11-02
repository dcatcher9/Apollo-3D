// PRIVATE queued-job permission. Disabled/unbound b5 is literal legacy mode, not a queued ticket.
// An enabled ticket must bind its exact finalized transaction; rejection disables Base AND UI.
cbuffer HostSbsQueuedRenderExpected : register(b5) {
    uint4 queued_control;      // enabled, QRG1, schema1, reserved0
    uint4 queued_root;         // token64, expected work0/1/2, existing work cookie
    uint4 queued_observation;  // source observation64, reserved0/0
};
ByteAddressBuffer QueuedRenderTransaction : register(t10);

bool QueuedRenderAuthorized() {
    if (queued_control.x == 0u) return true;
    if (any(queued_control != uint4(1u, 0x31475251u, 1u, 0u)) ||
        all(queued_root.xy == 0u) || all(queued_observation.xy == 0u) ||
        any(queued_observation.zw != 0u)) return false;
    uint work = queued_root.z;
    if (work != 0u && work != 1u && work != 2u) return false;
    uint work_cookie = work == 0u ? 0u : work ^ 0x6f435257u;
    if (queued_root.w != work_cookie) return false;

    uint bytes;
    QueuedRenderTransaction.GetDimensions(bytes);
    if (bytes != 256u) return false;
    // Literal loads/fields only: no dynamic cbuffer array or component selection.
    uint4 decision0 = QueuedRenderTransaction.Load4(0u);
    uint4 decision1 = QueuedRenderTransaction.Load4(16u);
    uint4 request0 = QueuedRenderTransaction.Load4(32u);
    uint4 request1 = QueuedRenderTransaction.Load4(48u);
    uint2 token_cookies = queued_root.xy ^ uint2(0xa3756c91u, 0x5c8a936eu);
    if (any(request0.xy != queued_root.xy) || any(request0.zw != token_cookies) ||
        any(request1 != uint4(0x54535152u, work, work_cookie, 0u))) return false;
    uint marker = decision1.w;
    if (marker != 0u && (marker != 0x52434f4fu || work != 1u)) return false;
    if (decision0.x != 1u || decision0.y != (1u ^ 0xd1ec15a5u ^ marker) ||
        any(decision0.zw != queued_root.xy) || any(decision1.xy != token_cookies) ||
        decision1.z != 0x47524243u) return false;

    uint2 field;
    BaseParallax.GetDimensions(field.x, field.y);
    if (any(field == 0u)) return false;
    if (any(QueuedRenderTransaction.Load4(80u) != uint4(1u, 1u, 1u, 0u)) ||
        any(QueuedRenderTransaction.Load4(96u) != uint4((field + 15u) / 16u, 1u, 0u))) return false;
    uint vectors, stride;
    ParallaxState.GetDimensions(vectors, stride);
    if (vectors != V2_SHADOW_STATE_VECTOR_COUNT || stride != 16u) return false;
    uint2 owner_clock = asuint(ParallaxState[V2_STATE_VECTOR_GAIN_LAST_OBSERVATION_LOW].xy);
    return all(owner_clock == queued_observation.xy);
}
