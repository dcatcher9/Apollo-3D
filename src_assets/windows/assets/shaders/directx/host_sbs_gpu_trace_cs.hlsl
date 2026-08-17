// Diagnostic-only completion ring. This pass snapshots the immutable postprocessed CBRG/RQST
// transaction, finalized SLR state, and condition parameters without CPU readback. It has no
// rendering authority and failure to compile or dispatch must not affect Host SBS production.

ByteAddressBuffer TraceTransaction : register(t0);
StructuredBuffer<uint> TraceSubtitleLocator : register(t1);
StructuredBuffer<uint> TraceSubtitleCondition : register(t2);
RWStructuredBuffer<uint> TraceRing : register(u0);

#include "include/depth_coordinate_v2_contract.generated.hlsl"

cbuffer TraceConstants : register(b0) {
    uint trace_frame_low;
    uint trace_frame_high;
    uint trace_analysis_generation_low;
    uint trace_analysis_generation_high;
    uint trace_domain_tag_low;
    uint trace_domain_tag_high;
    uint trace_transaction_token_low;
    uint trace_transaction_token_high;
    uint trace_expected_work;
    uint trace_submission_class;
    uint trace_flags;
    uint trace_host_subtitle_outcome;
    uint trace_source_width;
    uint trace_source_height;
    uint trace_field_width;
    uint trace_field_height;
    uint trace_observation_timestamp_low;
    uint trace_observation_timestamp_high;
    uint trace_padding0;
    uint trace_padding1;
};

#define GPU_TRACE_RING_SCHEMA 3u
#define GPU_TRACE_RING_TAG 0x48525447u
#define GPU_TRACE_RECORD_TAG 0x31525447u
#define GPU_TRACE_CAPACITY 300u
#define GPU_TRACE_HEADER_WORDS 16u
#define GPU_TRACE_RECORD_WORDS 176u
#define GPU_TRACE_TRANSACTION_WORDS 64u
#define GPU_TRACE_LOCATOR_WORDS 80u
#define GPU_TRACE_CONDITION_WORDS 6u

#define GPU_TRACE_HEADER_SCHEMA 0u
#define GPU_TRACE_HEADER_TAG 1u
#define GPU_TRACE_HEADER_CAPACITY 2u
#define GPU_TRACE_HEADER_RECORD_WORDS 3u
#define GPU_TRACE_HEADER_NEXT_SEQUENCE_LOW 4u
#define GPU_TRACE_HEADER_NEXT_SEQUENCE_HIGH 5u
#define GPU_TRACE_HEADER_NEXT_SLOT 6u
#define GPU_TRACE_HEADER_COMMITTED_COUNT 7u
#define GPU_TRACE_HEADER_RESERVED_BEGIN 8u

#define GPU_TRACE_RECORD_SCHEMA 0u
#define GPU_TRACE_RECORD_COMMIT_TAG 1u
#define GPU_TRACE_RECORD_SEQUENCE_LOW 2u
#define GPU_TRACE_RECORD_SEQUENCE_HIGH 3u
#define GPU_TRACE_RECORD_FRAME_LOW 4u
#define GPU_TRACE_RECORD_FRAME_HIGH 5u
#define GPU_TRACE_RECORD_ANALYSIS_GENERATION_LOW 6u
#define GPU_TRACE_RECORD_ANALYSIS_GENERATION_HIGH 7u
#define GPU_TRACE_RECORD_DOMAIN_TAG_LOW 8u
#define GPU_TRACE_RECORD_DOMAIN_TAG_HIGH 9u
#define GPU_TRACE_RECORD_TRANSACTION_TOKEN_LOW 10u
#define GPU_TRACE_RECORD_TRANSACTION_TOKEN_HIGH 11u
#define GPU_TRACE_RECORD_SUBMISSION_CLASS 12u
#define GPU_TRACE_RECORD_DEPTH_DISPOSITION 13u
#define GPU_TRACE_RECORD_EXPECTED_WORK 14u
#define GPU_TRACE_RECORD_SUBTITLE_DISPOSITION 15u
#define GPU_TRACE_RECORD_FLAGS 16u
#define GPU_TRACE_RECORD_HOST_SUBTITLE_OUTCOME 17u
#define GPU_TRACE_RECORD_SOURCE_WIDTH 18u
#define GPU_TRACE_RECORD_SOURCE_HEIGHT 19u
#define GPU_TRACE_RECORD_FIELD_WIDTH 20u
#define GPU_TRACE_RECORD_FIELD_HEIGHT 21u
#define GPU_TRACE_RECORD_TRANSACTION_WORD_COUNT 22u
#define GPU_TRACE_RECORD_RESERVED0 23u
#define GPU_TRACE_RECORD_TRANSACTION_BEGIN 24u
#define GPU_TRACE_RECORD_LOCATOR_BEGIN 88u
#define GPU_TRACE_RECORD_CONDITION_BEGIN 168u
#define GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_LOW 174u
#define GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_HIGH 175u

#define GPU_TRACE_CLASS_FORCE_INFER 1u
#define GPU_TRACE_CLASS_GPU_UNDECIDED 2u
#define GPU_TRACE_DEPTH_INVALID 0u
#define GPU_TRACE_DEPTH_REUSE 1u
#define GPU_TRACE_DEPTH_INFER 2u
#define GPU_TRACE_SUBTITLE_SUPPRESSED 0u
#define GPU_TRACE_SUBTITLE_OPTIONAL_OCR 1u
#define GPU_TRACE_SUBTITLE_ABSTENTION 2u
#define GPU_TRACE_SUBTITLE_HELD_WITH_DEPTH 5u
#define GPU_TRACE_SUBTITLE_INVALID 6u
#define GPU_TRACE_HOST_SUPPRESSED 0u
#define GPU_TRACE_HOST_ORDINARY_RECORD 1u
#define GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED (1u << 2u)
#define GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED (1u << 3u)
#define GPU_TRACE_FLAG_CONDITION_EXECUTED (1u << 4u)
#define GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED (1u << 5u)
#define GPU_TRACE_KNOWN_FLAGS 0x3fu

#define GPU_TRACE_DECISION_COOKIE 0xD1EC15A5u
#define GPU_TRACE_TOKEN_LOW_COOKIE 0xA3756C91u
#define GPU_TRACE_TOKEN_HIGH_COOKIE 0x5C8A936Eu
#define GPU_TRACE_PROPOSAL_MAGIC 0x504F5250u
#define GPU_TRACE_RECEIPT_MAGIC 0x47524243u
#define GPU_TRACE_REQUEST_MAGIC 0x54535152u
#define GPU_TRACE_OPTIONAL_RECEIPT_MAGIC 0x52434F4Fu
#define GPU_TRACE_WORK_FLAGS_COOKIE 0x6F435257u
#define GPU_TRACE_WORK_OPTIONAL_OCR (1u << 0u)
#define GPU_TRACE_WORK_SUBTITLE_OBSERVATION (1u << 1u)
#define GPU_TRACE_WORK_OPTIONAL_OCR_DUE (1u << 3u)
#define GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE (1u << 4u)

#if V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT != GPU_TRACE_LOCATOR_WORDS
#error GPU trace locator word count is stale
#endif
#if V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT != GPU_TRACE_CONDITION_WORDS
#error GPU trace condition word count is stale
#endif

uint TraceTransactionWord(uint index) {
    return TraceTransaction.Load(index * 4u);
}

bool TraceWorkValid(uint work) {
    return work == 0u || work == GPU_TRACE_WORK_OPTIONAL_OCR ||
        work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION ||
        work == GPU_TRACE_WORK_OPTIONAL_OCR_DUE ||
        work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE;
}

bool TraceHeaderValid() {
    bool valid = TraceRing[GPU_TRACE_HEADER_SCHEMA] == GPU_TRACE_RING_SCHEMA &&
        TraceRing[GPU_TRACE_HEADER_TAG] == GPU_TRACE_RING_TAG &&
        TraceRing[GPU_TRACE_HEADER_CAPACITY] == GPU_TRACE_CAPACITY &&
        TraceRing[GPU_TRACE_HEADER_RECORD_WORDS] == GPU_TRACE_RECORD_WORDS &&
        (TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_LOW] != 0u ||
         TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_HIGH] != 0u) &&
        TraceRing[GPU_TRACE_HEADER_NEXT_SLOT] < GPU_TRACE_CAPACITY &&
        TraceRing[GPU_TRACE_HEADER_COMMITTED_COUNT] <= GPU_TRACE_CAPACITY;
    [unroll]
    for (uint word = GPU_TRACE_HEADER_RESERVED_BEGIN;
         word < GPU_TRACE_HEADER_WORDS; ++word) {
        valid = valid && TraceRing[word] == 0u;
    }
    return valid;
}

bool TraceRequestValid() {
    uint token_low = TraceTransactionWord(8u);
    uint token_high = TraceTransactionWord(9u);
    uint work = TraceTransactionWord(13u);
    return
        (trace_transaction_token_low != 0u || trace_transaction_token_high != 0u) &&
        TraceWorkValid(trace_expected_work) &&
        token_low == trace_transaction_token_low &&
        token_high == trace_transaction_token_high &&
        TraceTransactionWord(10u) == (token_low ^ GPU_TRACE_TOKEN_LOW_COOKIE) &&
        TraceTransactionWord(11u) == (token_high ^ GPU_TRACE_TOKEN_HIGH_COOKIE) &&
        TraceTransactionWord(12u) == GPU_TRACE_REQUEST_MAGIC &&
        work == trace_expected_work && TraceWorkValid(work) &&
        TraceTransactionWord(14u) ==
            (work == 0u ? 0u : work ^ GPU_TRACE_WORK_FLAGS_COOKIE) &&
        TraceTransactionWord(15u) == 0u;
}

bool TraceReceiptValid(out bool optional_ocr, out uint decision) {
    optional_ocr = false;
    decision = TraceTransactionWord(0u);
    uint optional_marker = TraceTransactionWord(7u);
    uint token_low = TraceTransactionWord(2u);
    uint token_high = TraceTransactionWord(3u);
    bool request_valid = TraceRequestValid();
    bool receipt_valid = request_valid &&
        (trace_submission_class == GPU_TRACE_CLASS_FORCE_INFER ||
         trace_submission_class == GPU_TRACE_CLASS_GPU_UNDECIDED) &&
        (decision == 0u || decision == 1u) &&
        !(trace_submission_class == GPU_TRACE_CLASS_FORCE_INFER && decision == 0u) &&
        TraceTransactionWord(1u) ==
            (decision ^ GPU_TRACE_DECISION_COOKIE ^ optional_marker) &&
        token_low == trace_transaction_token_low &&
        token_high == trace_transaction_token_high &&
        TraceTransactionWord(4u) == (token_low ^ GPU_TRACE_TOKEN_LOW_COOKIE) &&
        TraceTransactionWord(5u) == (token_high ^ GPU_TRACE_TOKEN_HIGH_COOKIE) &&
        TraceTransactionWord(6u) == GPU_TRACE_RECEIPT_MAGIC &&
        (optional_marker == 0u ||
         (optional_marker == GPU_TRACE_OPTIONAL_RECEIPT_MAGIC &&
          ((trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR && decision == 1u) ||
           trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR_DUE)));
    optional_ocr = receipt_valid && optional_marker == GPU_TRACE_OPTIONAL_RECEIPT_MAGIC;
    return receipt_valid;
}

uint TraceSubtitleDisposition(bool receipt_valid, uint decision, bool optional_ocr) {
    if ((trace_flags & ~GPU_TRACE_KNOWN_FLAGS) != 0u) {
        return GPU_TRACE_SUBTITLE_INVALID;
    }
    bool record_submitted =
        (trace_flags & GPU_TRACE_FLAG_OCR_RECORD_SUBMITTED) != 0u;
    bool suppressed =
        (trace_flags & GPU_TRACE_FLAG_SUBTITLE_SUPPRESSED) != 0u;
    bool conditioned =
        (trace_flags & GPU_TRACE_FLAG_CONDITION_EXECUTED) != 0u;
    bool branch_gated =
        (trace_flags & GPU_TRACE_FLAG_SUBTITLE_BRANCH_GATED) != 0u;
    if (branch_gated) {
        bool infer_coupled =
            (trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR ||
             trace_expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION) &&
            trace_host_subtitle_outcome == GPU_TRACE_HOST_ORDINARY_RECORD;
        bool cadence_due =
            (trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR_DUE ||
             trace_expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE) &&
            trace_host_subtitle_outcome == GPU_TRACE_HOST_ORDINARY_RECORD;
        if (!receipt_valid || trace_expected_work == 0u || suppressed ||
            record_submitted || conditioned || (!infer_coupled && !cadence_due)) {
            return GPU_TRACE_SUBTITLE_INVALID;
        }
        if (infer_coupled && decision == 0u) {
            return !optional_ocr ? GPU_TRACE_SUBTITLE_HELD_WITH_DEPTH :
                GPU_TRACE_SUBTITLE_INVALID;
        }
        if (decision != 1u && !(cadence_due && decision == 0u)) {
            return GPU_TRACE_SUBTITLE_INVALID;
        }
        if (trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR || cadence_due) {
            return optional_ocr ? GPU_TRACE_SUBTITLE_OPTIONAL_OCR :
                GPU_TRACE_SUBTITLE_ABSTENTION;
        }
        if (trace_expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION) {
            return GPU_TRACE_SUBTITLE_ABSTENTION;
        }
        return GPU_TRACE_SUBTITLE_INVALID;
    }
    if (trace_expected_work == 0u) {
        return trace_host_subtitle_outcome == GPU_TRACE_HOST_SUPPRESSED && suppressed &&
            !record_submitted && !conditioned ?
            GPU_TRACE_SUBTITLE_SUPPRESSED : GPU_TRACE_SUBTITLE_INVALID;
    }
    if (trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR ||
        trace_expected_work == GPU_TRACE_WORK_OPTIONAL_OCR_DUE ||
        trace_expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION_DUE) {
        if (trace_host_subtitle_outcome != GPU_TRACE_HOST_ORDINARY_RECORD || suppressed ||
            !record_submitted || !conditioned) {
            return GPU_TRACE_SUBTITLE_INVALID;
        }
        return optional_ocr ?
            GPU_TRACE_SUBTITLE_OPTIONAL_OCR : GPU_TRACE_SUBTITLE_ABSTENTION;
    }
    if (trace_expected_work == GPU_TRACE_WORK_SUBTITLE_OBSERVATION) {
        return trace_host_subtitle_outcome == GPU_TRACE_HOST_ORDINARY_RECORD &&
            !suppressed && record_submitted && conditioned ?
            GPU_TRACE_SUBTITLE_ABSTENTION : GPU_TRACE_SUBTITLE_INVALID;
    }
    return GPU_TRACE_SUBTITLE_INVALID;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread : SV_DispatchThreadID) {
    bool header_valid = TraceHeaderValid();
    uint sequence_low = header_valid ?
        TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_LOW] : 1u;
    uint sequence_high = header_valid ?
        TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_HIGH] : 0u;
    uint slot = header_valid ? TraceRing[GPU_TRACE_HEADER_NEXT_SLOT] : 0u;
    uint count = header_valid ? TraceRing[GPU_TRACE_HEADER_COMMITTED_COUNT] : 0u;
    uint base = GPU_TRACE_HEADER_WORDS + slot * GPU_TRACE_RECORD_WORDS;

    // Invalidate the whole logical snapshot before overwriting a possibly still-referenced slot.
    // The tag is republished only after the new record and cursor are both complete.
    TraceRing[GPU_TRACE_HEADER_TAG] = 0u;
    DeviceMemoryBarrier();
    TraceRing[base + GPU_TRACE_RECORD_COMMIT_TAG] = 0u;
    [loop]
    for (uint word = 0u; word < GPU_TRACE_RECORD_WORDS; ++word) {
        if (word != GPU_TRACE_RECORD_COMMIT_TAG) {
            TraceRing[base + word] = 0u;
        }
    }

    bool optional_ocr = false;
    uint decision = 0u;
    bool receipt_valid = TraceReceiptValid(optional_ocr, decision);
    uint depth_disposition = !receipt_valid ? GPU_TRACE_DEPTH_INVALID :
        decision == 0u ? GPU_TRACE_DEPTH_REUSE : GPU_TRACE_DEPTH_INFER;
    uint subtitle_disposition =
        TraceSubtitleDisposition(receipt_valid, decision, optional_ocr);

    TraceRing[base + GPU_TRACE_RECORD_SCHEMA] = GPU_TRACE_RING_SCHEMA;
    TraceRing[base + GPU_TRACE_RECORD_SEQUENCE_LOW] = sequence_low;
    TraceRing[base + GPU_TRACE_RECORD_SEQUENCE_HIGH] = sequence_high;
    TraceRing[base + GPU_TRACE_RECORD_FRAME_LOW] = trace_frame_low;
    TraceRing[base + GPU_TRACE_RECORD_FRAME_HIGH] = trace_frame_high;
    TraceRing[base + GPU_TRACE_RECORD_ANALYSIS_GENERATION_LOW] =
        trace_analysis_generation_low;
    TraceRing[base + GPU_TRACE_RECORD_ANALYSIS_GENERATION_HIGH] =
        trace_analysis_generation_high;
    TraceRing[base + GPU_TRACE_RECORD_DOMAIN_TAG_LOW] = trace_domain_tag_low;
    TraceRing[base + GPU_TRACE_RECORD_DOMAIN_TAG_HIGH] = trace_domain_tag_high;
    TraceRing[base + GPU_TRACE_RECORD_TRANSACTION_TOKEN_LOW] =
        trace_transaction_token_low;
    TraceRing[base + GPU_TRACE_RECORD_TRANSACTION_TOKEN_HIGH] =
        trace_transaction_token_high;
    TraceRing[base + GPU_TRACE_RECORD_SUBMISSION_CLASS] = trace_submission_class;
    TraceRing[base + GPU_TRACE_RECORD_DEPTH_DISPOSITION] = depth_disposition;
    TraceRing[base + GPU_TRACE_RECORD_EXPECTED_WORK] = trace_expected_work;
    TraceRing[base + GPU_TRACE_RECORD_SUBTITLE_DISPOSITION] = subtitle_disposition;
    TraceRing[base + GPU_TRACE_RECORD_FLAGS] = trace_flags;
    TraceRing[base + GPU_TRACE_RECORD_HOST_SUBTITLE_OUTCOME] =
        trace_host_subtitle_outcome;
    TraceRing[base + GPU_TRACE_RECORD_SOURCE_WIDTH] = trace_source_width;
    TraceRing[base + GPU_TRACE_RECORD_SOURCE_HEIGHT] = trace_source_height;
    TraceRing[base + GPU_TRACE_RECORD_FIELD_WIDTH] = trace_field_width;
    TraceRing[base + GPU_TRACE_RECORD_FIELD_HEIGHT] = trace_field_height;
    TraceRing[base + GPU_TRACE_RECORD_TRANSACTION_WORD_COUNT] =
        GPU_TRACE_TRANSACTION_WORDS;
    TraceRing[base + GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_LOW] =
        trace_observation_timestamp_low;
    TraceRing[base + GPU_TRACE_RECORD_OBSERVATION_TIMESTAMP_HIGH] =
        trace_observation_timestamp_high;

    [unroll]
    for (uint word = 0u; word < GPU_TRACE_TRANSACTION_WORDS; ++word) {
        TraceRing[base + GPU_TRACE_RECORD_TRANSACTION_BEGIN + word] =
            TraceTransactionWord(word);
    }
    [loop]
    for (uint word = 0u; word < GPU_TRACE_LOCATOR_WORDS; ++word) {
        TraceRing[base + GPU_TRACE_RECORD_LOCATOR_BEGIN + word] =
            TraceSubtitleLocator[word];
    }
    [unroll]
    for (uint word = 0u; word < GPU_TRACE_CONDITION_WORDS; ++word) {
        TraceRing[base + GPU_TRACE_RECORD_CONDITION_BEGIN + word] =
            TraceSubtitleCondition[word];
    }
    DeviceMemoryBarrier();
    TraceRing[base + GPU_TRACE_RECORD_COMMIT_TAG] = GPU_TRACE_RECORD_TAG;
    DeviceMemoryBarrier();

    uint next_sequence_low = sequence_low + 1u;
    uint next_sequence_high = sequence_high + (next_sequence_low == 0u ? 1u : 0u);
    if (next_sequence_low == 0u && next_sequence_high == 0u) {
        next_sequence_low = 1u;
    }
    TraceRing[GPU_TRACE_HEADER_SCHEMA] = GPU_TRACE_RING_SCHEMA;
    TraceRing[GPU_TRACE_HEADER_CAPACITY] = GPU_TRACE_CAPACITY;
    TraceRing[GPU_TRACE_HEADER_RECORD_WORDS] = GPU_TRACE_RECORD_WORDS;
    TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_LOW] = next_sequence_low;
    TraceRing[GPU_TRACE_HEADER_NEXT_SEQUENCE_HIGH] = next_sequence_high;
    TraceRing[GPU_TRACE_HEADER_NEXT_SLOT] = (slot + 1u) % GPU_TRACE_CAPACITY;
    TraceRing[GPU_TRACE_HEADER_COMMITTED_COUNT] = min(count + 1u, GPU_TRACE_CAPACITY);
    [unroll]
    for (uint word = GPU_TRACE_HEADER_RESERVED_BEGIN;
         word < GPU_TRACE_HEADER_WORDS; ++word) {
        TraceRing[word] = 0u;
    }
    DeviceMemoryBarrier();
    TraceRing[GPU_TRACE_HEADER_TAG] = GPU_TRACE_RING_TAG;
}
