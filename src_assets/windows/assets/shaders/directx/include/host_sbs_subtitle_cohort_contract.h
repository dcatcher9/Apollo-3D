#ifndef HOST_SBS_SUBTITLE_COHORT_CONTRACT_H
#define HOST_SBS_SUBTITLE_COHORT_CONTRACT_H

// Shared C++/HLSL ABI for the private, withdrawal-only native cohort qualifier.
// This is not a drawing or current-OCR authority. Unknown state uses baseline S.
// Schema 5 separates immutable known births from local UNKNOWN support seeds.
// Previous support is the exact unfiltered baseline S paired with previous RGB/P/SCT.
// Acquisition stays strict, retirement tolerant; observation.w remains zero.
#define SBC_SCHEMA 5u
#define SBC_TAG 0x31434253u
#define SBC_CAPACITY 8u
#define SBC_HEADER_WORDS 32u
#define SBC_ENTRY_WORDS 32u
#define SBC_STATE_WORDS (SBC_HEADER_WORDS + SBC_CAPACITY * SBC_ENTRY_WORDS)
#define SBC_CONSTANT_WORDS 24u
#define SBC_PREVIOUS_BASELINE_SUPPORT_SRV_SLOT 14u
#define SBC_OWNER_UNKNOWN_BIT 0x80000000u
#define SBC_OWNER_ID_MASK 0x7fffffffu
#define SBC_OWNER_ID_LIMIT 0x80000000u

#define SBC_H_TAG 0u
#define SBC_H_SCHEMA 1u
#define SBC_H_VALID 2u
#define SBC_H_CUT 3u
#define SBC_H_FRAME 4u
#define SBC_H_GENERATION 6u
#define SBC_H_CLOCK 8u
#define SBC_H_SOURCE 10u
#define SBC_H_CROP 12u
#define SBC_H_FIELD 14u
#define SBC_H_CONTENT 16u
#define SBC_H_REFERENCE 20u
#define SBC_H_NEXT_ID 22u
#define SBC_H_EPOCH 23u
#define SBC_H_COUNT 24u
#define SBC_H_FLAGS 25u
#define SBC_H_RESERVED 26u

#define SBC_E_ID 0u
#define SBC_E_PHASE 1u
#define SBC_E_FLAGS 2u
#define SBC_E_KIND 3u
#define SBC_E_BIRTH_CORE 4u
#define SBC_E_BIRTH_COVER 8u
#define SBC_E_LAST_CORE 12u
#define SBC_E_LAST_COVER 16u
#define SBC_E_BIRTH_FRAME 20u
#define SBC_E_BIRTH_CLOCK 22u
#define SBC_E_PREVIOUS_SLOT 24u
#define SBC_E_STABLE_ANCHORS 25u
#define SBC_E_TESTED_ANCHORS 26u
#define SBC_E_REFERENCE_FRAME 27u
#define SBC_E_REFERENCE_FLAGS 29u
#define SBC_E_RESERVED 30u

#define SBC_PHASE_UNKNOWN 0u
#define SBC_PHASE_PENDING 1u
#define SBC_PHASE_ACTIVE 2u
#define SBC_PHASE_SUSPENDED 3u
#define SBC_INVALID_SLOT 0xffffffffu
#define SBC_ENTRY_INVALID_REFERENCE 1u
#define SBC_ENTRY_AMBIGUOUS_PIXELS 2u
#define SBC_ENTRY_UNCERTAIN_REPLACEMENT 4u
#define SBC_ENTRY_NEW_BIRTH 8u
#define SBC_ENTRY_BLOCKING_FLAGS 7u
#define SBC_ENTRY_KNOWN_FLAGS 15u
#define SBC_HEADER_DUPLICATE 1u
#define SBC_HEADER_CURRENT_INVALID 2u
#define SBC_HEADER_OLD_VALID 4u
#define SBC_HEADER_PREVIOUS_SNAPSHOT_VALID 8u
#define SBC_HEADER_P_MAP_VALID 16u
#define SBC_HEADER_AMBIGUOUS_ASSOCIATION 32u
#define SBC_HEADER_ID_EXHAUSTED 64u
#define SBC_HEADER_KNOWN_FLAGS 127u

// RGB occupies the low three bytes. Owner zero, rather than RGB zero, means no birth.
// The separate anchor bit records an OWN first-observation P/edge witness, not age.
#define SBC_BIRTH_RGB_MASK 0x00ffffffu
#define SBC_BIRTH_ANCHOR_BIT 0x01000000u
#define SBC_BIRTH_KNOWN_MASK 0x01ffffffu
// UNKNOWN owns only a fixed local RGB reference, never an acquisition anchor or age.
#define SBC_UNKNOWN_RGB_MASK SBC_BIRTH_RGB_MASK
#define SBC_ACQUISITION_RGB8_MAX_DELTA 1u
#define SBC_REFERENCE_PREVIOUS_P_QUIET 1u
#define SBC_REFERENCE_IDENTITY_COORDINATES 2u
#define SBC_REFERENCE_KNOWN_FLAGS 3u

#endif
