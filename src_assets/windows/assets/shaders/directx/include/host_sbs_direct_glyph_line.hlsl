#ifndef HOST_SBS_DIRECT_GLYPH_LINE
#define HOST_SBS_DIRECT_GLYPH_LINE
struct GlyphLine {
    uint4 base;       // Current OCR work before same-cue geometry carry
    uint4 box;        // Actual inference rectangle
    uint4 cover;      // Original detector cap
    uint4 state;      // Previous slot + 1, raw alpha offset, confirmed, reserved
};
#endif
