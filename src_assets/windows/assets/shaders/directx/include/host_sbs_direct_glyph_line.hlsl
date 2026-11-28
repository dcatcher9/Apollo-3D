#ifndef HOST_SBS_DIRECT_GLYPH_LINE
#define HOST_SBS_DIRECT_GLYPH_LINE
static const uint NoGlyphInk = 0xffffffffu;
struct GlyphLine {
    uint4 base;       // Current OCR work before same-cue geometry carry
    uint4 box;        // Actual inference rectangle
    uint4 cover;      // Original detector cap
    uint4 state;      // Previous slot + 1, raw alpha offset, temporal state bits, palette state.
                      // Before palette selection, w requests glyph confirmation. After selection
                      // it contains packed fill/outline indices or NoGlyphInk for a rejected line.
};
// state.z: the score agreed with the previous one and is blended with it; the line was carried
// over from the previous observation without a score; the line asked for glyph confirmation.
static const uint LineConfirmed = 1u;
static const uint LineCarried = 2u;
static const uint LineAsked = 4u;
float3 PalColor(uint i) {
    return min(255.0f, 16 * float3(i / 289, (i / 17) % 17, i % 17));
}
#endif
