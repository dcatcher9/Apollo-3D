// Draws one eye of a packed left/right pair for the local shutter monitor and takes out the
// other eye's picture where the lens is open while the screen is still being redrawn.
//
// The monitor redraws line by line. A lens window longer than the time the whole screen shows
// one picture opens while the last lines still show the picture before, and closes after the
// first lines already show the picture after. Both belong to the other eye. On a line where
// that covers the share g of the window, the eye receives
//     (1 - g) * shown_own + g * shown_other
// For both eyes to receive what was meant, the pictures to show are
//     shown_own = ((1 - g) * meant_own - g * meant_other) / (1 - 2 g)
// A screen cannot show less than black, so a dark place beside a bright one in the other eye
// keeps some of it unless black is raised there; black_raise chooses how much of that is added.

Texture2D<float4> pair_texture : register(t0);

cbuffer local_shutter_eye_cbuffer : register(b0) {
    int own_left;       // first pair column of the eye being drawn
    int other_left;     // first pair column of the other eye
    int eye_height;
    int srgb_coded;     // the pictures carry the sRGB curve (SDR); otherwise linear scRGB (HDR)
    float top_peak;     // share of the window in which the first line already shows the next picture
    float top_extent;   // part of the height, from the top, that this reaches
    float bottom_peak;  // share of the window in which the last line still shows the picture before
    float bottom_extent;
    float black_raise;  // 0: none; 1: enough for white beside black
    float white_level;  // linear level of white
    int pattern;        // 0: the pair; 1: balance pattern; 2: bars pattern
    int drawing_left;   // the eye being drawn is the left one
    int view_left;      // the balance pattern is judged through the left eye
    int eye_width;
    float2 unused;
};

#include "include/common.hlsl"

// Largest share the correction is asked to undo. At one half the two pictures can no longer be
// told apart, and the division above has no answer.
static const float largest_share = 0.45;

float ghost_share(float height_position)
{
    const float top = top_extent > 0.0 ? top_peak * saturate(1.0 - height_position / top_extent) : 0.0;
    const float bottom = bottom_extent > 0.0 ? bottom_peak * saturate(1.0 - (1.0 - height_position) / bottom_extent) : 0.0;
    return clamp(top + bottom, 0.0, largest_share);
}

// What the pattern means each eye to see, as a part of white.
float3 pattern_picture(bool left_picture, int2 pixel)
{
    const int stripe = (pixel.x * 16) / max(eye_width, 1);
    if (pattern == 1) {
        // The judged eye gets an even grey. The other eye gets light and dark stripes. With the
        // right correction the grey stays even from top to bottom; stripes in it mean too
        // little (same sense as the other eye's) or too much (opposite sense) is taken out.
        // The levels leave room to correct shares up to about a quarter without raising black.
        // Short marks at half height, where nothing leaks, show the judged eye which columns
        // are the other eye's light stripes.
        if (left_picture == (view_left != 0)) {
            const float from_middle = abs((pixel.y + 0.5) / max(eye_height, 1) - 0.5);
            return ((stripe & 1) && from_middle < 0.004) ? float3(1.0, 1.0, 1.0) : float3(0.4, 0.4, 0.4);
        }
        return (stripe & 1) ? float3(1.0, 1.0, 1.0) : float3(0.15, 0.15, 0.15);
    }
    // Bars: white bars on black, in different places for the two eyes. Whatever shows between an
    // eye's own bars comes from the other eye.
    const bool bar = (stripe & 3) == (left_picture ? 0 : 2);
    return bar ? float3(1.0, 1.0, 1.0) : float3(0.0, 0.0, 0.0);
}

float3 meant_picture(bool left_picture, int first_column, int2 pixel)
{
    if (pattern != 0) {
        // An SDR screen has nothing above white, and a corrected light stripe needs to be
        // brighter than it was meant, so the patterns keep below white there.
        return pattern_picture(left_picture, pixel) * white_level * (srgb_coded ? 0.7 : 1.0);
    }
    const float3 stored = pair_texture.Load(int3(first_column + pixel.x, pixel.y, 0)).rgb;
    return srgb_coded ? RemoveSRGBCurve(stored) : stored;
}

float4 main_ps(float4 position : SV_Position) : SV_Target
{
    const int2 pixel = int2(position.xy);
    const bool left_picture = drawing_left != 0;
    const float3 own = meant_picture(left_picture, own_left, pixel);
    const float share = ghost_share((pixel.y + 0.5) / max(eye_height, 1));

    float3 shown = own;
    if (share > 0.0) {
        const float3 other = meant_picture(!left_picture, other_left, pixel);
        const float rest = 1.0 - 2.0 * share;
        const float raise = black_raise * white_level * share / rest;
        // Never below black, and never below a colour the picture itself carried under it.
        shown = max(((1.0 - share) * own - share * other) / rest + raise, min(own, 0.0));
    }
    return float4(srgb_coded ? ApplySRGBCurve(saturate(shown)) : shown, 1.0);
}
