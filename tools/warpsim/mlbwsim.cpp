// Offline validation of the planned MLBW composite shader math against iw3's own
// composite (the python reference). Per output pixel and layer:
//   field uv   : align_corners bilinear at fx = ex*(FW-1)/(EW-1)
//   src x      : sx = ex + delta*(EW-1)/(2*(FW/2-1))   (grid_sample align_corners units)
//   color      : bilinear(c, clamp(sx), ey), weighted by softmax layer weight
// Inputs from mlbw_fields.py / mlbw_ccrop.py. Usage: mlbwsim <left|right>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static const int FW = 798, FH = 336, L = 2;    // field dims / layers
static const int EW = 4096, EH = 1728;         // eye dims
static const int CX = 1950, CY = 550, CW = 460, CH = 340;  // ccrop region
static const int OX = 2000, OY = 560, OW = 360, OH = 320;  // output crop

static std::vector<float> g_delta, g_weight, g_c;

static float bilerp_field(const std::vector<float>& f, int layer, float x, float y) {
    x = std::max(0.0f, std::min((float) FW - 1, x));
    y = std::max(0.0f, std::min((float) FH - 1, y));
    int x0 = (int) x, y0 = (int) y;
    int x1 = std::min(FW - 1, x0 + 1), y1 = std::min(FH - 1, y0 + 1);
    float tx = x - x0, ty = y - y0;
    const float* p = f.data() + (size_t) layer * FW * FH;
    float a = p[y0 * FW + x0] * (1 - tx) + p[y0 * FW + x1] * tx;
    float b = p[y1 * FW + x0] * (1 - tx) + p[y1 * FW + x1] * tx;
    return a * (1 - ty) + b * ty;
}

static void sample_c(float ex, float ey, float* rgb) {
    // border padding: clamp in full-eye coords, then map into the crop
    ex = std::max(0.0f, std::min((float) EW - 1, ex));
    ey = std::max(0.0f, std::min((float) EH - 1, ey));
    float x = ex - CX, y = ey - CY;
    x = std::max(0.0f, std::min((float) CW - 1, x));
    y = std::max(0.0f, std::min((float) CH - 1, y));
    int x0 = (int) x, y0 = (int) y;
    int x1 = std::min(CW - 1, x0 + 1), y1 = std::min(CH - 1, y0 + 1);
    float tx = x - x0, ty = y - y0;
    for (int ch = 0; ch < 3; ch++) {
        float a = g_c[(y0 * CW + x0) * 3 + ch] * (1 - tx) + g_c[(y0 * CW + x1) * 3 + ch] * tx;
        float b = g_c[(y1 * CW + x0) * 3 + ch] * (1 - tx) + g_c[(y1 * CW + x1) * 3 + ch] * tx;
        rgb[ch] = a * (1 - ty) + b * ty;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mlbwsim <left|right>\n"); return 2; }
    const char* eye = argv[1];

    auto load = [](const char* p, std::vector<float>& v, size_t n) {
        FILE* f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
        v.resize(n);
        if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(1); }
        fclose(f);
    };
    char path[256];
    snprintf(path, sizeof(path), "fields/delta_%s_%dx%dx2.bin", eye, FW, FH);
    load(path, g_delta, (size_t) L * FW * FH);
    snprintf(path, sizeof(path), "fields/weight_%s_%dx%dx2.bin", eye, FW, FH);
    load(path, g_weight, (size_t) L * FW * FH);
    snprintf(path, sizeof(path), "fields/ccrop_x%dy%d_%dx%d.bin", CX, CY, CW, CH);
    load(path, g_c, (size_t) CW * CH * 3);

    const float px_scale = (float) (EW - 1) / (2.0f * (FW / 2 - 1));  // 4095/796

    std::vector<unsigned char> out((size_t) OW * OH * 3);
    for (int py = 0; py < OH; py++) {
        for (int px = 0; px < OW; px++) {
            float ex = (float) (OX + px), ey = (float) (OY + py);
            float fx = ex * (float) (FW - 1) / (float) (EW - 1);
            float fy = ey * (float) (FH - 1) / (float) (EH - 1);
            float acc[3] = {0, 0, 0};
            for (int i = 0; i < L; i++) {
                float d = bilerp_field(g_delta, i, fx, fy);
                float w = bilerp_field(g_weight, i, fx, fy);
                float rgb[3];
                sample_c(ex + d * px_scale, ey, rgb);
                for (int ch = 0; ch < 3; ch++) acc[ch] += w * rgb[ch];
            }
            for (int ch = 0; ch < 3; ch++) {
                float v = std::max(0.0f, std::min(1.0f, acc[ch]));
                out[((size_t) py * OW + px) * 3 + ch] = (unsigned char) lroundf(v * 255.0f);
            }
        }
    }

    snprintf(path, sizeof(path), "fields/sim_%s_%dx%d.bin", eye, OW, OH);
    FILE* f = fopen(path, "wb");
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    printf("%s -> %s\n", eye, path);
    return 0;
}
