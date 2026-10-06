// FastConformer inference for Intel macOS: compact linears plus Accelerate BLAS.
// This implements the published Parakeet inference graph, without training paths.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" int p2_gemm(void *, const float *, int, float *) noexcept;

namespace {
using Array = std::vector<float>;

struct Layer {
    void *matrices[11] = {};
    Array vectors[17];
    bool ready = false;
};

struct Encoder {
    int depth, width, inner, heads, kernel, mels, channels;
    Array sub[13];
    std::vector<Layer> layers;
    bool ready = false;
};

// Row-major Y = X W^T; used only for the small float subsampling operators.
Array linear(const Array &x, int rows, int inputs, int outputs, const Array &w, const Array &bias) {
    Array out(static_cast<size_t>(rows) * outputs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, outputs, inputs,
                1, x.data(), inputs, w.data(), inputs, 0, out.data(), outputs);
    if (!bias.empty()) for (int r = 0; r < rows; ++r) for (int k = 0; k < outputs; ++k) out[r * outputs + k] += bias[k];
    return out;
}

Array packed(void *matrix, const Array &x, int rows, int outputs) {
    Array out(static_cast<size_t>(rows) * outputs);
    if (p2_gemm(matrix, x.data(), rows, out.data()) != 0) throw std::runtime_error("packed encoder multiply failed");
    return out;
}

Array norm(const Array &x, int width, const Array &weight, const Array &bias) {
    Array out(x.size());
    for (size_t row = 0; row < x.size(); row += width) {
        float mean = 0, variance = 0;
        for (int k = 0; k < width; ++k) mean += x[row + k];
        mean /= width;
        for (int k = 0; k < width; ++k) { float d = x[row + k] - mean; variance += d * d; }
        float inverse = 1 / std::sqrt(variance / width + 1e-5f);
        for (int k = 0; k < width; ++k) out[row + k] = (x[row + k] - mean) * inverse * weight[k] + bias[k];
    }
    return out;
}

// vForce evaluates exp in SIMD batches rather than a scalar libm call per value.
void silu(Array &x) {
    Array exp(x.size());
    for (size_t i = 0; i < x.size(); ++i) exp[i] = -x[i];
    int count = static_cast<int>(x.size());
    vvexpf(exp.data(), exp.data(), &count);
    for (size_t i = 0; i < x.size(); ++i) x[i] /= 1 + exp[i];
}

void residual(Array &x, const Array &update, float factor = 1) {
    for (size_t i = 0; i < x.size(); ++i) x[i] += factor * update[i];
}

// Spatial convolution uses time/frequency/channel storage. Weight order remains
// PyTorch's [out, in, kernel-time, kernel-frequency] checkpoint order.
Array spatial(const Array &x, int &time, int &frequency, int inputs, int outputs,
              const Array &weights, const Array &bias, bool depthwise) {
    int ot = (time + 1) / 2, of = (frequency + 1) / 2;
    Array out(static_cast<size_t>(ot) * of * outputs);
    if (depthwise) {
        for (int t = 0; t < ot; ++t) for (int f = 0; f < of; ++f) {
            float *y = out.data() + (t * of + f) * outputs;
            std::copy(bias.begin(), bias.end(), y);
            for (int kt = 0; kt < 3; ++kt) for (int kf = 0; kf < 3; ++kf) {
                int it = t * 2 + kt - 1, jf = f * 2 + kf - 1;
                if (it < 0 || it >= time || jf < 0 || jf >= frequency) continue;
                const float *a = x.data() + (it * frequency + jf) * inputs;
                for (int c = 0; c < inputs; ++c) y[c] += a[c] * weights[c * 9 + kt * 3 + kf];
            }
        }
    } else {
        Array patches(static_cast<size_t>(ot) * of * inputs * 9, 0);
        for (int t = 0; t < ot; ++t) for (int f = 0; f < of; ++f)
            for (int c = 0; c < inputs; ++c) for (int kt = 0; kt < 3; ++kt) for (int kf = 0; kf < 3; ++kf) {
                int it = t * 2 + kt - 1, jf = f * 2 + kf - 1;
                if (it >= 0 && it < time && jf >= 0 && jf < frequency)
                    patches[((t * of + f) * inputs + c) * 9 + kt * 3 + kf] = x[(it * frequency + jf) * inputs + c];
            }
        out = linear(patches, ot * of, inputs * 9, outputs, weights, bias);
    }
    time = ot; frequency = of;
    return out;
}

Array subsample(const Encoder &e, const float *input, int &time) {
    int frequency = e.mels, channels = e.channels;
    Array x(input, input + static_cast<size_t>(time) * frequency);
    x = spatial(x, time, frequency, 1, channels, e.sub[0], e.sub[1], false);
    for (float &v : x) v = std::max(0.0f, v);
    for (int layer = 0; layer < 2; ++layer) {
        int at = 2 + layer * 4;
        x = spatial(x, time, frequency, channels, channels, e.sub[at], e.sub[at + 1], true);
        x = linear(x, time * frequency, channels, channels, e.sub[at + 2], e.sub[at + 3]);
        for (float &v : x) v = std::max(0.0f, v);
    }
    // The projection flattens [channel, frequency], not [frequency, channel].
    Array flattened(x.size());
    for (int t = 0; t < time; ++t) for (int c = 0; c < channels; ++c) for (int f = 0; f < frequency; ++f)
        flattened[(t * channels + c) * frequency + f] = x[(t * frequency + f) * channels + c];
    return linear(flattened, time, channels * frequency, e.width, e.sub[10], e.sub[11]);
}

Array positions(const Encoder &e, int time) {
    Array out(static_cast<size_t>(2 * time - 1) * e.width);
    for (int t = 0; t < 2 * time - 1; ++t) for (int k = 0; k < e.width / 2; ++k) {
        float angle = (time - 1 - t) * e.sub[12][k];
        out[t * e.width + 2 * k] = std::sin(angle);
        out[t * e.width + 2 * k + 1] = std::cos(angle);
    }
    return out;
}

void feed_forward(const Encoder &e, const Layer &l, Array &x, int time, int matrix, int vector) {
    Array y = norm(x, e.width, l.vectors[vector], l.vectors[vector + 1]);
    y = packed(l.matrices[matrix], y, time, e.inner);
    silu(y);
    y = packed(l.matrices[matrix + 1], y, time, e.width);
    residual(x, y, 0.5f);
}

void attention(const Encoder &e, const Layer &l, Array &x, const Array &pos, int time) {
    int d = e.width, h = d / e.heads, relative = 2 * time - 1;
    Array y = norm(x, d, l.vectors[2], l.vectors[3]);
    Array q = packed(l.matrices[2], y, time, d), k = packed(l.matrices[3], y, time, d), v = packed(l.matrices[4], y, time, d);
    Array r = packed(l.matrices[6], pos, relative, d), combined(x.size());
    Array qu(static_cast<size_t>(time) * h), qv(qu.size()), ac(static_cast<size_t>(time) * time), bd(static_cast<size_t>(time) * relative);
    float scale = 1 / std::sqrt(static_cast<float>(h));
    for (int head = 0; head < e.heads; ++head) {
        int offset = head * h;
        for (int t = 0; t < time; ++t) for (int j = 0; j < h; ++j) {
            qu[t * h + j] = q[t * d + offset + j] + l.vectors[10][offset + j];
            qv[t * h + j] = q[t * d + offset + j] + l.vectors[11][offset + j];
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, time, time, h, scale, qu.data(), h, k.data() + offset, d, 0, ac.data(), time);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, time, relative, h, scale, qv.data(), h, r.data() + offset, d, 0, bd.data(), relative);
        // This index is the HF relative-shift operation with its padding removed.
        for (int i = 0; i < time; ++i) {
            for (int j = 0; j < time; ++j) ac[i * time + j] += bd[i * relative + time - 1 - i + j];
            float maximum = *std::max_element(ac.begin() + i * time, ac.begin() + (i + 1) * time);
            for (int j = 0; j < time; ++j) ac[i * time + j] -= maximum;
        }
        int count = time * time;
        vvexpf(ac.data(), ac.data(), &count);
        for (int i = 0; i < time; ++i) {
            float total = 0;
            for (int j = 0; j < time; ++j) total += ac[i * time + j];
            for (int j = 0; j < time; ++j) ac[i * time + j] /= total;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, time, h, time, 1, ac.data(), time, v.data() + offset, d, 0, combined.data() + offset, d);
    }
    residual(x, packed(l.matrices[5], combined, time, d));
}

void convolution(const Encoder &e, const Layer &l, Array &x, int time) {
    int d = e.width;
    Array y = norm(x, d, l.vectors[4], l.vectors[5]);
    y = packed(l.matrices[7], y, time, 2 * d);
    Array gates(x.size()), glu(x.size());
    for (int t = 0; t < time; ++t) for (int k = 0; k < d; ++k) gates[t * d + k] = -y[t * 2 * d + d + k];
    int count = static_cast<int>(gates.size());
    vvexpf(gates.data(), gates.data(), &count);
    for (int t = 0; t < time; ++t) for (int k = 0; k < d; ++k) glu[t * d + k] = y[t * 2 * d + k] / (1 + gates[t * d + k]);
    Array depth(x.size(), 0);
    for (int t = 0; t < time; ++t) for (int j = 0; j < e.kernel; ++j) {
        int input = t + j - e.kernel / 2;
        if (input < 0 || input >= time) continue;
        for (int k = 0; k < d; ++k) depth[t * d + k] += glu[input * d + k] * l.vectors[12][k * e.kernel + j];
    }
    for (int k = 0; k < d; ++k) {
        float gain = l.vectors[13][k] / std::sqrt(l.vectors[16][k] + 1e-5f);
        float bias = l.vectors[14][k] - l.vectors[15][k] * gain;
        for (int t = 0; t < time; ++t) depth[t * d + k] = depth[t * d + k] * gain + bias;
    }
    silu(depth);
    residual(x, packed(l.matrices[8], depth, time, d));
}
} // namespace

extern "C" {
void *p2_encoder_create(int depth, int width, int inner, int heads, int kernel, int mels, int channels) noexcept {
    if (depth < 1 || depth > 64 || width < 1 || width > 4096 || heads < 1 || width % heads || width % 2 ||
        inner < 1 || inner > 16384 || kernel < 1 || kernel % 2 == 0 || mels < 8 || mels % 8 || channels < 1) return nullptr;
    try {
        auto e = std::make_unique<Encoder>();
        e->depth = depth; e->width = width; e->inner = inner; e->heads = heads;
        e->kernel = kernel; e->mels = mels; e->channels = channels; e->layers.resize(depth);
        return e.release();
    } catch (...) { return nullptr; }
}

void p2_encoder_destroy(void *handle) noexcept { delete static_cast<Encoder *>(handle); }

// Python validates the exact tensor lengths before handing these borrowed pointers in.
int p2_encoder_sub(void *handle, const float *const *arrays) noexcept {
    if (!handle || !arrays) return -1;
    try {
        auto &e = *static_cast<Encoder *>(handle);
        int c = e.channels, d = e.width;
        int sizes[] = {c * 9, c, c * 9, c, c * c, c, c * 9, c, c * c, c, d * c * (e.mels / 8), d, d / 2};
        for (int i = 0; i < 13; ++i) { if (!arrays[i]) return -1; e.sub[i].assign(arrays[i], arrays[i] + sizes[i]); }
        e.ready = true;
        return 0;
    } catch (...) { return -2; }
}

int p2_encoder_layer(void *handle, int layer, void *const *matrices, const float *const *vectors) noexcept {
    if (!handle || !matrices || !vectors) return -1;
    try {
        auto &e = *static_cast<Encoder *>(handle);
        if (layer < 0 || layer >= e.depth) return -1;
        auto &l = e.layers[layer];
        for (int i = 0; i < 11; ++i) { if (!matrices[i]) return -1; l.matrices[i] = matrices[i]; }
        for (int i = 0; i < 17; ++i) {
            if (!vectors[i]) return -1;
            int size = i == 12 ? e.width * e.kernel : e.width;
            l.vectors[i].assign(vectors[i], vectors[i] + size);
        }
        l.ready = true;
        return 0;
    } catch (...) { return -2; }
}

// Stop after selected layers for direct encoder-boundary parity tests.
int p2_encoder_forward(void *handle, const float *features, int frames, int layers, float *output) noexcept {
    if (!handle || !features || !output || frames < 1 || frames > 3001) return -1;
    try {
        const auto &e = *static_cast<Encoder *>(handle);
        if (!e.ready || layers < 0 || layers > e.depth) return -1;
        for (int i = 0; i < layers; ++i) if (!e.layers[i].ready) return -1;
        Array x = subsample(e, features, frames), pos = positions(e, frames);
        for (int i = 0; i < layers; ++i) {
            const auto &l = e.layers[i];
            feed_forward(e, l, x, frames, 0, 0);
            attention(e, l, x, pos, frames);
            convolution(e, l, x, frames);
            feed_forward(e, l, x, frames, 9, 6);
            x = norm(x, e.width, l.vectors[8], l.vectors[9]);
        }
        std::copy(x.begin(), x.end(), output);
        return frames;
    } catch (...) { return -2; }
}
} // extern C
