#ifndef LUT_EXPANSIONS_ONLY
#define LUT_EXPANSIONS_ONLY 0
#endif
// Intel AVX2 kernels for the published fermion-five-value-parakeet-v1 format.
// Weights remain integer codes; no dense float32 weight matrix is constructed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dispatch/dispatch.h>
#include <immintrin.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Matrix {
    int rows, cols, stride, threads;
    bool exact;
    std::vector<uint16_t> codes;
    std::vector<int> magnitudes;
    std::vector<int8_t> q;
    std::vector<uint8_t> a, b;
    std::vector<float> scale, low;
};

struct TritByte {
    int8_t sign[5]{};
    uint8_t nonzero[6]{};
};

// The wire format has just 243 valid bytes. Prefix counts locate magnitude bits
// without a per-weight branch and also handle partially filled final bytes.
constexpr auto trit_table() {
    std::array<TritByte, 243> table{};
    for (unsigned code = 0; code < table.size(); ++code) {
        unsigned value = code;
        for (int j = 0; j < 5; ++j, value /= 3) {
            table[code].sign[j] = static_cast<int>(value % 3) - 1;
            table[code].nonzero[j + 1] = table[code].nonzero[j] + (value % 3 != 1);
        }
    }
    return table;
}
constexpr auto TRITS = trit_table();

// Expand bitmap positions once for the wire alphabet, not once per weight.
struct DecodeBytes {
    uint64_t signs = 0;
    std::array<uint64_t,32> masks{};
};
constexpr auto decode_bytes() {
    std::array<DecodeBytes,243> table{};
    for (unsigned code = 0; code < table.size(); ++code) {
        const auto &trit = TRITS[code];
        for (int j = 0; j < 5; ++j) {
            table[code].signs |= static_cast<uint64_t>(static_cast<uint8_t>(trit.sign[j])) << (8*j);
            for (unsigned bits = 0; bits < 32; ++bits) {
                if ((bits >> trit.nonzero[j]) & 1)
                    table[code].masks[bits] |= UINT64_C(255) << (8*j);
            }
        }
    }
    return table;
}
constexpr auto DECODE_BYTES = decode_bytes();


// Read unaligned IEEE binary16 values without aliasing or alignment assumptions.
float half(const uint8_t *p) {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return _cvtsh_ss(bits);
}

// Four 2-bit codes per byte become four signed bytes in each 32-bit lane.
__m256i unpack(const uint8_t *p) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p)));
    __m256i w = _mm256_and_si256(v, _mm256_set1_epi32(3));
    w = _mm256_or_si256(w, _mm256_and_si256(_mm256_slli_epi32(v, 6), _mm256_set1_epi32(0x300)));
    w = _mm256_or_si256(w, _mm256_and_si256(_mm256_slli_epi32(v, 12), _mm256_set1_epi32(0x30000)));
    w = _mm256_or_si256(w, _mm256_and_si256(_mm256_slli_epi32(v, 18), _mm256_set1_epi32(0x3000000)));
    return _mm256_sub_epi8(w, _mm256_set1_epi8(1));
}

// Both operands are in [-127,127], so maddubs' signed 16-bit pairs cannot saturate.
// Move activation signs onto the weights; abs(activation) is the unsigned operand.
__m256i dot32(__m256i x, __m256i w) {
    __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

int sum32(__m256i x) {
    __m128i v = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(v, v));
}

struct Work {
    const Matrix &w;
    const int8_t *x;
    const float *scales;
    float *y;
    int count;
};

// Reuse each activation load across output rows without changing integer sums.
// Tile dimensions and exactness are compile-time choices in the vector loop.
template<int Batch, bool Exact, int Outputs = 1>
void tile(const Work &work, int n, int m) {
    const auto &w = work.w;
    __m256i acc[Outputs][Batch] = {}, extra[Outputs][Batch] = {};
    for (int k = 0; k < w.stride; k += 32) {
        __m256i a[Outputs], b[Outputs] = {};
        #pragma clang loop unroll(full)
        for (int i = 0; i < Outputs; ++i) {
            if constexpr (Exact) {
                a[i] = unpack(w.a.data() + (static_cast<size_t>(n + i) * w.stride + k) / 4);
                b[i] = unpack(w.b.data() + (static_cast<size_t>(n + i) * w.stride + k) / 4);
            } else {
                a[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.q.data() + static_cast<size_t>(n + i) * w.stride + k));
            }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < Batch; ++j) {
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(work.x + static_cast<size_t>(m + j) * w.stride + k));
            #pragma clang loop unroll(full)
            for (int i = 0; i < Outputs; ++i) {
                acc[i][j] = _mm256_add_epi32(acc[i][j], dot32(x, a[i]));
                if constexpr (Exact) extra[i][j] = _mm256_add_epi32(extra[i][j], dot32(x, b[i]));
            }
        }
    }
    // Pairwise horizontal adds reduce four output rows together. Keep the two
    // float multiplies separate so scaling retains the scalar path's rounding.
    if constexpr (!Exact && Outputs == 4) {
        for (int j = 0; j < Batch; ++j) {
            __m256i ab = _mm256_hadd_epi32(acc[0][j], acc[1][j]);
            __m256i cd = _mm256_hadd_epi32(acc[2][j], acc[3][j]);
            __m256i sums = _mm256_hadd_epi32(ab, cd);
            __m128i totals = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            __m128 values = _mm_mul_ps(_mm_cvtepi32_ps(totals), _mm_loadu_ps(w.scale.data() + n));
            values = _mm_mul_ps(values, _mm_set1_ps(work.scales[m + j]));
            _mm_storeu_ps(work.y + static_cast<size_t>(m + j) * w.rows + n, values);
        }
    } else for (int i = 0; i < Outputs; ++i) for (int j = 0; j < Batch; ++j) {
        float value = static_cast<float>(sum32(acc[i][j])) * w.scale[n + i];
        if constexpr (Exact) value = static_cast<float>(sum32(acc[i][j])) * w.low[n + i] + static_cast<float>(sum32(extra[i][j])) * w.scale[n + i];
        work.y[static_cast<size_t>(m + j) * w.rows + n + i] = value * work.scales[m + j];
    }
}

// Compute one input tile against a bounded set of output rows.
template<int Batch, bool Exact>
void row_block(const Work &work, int begin, int end, int m) {
    int n = begin;
    if constexpr (!Exact) {
        for (; n + 4 <= end; n += 4) tile<Batch, false, 4>(work, n, m);
    }
    for (; n < end; ++n) tile<Batch, Exact>(work, n, m);
}

template<bool Exact>
void multiply_rows(void *context, size_t worker) {
    const auto &work = *static_cast<Work *>(context);
    const auto &w = work.w;
    int begin = w.rows * static_cast<int>(worker) / w.threads;
    int end = w.rows * static_cast<int>(worker + 1) / w.threads;
    // Bound reused weights to 24 rows while each two-input tile stays hot.
    // Two-plane weights retain the one-output/four-input register layout.
    constexpr int output_block = Exact ? 1 : 24, input_block = Exact ? 4 : 2;
    for (int base = begin; base < end; base += output_block) {
        int limit = std::min(base + output_block, end), m = 0;
        for (; m + input_block <= work.count; m += input_block) row_block<input_block, Exact>(work, base, limit, m);
        for (; m < work.count; ++m) row_block<1, Exact>(work, base, limit, m);
    }
}

// Experimental vector LUT: two ternary planes reproduce existing int8 weights.
#ifndef LUT_TRITS
#define LUT_TRITS 3
#endif
#ifndef LUT_GROUPS
#define LUT_GROUPS 32
#endif
#ifndef LUT_ROW_MAJOR
#define LUT_ROW_MAJOR 0
#endif
#ifndef LUT_BATCH
#define LUT_BATCH 32
#endif
#ifndef LUT_OUTPUT_BLOCK
#define LUT_OUTPUT_BLOCK 0
#endif
constexpr int lut_entries = LUT_TRITS == 2 ? 9 : LUT_TRITS == 3 ? 27 : LUT_TRITS == 4 ? 81 : 243;
constexpr int lut_batch = LUT_BATCH;
static_assert(LUT_GROUPS * LUT_TRITS * 127 <= 32767);

// Map the container's five trits and nonzero bitmap to both lookup indices.
constexpr auto five_lut_codes() {
    std::array<std::array<uint16_t,32>,243> result{};
    for (unsigned code = 0; code < 243; ++code) {
        for (unsigned mask = 0; mask < 32; ++mask) {
            unsigned a = 0, b = 0, power = 1;
            for (int j = 0; j < 5; ++j, power *= 3) {
                int sign = TRITS[code].sign[j];
                unsigned digit = sign < 0 ? 2 : sign;
                a += digit * power;
                if ((mask >> TRITS[code].nonzero[j]) & 1) b += digit * power;
            }
            result[code][mask] = a | (b << 8);
        }
    }
    return result;
}
static_assert(LUT_TRITS == 5 && LUT_ROW_MAJOR == 2 && LUT_OUTPUT_BLOCK == 0);
constexpr auto FIVE_LUT_CODES = five_lut_codes();

// Build sums for sixteen different frames, keeping every sum exact in int16.
void make_lut(const Work &work, int m, int g, int16_t *__restrict table) {
    for (int part = 0; part < lut_batch; part += 16)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(table+part), _mm256_setzero_si256());
    int power = 1;
    for (int j = 0; j < LUT_TRITS; ++j, power *= 3) {
        alignas(32) int16_t input[lut_batch]{};
        int k = g * LUT_TRITS + j;
        if (k < work.w.cols) for (int t = 0; t < lut_batch && m+t < work.count; ++t)
            input[t] = work.x[static_cast<size_t>(m+t) * work.w.stride + k];
        for (int c = 0; c < power; ++c) {
            for (int part = 0; part < lut_batch; part += 16) {
                __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i *>(input+part));
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(table+c*lut_batch+part));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(table+(c+power)*lut_batch+part), _mm256_add_epi16(v,x));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(table+(c+2*power)*lut_batch+part), _mm256_sub_epi16(v,x));
            }
        }
    }
}

// Convert a small square in registers instead of repeatedly striding over scratch.
void finish_lut(const Work &work, const int32_t *__restrict sums, int m, int begin, int end) {
    const auto &w = work.w;
    int valid = std::min(lut_batch, work.count-m), n = begin;
    for (; n+8 <= end; n += 8) {
        int t = 0;
        for (; t+8 <= valid; t += 8) {
            __m256 v[8], u[8], z[8];
            __m256 xs = _mm256_loadu_ps(work.scales+m+t);
            for (int i = 0; i < 8; ++i) {
                __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sums+static_cast<size_t>(n+i-begin)*lut_batch+t));
                v[i] = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(value), _mm256_set1_ps(w.scale[n+i])), xs);
            }
            for (int i = 0; i < 8; i += 2) {
                u[i] = _mm256_unpacklo_ps(v[i],v[i+1]);
                u[i+1] = _mm256_unpackhi_ps(v[i],v[i+1]);
            }
            for (int i = 0; i < 8; i += 4) {
                z[i] = _mm256_shuffle_ps(u[i],u[i+2],0x44);
                z[i+1] = _mm256_shuffle_ps(u[i],u[i+2],0xee);
                z[i+2] = _mm256_shuffle_ps(u[i+1],u[i+3],0x44);
                z[i+3] = _mm256_shuffle_ps(u[i+1],u[i+3],0xee);
            }
            for (int i = 0; i < 4; ++i) {
                _mm256_storeu_ps(work.y+static_cast<size_t>(m+t+i)*w.rows+n, _mm256_permute2f128_ps(z[i],z[i+4],0x20));
                _mm256_storeu_ps(work.y+static_cast<size_t>(m+t+i+4)*w.rows+n, _mm256_permute2f128_ps(z[i],z[i+4],0x31));
            }
        }
        for (; t < valid; ++t) for (int i = 0; i < 8; ++i)
            work.y[static_cast<size_t>(m+t)*w.rows+n+i] = (static_cast<float>(sums[static_cast<size_t>(n+i-begin)*lut_batch+t])*w.scale[n+i])*work.scales[m+t];
    }
    for (; n < end; ++n) for (int t = 0; t < valid; ++t)
        work.y[static_cast<size_t>(m+t)*w.rows+n] = (static_cast<float>(sums[static_cast<size_t>(n-begin)*lut_batch+t])*w.scale[n])*work.scales[m+t];
}

struct LookupWork {
    const Work &work;
    bool *failures;
};

// Frame partitions own disjoint outputs and report allocation failures to the caller.
void multiply_lut(void *context, size_t worker) {
    const auto &lookup = *static_cast<LookupWork *>(context);
    try {
    const auto &work = lookup.work;
    const auto &w = work.w;
    int groups = (w.cols + LUT_TRITS - 1) / LUT_TRITS;
    int row_block = LUT_OUTPUT_BLOCK ? LUT_OUTPUT_BLOCK : w.rows;
    std::vector<int16_t> table_storage(static_cast<size_t>(LUT_OUTPUT_BLOCK ? groups : LUT_GROUPS)*lut_entries*lut_batch);
    std::vector<int32_t> sum_storage(static_cast<size_t>(std::min(row_block,w.rows))*lut_batch);
    int16_t *__restrict tables = table_storage.data();
    int32_t *__restrict sums = sum_storage.data();
    int full_count = work.count / (w.threads*lut_batch) * (w.threads*lut_batch);
    for (int m = static_cast<int>(worker) * lut_batch; m < full_count; m += w.threads * lut_batch) {
        if (LUT_OUTPUT_BLOCK) for (int g = 0; g < groups; ++g)
            make_lut(work, m, g, tables+static_cast<size_t>(g)*lut_entries*lut_batch);
        for (int row = 0; row < w.rows; row += row_block) {
        int row_end = std::min(row+row_block,w.rows);
        std::fill(sums, sums+sum_storage.size(), 0);
        for (int base = 0; base < groups; base += LUT_GROUPS) {
            int limit = std::min(groups-base, LUT_GROUPS);
            if (!LUT_OUTPUT_BLOCK) for (int g = 0; g < limit; ++g)
                make_lut(work, m, base+g, tables+g*lut_entries*lut_batch);
            for (int n = row; n < row_end; ++n) {
                __m256i a[lut_batch/16]{}, b[lut_batch/16]{};
                for (int g = 0; g < limit; ++g) {
                    size_t at = LUT_ROW_MAJOR == 2 ? static_cast<size_t>(base/LUT_GROUPS)*w.rows*LUT_GROUPS+n*LUT_GROUPS+g :
                        LUT_ROW_MAJOR ? static_cast<size_t>(n)*groups+base+g : static_cast<size_t>(base+g)*w.rows+n;
                    unsigned codes = w.codes[at];
                    const int16_t *tab = tables+static_cast<size_t>(LUT_OUTPUT_BLOCK ? base+g : g)*lut_entries*lut_batch;
                    for (int part = 0; part < lut_batch/16; ++part) {
                        a[part] = _mm256_add_epi16(a[part], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tab+(codes & 255)*lut_batch+part*16)));
                        b[part] = _mm256_add_epi16(b[part], _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tab+(codes >> 8)*lut_batch+part*16)));
                    }
                }
                __m256i coefficients = _mm256_set1_epi32(w.magnitudes[2*n] | (w.magnitudes[2*n+1] << 16));
                for (int part = 0; part < lut_batch/16; ++part) {
                    __m256i low = _mm256_madd_epi16(_mm256_unpacklo_epi16(a[part],b[part]),coefficients);
                    __m256i high = _mm256_madd_epi16(_mm256_unpackhi_epi16(a[part],b[part]),coefficients);
                    for (int half = 0; half < 2; ++half) {
                        __m256i total = half ? _mm256_permute2x128_si256(low,high,0x31) : _mm256_permute2x128_si256(low,high,0x20);
                        auto *dst = reinterpret_cast<__m256i *>(sums+static_cast<size_t>(n-row)*lut_batch+part*16+half*8);
                        _mm256_storeu_si256(dst, _mm256_add_epi32(_mm256_loadu_si256(dst),total));
                    }
                }
            }
        }
        finish_lut(work, sums, m, row, row_end);
        }
    }
    // Incomplete frame batches retain the multiply kernel's balanced row split.
    if (full_count < work.count) {
        Work tail{w, work.x+static_cast<size_t>(full_count)*w.stride, work.scales+full_count,
                  work.y+static_cast<size_t>(full_count)*w.rows, work.count-full_count};
        multiply_rows<false>(&tail, worker);
    }
    } catch (const std::bad_alloc &) {
        lookup.failures[worker] = true;
    }
}

// Synchronous dispatch publishes every worker's failure before p2_gemm returns.
void run_lut(const Work &work) {
    const auto &w = work.w;
    bool failures[64]{};
    LookupWork lookup{work,failures};
    if (w.threads == 1) multiply_lut(&lookup,0);
    else dispatch_apply_f(w.threads,dispatch_get_global_queue(QOS_CLASS_UTILITY,0),&lookup,multiply_lut);
    if (std::find(failures,failures+w.threads,true) != failures+w.threads) throw std::bad_alloc();
}

// Quantize eight activations together, preserving round-to-nearest-even and
// rejecting nonfinite values before float-to-integer conversion.
float quantize(const float *input, int cols, int8_t *output) {
    __m256 maximum8 = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 infinity = _mm256_set1_ps(INFINITY);
    int k = 0;
    for (; k + 8 <= cols; k += 8) {
        __m256 value = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(input + k));
        if (_mm256_movemask_ps(_mm256_cmp_ps(value, infinity, _CMP_LT_OQ)) != 255)
            throw std::runtime_error("nonfinite activation");
        maximum8 = _mm256_max_ps(maximum8, value);
    }
    float maxima[8];
    _mm256_storeu_ps(maxima, maximum8);
    float maximum = *std::max_element(maxima, maxima + 8);
    for (; k < cols; ++k) {
        if (!std::isfinite(input[k])) throw std::runtime_error("nonfinite activation");
        maximum = std::max(maximum, std::abs(input[k]));
    }
    float scale = maximum > 0 ? maximum / 127.0f : 1.0f;
    if (scale == 0) scale = 1.0f; // Subnormal values below a representable scale quantize to zero.
    __m256 scale8 = _mm256_set1_ps(scale);
    k = 0;
    for (; k + 8 <= cols; k += 8) {
        __m256 value = _mm256_div_ps(_mm256_loadu_ps(input + k), scale8);
        value = _mm256_max_ps(_mm256_set1_ps(-127), _mm256_min_ps(_mm256_set1_ps(127), value));
        __m256i integers = _mm256_cvtps_epi32(_mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m128i shorts = _mm_packs_epi32(_mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(output + k), _mm_packs_epi16(shorts, _mm_setzero_si128()));
    }
    for (; k < cols; ++k) output[k] = static_cast<int8_t>(std::clamp(std::nearbyint(input[k] / scale), -127.0f, 127.0f));
    return scale;
}

// The CPU reference uses per-input-row symmetric int8 activation quantization.
void multiply(const Matrix &w, const float *x, int count, float *y) {
    std::vector<int8_t> q(static_cast<size_t>(count) * w.stride, 0);
    std::vector<float> scales(count);
    for (int m = 0; m < count; ++m) {
        scales[m] = quantize(x + static_cast<size_t>(m) * w.cols, w.cols, q.data() + static_cast<size_t>(m) * w.stride);
    }
    Work work{w, q.data(), scales.data(), y, count};
    if (!w.codes.empty() && count >= lut_batch*w.threads) { run_lut(work); return; }
    auto run = w.exact ? multiply_rows<true> : multiply_rows<false>;
    if (w.threads == 1) run(&work, 0);
    else dispatch_apply_f(w.threads, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), &work, run);
}

std::unique_ptr<Matrix> matrix(int rows, int cols, int threads, bool exact) {
    // Bound sizes before allocation and dot products; the published model fits comfortably.
    if (rows < 1 || rows > 16384 || cols < 1 || cols > 16384 || threads < 1 || threads > 64) return nullptr;
    auto w = std::make_unique<Matrix>();
    w->rows = rows; w->cols = cols; w->stride = (cols + 31) / 32 * 32;
    w->threads = threads; w->exact = exact;
    w->scale.resize(rows); w->low.resize(rows);
    size_t count = static_cast<size_t>(rows) * w->stride;
    if (exact) { w->a.assign(count / 4, 0x55); w->b.assign(count / 4, 0x55); }
    else w->q.resize(count, 0);
    return w;
}
} // namespace

extern "C" {
// The native ABI takes owned copies of compact records. Callers may release input bytes.
void *p2_five_create(int rows, int cols, const uint8_t *blob, size_t length, int threads, int exact) noexcept {
    try {
        auto w = matrix(rows, cols, threads, exact != 0);
        if (!w || !blob) return nullptr;
        size_t row_bytes = (cols + 4) / 5, trit_bytes = rows * row_bytes, nonzero = 0;
        if (length < trit_bytes + 4 * rows) return nullptr;
        for (int r = 0; r < rows; ++r) for (int k = 0; k < cols; k += 5) {
            unsigned byte = blob[r * row_bytes + k / 5];
            if (byte >= 243) return nullptr;
            nonzero += TRITS[byte].nonzero[std::min(5, cols - k)];
        }
        size_t bits_size = (nonzero + 7) / 8;
        if (length != trit_bytes + bits_size + 4 * rows) return nullptr;
        const uint8_t *bits = blob + trit_bytes, *scales = bits + bits_size;
        if (!exact && (!LUT_EXPANSIONS_ONLY || rows >= 2*cols)) {
            size_t groups = (row_bytes+LUT_GROUPS-1)/LUT_GROUPS*LUT_GROUPS;
            w->codes.resize(groups*rows);
            w->magnitudes.resize(2*rows);
        }
        size_t bit = 0;
        for (int r = 0; r < rows; ++r) {
            float low = half(scales + 2 * r), high = half(scales + 2 * rows + 2 * r);
            if (!std::isfinite(low) || !std::isfinite(high) || low < 0 || high < low) return nullptr;
            w->low[r] = low; w->scale[r] = exact ? high - low : high / 127.0f;
            // A row has only two magnitudes. Quantize each once, preserving the
            // per-weight division and rounding contract without repeating it.
            int qlow = 0, qhigh = 0;
            if (!exact && high != 0) {
                qlow = static_cast<int>(std::nearbyint(low / w->scale[r]));
                qhigh = static_cast<int>(std::nearbyint(high / w->scale[r]));
            }
            if (!exact && (!LUT_EXPANSIONS_ONLY || rows >= 2*cols)) {
                w->magnitudes[2*r] = qlow;
                w->magnitudes[2*r+1] = qhigh-qlow;
            }
            for (int k = 0; k < cols; k += 5) {
                const auto &trit = TRITS[blob[r * row_bytes + k / 5]];
                int count = std::min(5, cols - k), used = trit.nonzero[count];
                unsigned high_bits = 0;
                if (used) {
                    high_bits = bits[bit / 8];
                    if (bit % 8 + used > 8) high_bits |= static_cast<unsigned>(bits[bit / 8 + 1]) << 8;
                    high_bits >>= bit % 8;
                }
                bit += used;
                if (!exact && (!LUT_EXPANSIONS_ONLY || rows >= 2*cols)) {
                    size_t g = k/5;
                    size_t at = (g/LUT_GROUPS)*rows*LUT_GROUPS+r*LUT_GROUPS+g%LUT_GROUPS;
                    w->codes[at] = FIVE_LUT_CODES[blob[r*row_bytes+g]][high_bits & 31];
                }
                size_t offset = static_cast<size_t>(r) * w->stride + k;
                if (!exact) {
                    const auto &decode = DECODE_BYTES[blob[r*row_bytes+k/5]];
                    __m128i mask = _mm_cvtsi64_si128(decode.masks[high_bits & 31]);
                    __m128i magnitudes = _mm_blendv_epi8(_mm_set1_epi8(qlow), _mm_set1_epi8(qhigh), mask);
                    __m128i values = _mm_sign_epi8(magnitudes, _mm_cvtsi64_si128(decode.signs));
                    uint64_t bytes = static_cast<uint64_t>(_mm_cvtsi128_si64(values));
                    if (count == 5) {
                        uint32_t four = static_cast<uint32_t>(bytes);
                        std::memcpy(&w->q[offset], &four, sizeof(four));
                        w->q[offset+4] = static_cast<int8_t>(bytes >> 32);
                    } else std::memcpy(&w->q[offset], &bytes, count);
                } else for (int j = 0; j < count; ++j) {
                    int sign = trit.sign[j], is_high = (high_bits >> trit.nonzero[j]) & 1;
                    size_t at = offset + j;
                    unsigned shift = (at % 4) * 2, mask = ~(3u << shift);
                    w->a[at / 4] = (w->a[at / 4] & mask) | ((sign + 1) << shift);
                    w->b[at / 4] = (w->b[at / 4] & mask) | ((sign * is_high + 1) << shift);
                }
            }
        }
        return w.release();
    } catch (...) { return nullptr; }
}

// int6 tables expand to signed bytes, not floats; per-row scales remain exact.
void *p2_int_create(int rows, int cols, const uint8_t *blob, size_t length, int bits, int threads) noexcept {
    try {
        auto w = matrix(rows, cols, threads, false);
        if (!w || !blob || (bits != 6 && bits != 8)) return nullptr;
        size_t total = static_cast<size_t>(rows) * cols;
        size_t body = bits == 8 ? total : (total + 3) / 4 * 3;
        if (length != body + 2 * rows) return nullptr;
        for (int r = 0; r < rows; ++r) {
            w->scale[r] = half(blob + body + 2 * r);
            if (!std::isfinite(w->scale[r]) || w->scale[r] < 0) return nullptr;
            for (int k = 0; k < cols; ++k) {
                size_t at = static_cast<size_t>(r) * cols + k;
                int value;
                if (bits == 8) value = static_cast<int8_t>(blob[at]);
                else {
                    size_t offset = at / 4 * 3;
                    unsigned packed = blob[offset] | (blob[offset + 1] << 8) | (blob[offset + 2] << 16);
                    value = static_cast<int>((packed >> (6 * (at % 4))) & 63) - 32;
                }
                // -128 would break sign-flipping in the AVX2 signed-dot idiom.
                if (value == -128) return nullptr;
                w->q[static_cast<size_t>(r) * w->stride + k] = static_cast<int8_t>(value);
            }
        }
        return w.release();
    } catch (...) { return nullptr; }
}

void p2_destroy(void *handle) noexcept { delete static_cast<Matrix *>(handle); }

int p2_gemm(void *handle, const float *x, int count, float *y) noexcept {
    if (!handle || !x || !y || count < 0) return -1;
    try { multiply(*static_cast<Matrix *>(handle), x, count, y); return 0; }
    catch (...) { return -2; }
}

// Inspect a single row for kernel tests and decoder embedding lookup.
int p2_row(void *handle, int row, float *out) noexcept {
    if (!handle || !out) return -1;
    const auto &w = *static_cast<Matrix *>(handle);
    if (row < 0 || row >= w.rows) return -1;
    for (int k = 0; k < w.cols; ++k) {
        size_t at = static_cast<size_t>(row) * w.stride + k;
        if (w.exact) {
            int a = ((w.a[at / 4] >> (2 * (at % 4))) & 3) - 1;
            int b = ((w.b[at / 4] >> (2 * (at % 4))) & 3) - 1;
            out[k] = a * w.low[row] + b * w.scale[row];
        } else out[k] = w.q[at] * w.scale[row];
    }
    return 0;
}

// One native greedy TDT invocation owns all recurrent state. Matrix handles are
// borrowed only for this synchronous call; Python retains their owners throughout.
int p2_tdt(void *const *handles, const float *const *bias, int hidden, int vocab,
           const int *durations, int duration_count, int blank, int max_symbols,
           const float *encoded, int frames, int *tokens, int *times, int *lengths,
           int capacity) noexcept {
    if (!handles || !bias || !encoded || !tokens || !times || !lengths || !durations ||
        hidden < 1 || hidden > 4096 || vocab < 1 || blank < 0 || blank >= vocab ||
        frames < 0 || duration_count < 1 || max_symbols < 1 || capacity < 0) return -1;
    const Matrix *w[7];
    for (int i = 0; i < 7; ++i) {
        w[i] = static_cast<Matrix *>(handles[i]);
        if (!w[i] || w[i]->cols != hidden) return -1;
    }
    if (w[0]->rows != vocab || w[5]->rows != hidden || w[6]->rows != vocab + duration_count) return -1;
    for (int i = 1; i < 5; ++i) if (w[i]->rows != 4 * hidden) return -1;
    for (int i = 0; i < 6; ++i) if (!bias[i]) return -1;
    for (int i = 0; i < duration_count; ++i) if (durations[i] < 0) return -1;
    try {
        std::vector<float> h(2 * hidden, 0), c(2 * hidden, 0), next_h(2 * hidden), next_c(2 * hidden);
        std::vector<float> input(hidden), ih(4 * hidden), hh(4 * hidden), prediction(hidden), joint(hidden), scores(vocab + duration_count);
        int frame = 0, last = blank, symbols = 0, emitted = 0;
        bool refresh = true;
        for (int64_t step = 0; frame < frames && step < static_cast<int64_t>(max_symbols) * frames + 16; ++step) {
            // A blank preserves prediction state, so its identical LSTM result can
            // be reused for the next acoustic frame. A token invalidates it.
            if (refresh) {
                p2_row(handles[0], last, input.data());
                for (int layer = 0; layer < 2; ++layer) {
                    multiply(*w[1 + 2 * layer], input.data(), 1, ih.data());
                    multiply(*w[2 + 2 * layer], h.data() + layer * hidden, 1, hh.data());
                    for (int k = 0; k < 4 * hidden; ++k) ih[k] += hh[k] + bias[2 * layer][k] + bias[2 * layer + 1][k];
                    for (int k = 0; k < hidden; ++k) {
                        float in_gate = 1.0f / (1.0f + std::exp(-ih[k]));
                        float forget = 1.0f / (1.0f + std::exp(-ih[hidden + k]));
                        float candidate = std::tanh(ih[2 * hidden + k]);
                        float out_gate = 1.0f / (1.0f + std::exp(-ih[3 * hidden + k]));
                        size_t at = layer * hidden + k;
                        next_c[at] = forget * c[at] + in_gate * candidate;
                        input[k] = next_h[at] = out_gate * std::tanh(next_c[at]);
                    }
                }
                multiply(*w[5], input.data(), 1, prediction.data());
                for (int k = 0; k < hidden; ++k) prediction[k] += bias[4][k];
                refresh = false;
            }
            for (int k = 0; k < hidden; ++k) joint[k] = std::max(0.0f, encoded[static_cast<size_t>(frame) * hidden + k] + prediction[k]);
            multiply(*w[6], joint.data(), 1, scores.data());
            for (int k = 0; k < vocab + duration_count; ++k) scores[k] += bias[5][k];
            int token = static_cast<int>(std::max_element(scores.begin(), scores.begin() + vocab) - scores.begin());
            int duration = durations[std::max_element(scores.begin() + vocab, scores.end()) - scores.begin() - vocab];
            if (token == blank && duration == 0) duration = 1;
            if (token != blank) {
                if (emitted >= capacity) return -3;
                tokens[emitted] = token; times[emitted] = frame; lengths[emitted] = duration; ++emitted;
                last = token; h = next_h; c = next_c; refresh = true;
            }
            symbols = duration == 0 ? symbols + 1 : 0;
            if (symbols >= max_symbols) { duration = 1; symbols = 0; }
            frame += duration;
        }
        return frame >= frames ? emitted : -4;
    } catch (...) { return -2; }
}
} // extern C
