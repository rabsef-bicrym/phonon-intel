// Use the installed ATen CPU operators without importing Python's torch package.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {
thread_local char error_message[512]{};
struct Frontend {
    int inputs, outputs, mels;
    at::Tensor window, filters, weight, bias;
};

// The C boundary owns errors; no tensor or exception crosses into ctypes.
template<class Function>
int checked(Function function) noexcept {
    try {
        c10::InferenceMode inference;
        function();
        error_message[0] = '\0';
        return 0;
    } catch (const std::exception &error) {
        std::snprintf(error_message, sizeof(error_message), "%s", error.what());
    } catch (...) {
        std::snprintf(error_message, sizeof(error_message), "Unknown ATen failure");
    }
    return -2;
}

// Borrowed input arrays remain alive for each synchronous CPU operation.
at::Tensor view(const float *data, at::IntArrayRef shape) {
    return at::from_blob(const_cast<float *>(data), shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}
}

extern "C" {
// Clone checkpoint tensors once into the model-owned frontend, not a process cache.
void *p2_frontend_create(const float *filters, const float *weight, const float *bias,
                         int inputs, int outputs, int mels, int threads) noexcept {
    std::unique_ptr<Frontend> owner;
    int status = checked([&] {
        TORCH_CHECK(filters && weight && bias, "Missing frontend weights");
        TORCH_CHECK(inputs > 0 && inputs <= 4096 && inputs % 2 == 0 && outputs > 0 && outputs <= 4096 &&
                    mels > 0 && mels <= 512 && threads > 0 && threads <= 64, "Invalid frontend dimensions");
        at::set_num_threads(threads);
        owner = std::make_unique<Frontend>();
        owner->inputs = inputs; owner->outputs = outputs; owner->mels = mels;
        owner->window = at::hann_window(400, false, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        owner->filters = view(filters, {mels,257}).clone();
        owner->weight = view(weight, {outputs,inputs}).clone();
        owner->bias = view(bias, {outputs}).clone();
    });
    return status == 0 ? owner.release() : nullptr;
}

// Destruction happens only after the owner's synchronous calls have returned.
void p2_frontend_destroy(void *handle) noexcept { delete static_cast<Frontend *>(handle); }

// The error buffer belongs to the calling thread and is replaced by its next call.
const char *p2_frontend_error() noexcept { return error_message; }

// Preserve Python STFT padding, intermediate strides, and reduction order exactly.
int p2_frontend_features(void *handle, const float *audio, int samples, float *out, int capacity) noexcept {
    return checked([&] {
        TORCH_CHECK(handle && audio && out && samples >= 160 && samples <= 480000, "Invalid feature input");
        const auto &f = *static_cast<Frontend *>(handle);
        int frames = samples/160+1;
        TORCH_CHECK(capacity >= frames, "Feature output capacity too small");
        auto wave = view(audio, {1,samples});
        TORCH_CHECK(at::isfinite(wave).all().item<bool>(), "Nonfinite waveform");
        wave = at::cat({wave.slice(1,0,1), wave.slice(1,1)-wave.slice(1,0,samples-1)*0.97}, 1);
        auto padded = at::constant_pad_nd(wave.view({1,1,samples}), {256,256}, 0).view({1,samples+512});
        auto spectrum = at::stft(padded,512,160,400,f.window,false,c10::nullopt,true);
        auto power = at::view_as_real(spectrum).pow(2).sum(at::IntArrayRef{-1},false).sqrt().pow(2);
        auto features = (at::matmul(f.filters,power)+0x1p-24).log().permute({0,2,1});
        auto mean = features.mean(at::IntArrayRef{1},true);
        auto variance = (features-mean).pow(2).sum(at::IntArrayRef{1},false)/(features.size(1)-1);
        features = ((features-mean)/(variance.sqrt().unsqueeze(1)+1e-5)).contiguous();
        TORCH_CHECK(features.size(1) == frames, "Unexpected STFT frame count");
        std::memcpy(out,features.const_data_ptr<float>(),static_cast<size_t>(frames)*f.mels*sizeof(float));
    });
}

// Keep the same three-dimensional input shape used by the Python projection.
int p2_frontend_project(void *handle, const float *input, int frames, float *output) noexcept {
    return checked([&] {
        TORCH_CHECK(handle && input && output && frames > 0 && frames <= 376, "Invalid projection input");
        const auto &f = *static_cast<Frontend *>(handle);
        auto x = view(input,{1,frames,f.inputs});
        TORCH_CHECK(at::isfinite(x).all().item<bool>(), "Nonfinite projection input");
        auto y = at::linear(x,f.weight,f.bias).contiguous();
        std::memcpy(output,y.const_data_ptr<float>(),static_cast<size_t>(frames)*f.outputs*sizeof(float));
    });
}

// Python's reverse division is reciprocal followed by multiplication by 1.0.
int p2_frontend_positions(void *handle, float *output) noexcept {
    return checked([&] {
        TORCH_CHECK(handle && output, "Invalid position output");
        const auto &f = *static_cast<Frontend *>(handle);
        auto exponent = at::arange(0,f.inputs,2,at::TensorOptions().dtype(at::kLong)).to(at::kFloat)/f.inputs;
        auto frequency = at::pow(10000.0,exponent).reciprocal()*1.0;
        std::memcpy(output,frequency.const_data_ptr<float>(),static_cast<size_t>(f.inputs/2)*sizeof(float));
    });
}
}
