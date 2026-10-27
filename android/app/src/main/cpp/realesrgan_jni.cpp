// JNI bridge for Real-ESRGAN x4plus via ncnn (Vulkan or CPU).
//
// Kotlin does the tiling and bitmap conversion; this file only runs one tile:
//   ncnnLoadNet(param, bin, useVulkan) -> handle
//   ncnnReleaseNet(handle)
//   ncnnUpscaleTile(handle, rgbBytes, w, h) -> rgbBytes (4w x 4h)
//
// ncnn is compiled with -fno-exceptions / -fno-rtti; keep this file plain C++.
#include <jni.h>

#include <net.h>

#include <android/log.h>
#include <cstdint>
#include <cstring>

namespace {

ncnn::Net* to_net(jlong handle) {
    return reinterpret_cast<ncnn::Net*>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_eu_odysseus_diffusion_RealEsrganNcnnEngine_ncnnLoadNet(
        JNIEnv* env, jobject /*thiz*/, jstring param_path, jstring bin_path,
        jboolean use_vulkan) {
    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin = env->GetStringUTFChars(bin_path, nullptr);
    if (param == nullptr || bin == nullptr) {
        if (param) env->ReleaseStringUTFChars(param_path, param);
        if (bin) env->ReleaseStringUTFChars(bin_path, bin);
        return 0;
    }
    auto* net = new ncnn::Net();
    net->opt.use_vulkan_compute = use_vulkan == JNI_TRUE;
    net->opt.num_threads = 2;
    net->opt.use_fp16_packed = true;
    net->opt.use_fp16_storage = true;
    net->opt.use_fp16_arithmetic = true;
    if (net->opt.use_vulkan_compute) {
        net->set_vulkan_device(0);
    }
    const int rc_param = net->load_param(param);
    const int rc_model = net->load_model(bin);
    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);
    if (rc_param != 0 || rc_model != 0) {
        delete net;
        return 0;
    }
    return reinterpret_cast<jlong>(net);
}

extern "C" JNIEXPORT void JNICALL
Java_eu_odysseus_diffusion_RealEsrganNcnnEngine_ncnnReleaseNet(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    delete to_net(handle);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_eu_odysseus_diffusion_RealEsrganNcnnEngine_ncnnUpscaleTile(
        JNIEnv* env, jobject /*thiz*/, jlong handle, jbyteArray rgb, jint w, jint h) {
    ncnn::Net* net = to_net(handle);
    if (net == nullptr || w <= 0 || h <= 0) {
        return nullptr;
    }
    const jsize len = env->GetArrayLength(rgb);
    if (len != static_cast<jsize>(w) * h * 3) {
        return nullptr;
    }
    jbyte* src = env->GetByteArrayElements(rgb, nullptr);
    if (src == nullptr) {
        return nullptr;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(
            reinterpret_cast<const unsigned char*>(src),
            ncnn::Mat::PIXEL_RGB, w, h);
    env->ReleaseByteArrayElements(rgb, src, JNI_ABORT);
    // The shipped realesrgan-x4plus ncnn model is trained on 0..1 floats; the
    // official Vulkan app feeds it 0..255 and gets near-black output, which is
    // what this bridge did until the range probe below showed min/max ~0.16.
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = net->create_extractor();
    int rc_in = ex.input("data", in);
    ncnn::Mat out;
    int rc_out = ex.extract("output", out);
    if (rc_in != 0 || rc_out != 0) {
        __android_log_print(ANDROID_LOG_WARN, "realesrgan_jni",
                            "tile %dx%d: input rc=%d extract rc=%d", in.w, in.h,
                            rc_in, rc_out);
    }
    if (rc_out != 0 || out.empty()) {
        return nullptr;
    }
    const int ow = out.w;
    const int oh = out.h;
    if (ow <= 0 || oh <= 0) {
        return nullptr;
    }
    // Range probe. On some Vulkan drivers (observed on Mali-G925, Android 16)
    // the graph runs but emits a constant frame regardless of the input, which
    // looked like a black upscale. Detect it and fail the tile so Kotlin can
    // retry on CPU; a genuinely flat input is not treated as broken.
    float in_min = 1e9f, in_max = -1e9f;
    for (int c = 0; c < in.c; ++c) {
        const float* p = in.channel(c);
        for (int i = 0; i < in.w * in.h; ++i) {
            if (p[i] < in_min) in_min = p[i];
            if (p[i] > in_max) in_max = p[i];
        }
    }
    float out_min = 1e9f, out_max = -1e9f;
    for (int c = 0; c < out.c; ++c) {
        const float* p = out.channel(c);
        for (int i = 0; i < out.w * out.h; ++i) {
            if (p[i] < out_min) out_min = p[i];
            if (p[i] > out_max) out_max = p[i];
        }
    }
    if (net->opt.use_vulkan_compute) {
        // Only the Vulkan path needs per-tile range telemetry; the CPU path is
        // the steady state and would otherwise spam the log.
        __android_log_print(ANDROID_LOG_INFO, "realesrgan_jni",
                            "range in %.3f..%.3f out %.3f..%.3f pack=%d",
                            in_min, in_max, out_min, out_max, out.elempack);
    }
    // An ESRGAN pass keeps (or widens) the input contrast. A broken Vulkan
    // graph returns a compressed constant-ish frame: measured on Mali-G925 the
    // output range was 7 % of the input range for both a solid colour and a
    // gradient. Treat that as a failed tile so Kotlin retries on CPU.
    const float in_range = in_max - in_min;
    const float out_range = out_max - out_min;
    const bool vulkan_suspect = net->opt.use_vulkan_compute &&
                                in_range > 0.10f && out_range < 0.25f * in_range;
    if (vulkan_suspect) {
        __android_log_print(ANDROID_LOG_WARN, "realesrgan_jni",
                            "degenerate Vulkan output (in %.3f..%.3f out %.3f..%.3f); "
                            "requesting CPU", in_min, in_max, out_min, out_max);
        return nullptr;
    }

    jbyteArray result = env->NewByteArray(static_cast<jsize>(ow) * oh * 3);
    if (result == nullptr) {
        return nullptr;
    }
    jbyte* dst = env->GetByteArrayElements(result, nullptr);
    if (dst == nullptr) {
        return nullptr;
    }
    // Manual planar float -> interleaved RGB bytes with the 0..1 -> 0..255
    // rescale and clamping. to_pixels() would clamp without scaling.
    {
        unsigned char* o = reinterpret_cast<unsigned char*>(dst);
        const int plane = ow * oh;
        const float* r = out.channel(0);
        const float* g = out.c > 1 ? out.channel(1) : out.channel(0);
        const float* b = out.c > 2 ? out.channel(2) : out.channel(0);
        for (int i = 0; i < plane; ++i) {
            float rv = r[i] * 255.f;
            float gv = g[i] * 255.f;
            float bv = b[i] * 255.f;
            o[i * 3 + 0] = (unsigned char) (rv < 0.f ? 0 : (rv > 255.f ? 255 : rv));
            o[i * 3 + 1] = (unsigned char) (gv < 0.f ? 0 : (gv > 255.f ? 255 : gv));
            o[i * 3 + 2] = (unsigned char) (bv < 0.f ? 0 : (bv > 255.f ? 255 : bv));
        }
    }
    env->ReleaseByteArrayElements(result, dst, 0);
    return result;
}
