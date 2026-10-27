// JNI bridge for SmolVLM (llama.cpp + libmtmd).
//
// The Kotlin engine (SmolVlmEngine.kt) calls:
//   llamaLoadContext(modelPath, mmprojPath) -> handle (0 = failure)
//   llamaReleaseContext(handle)
//   llamaGenerateCaption(handle, bitmap, prompt, maxTokens) -> text? (null = failure)
//
// Everything runs on the CPU backend: the phone already uses its Vulkan driver
// for ncnn upscale and ORT for the LCM pipeline, and VLM latency is dominated
// by the text decoder, not by a GPU offload.
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define VLM_TAG "smolvlm_jni"
#define VLM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, VLM_TAG, __VA_ARGS__)
#define VLM_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, VLM_TAG, __VA_ARGS__)

namespace {

constexpr int kThreads = 4;
constexpr int kCtxSize = 4096;
constexpr int kBatch = 512;

struct VlmHandle {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    mtmd_context* mtmd = nullptr;
};

VlmHandle* as_handle(jlong h) { return reinterpret_cast<VlmHandle*>(h); }

std::string jstring_to_std(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

bool bitmap_to_rgb(JNIEnv* env, jobject bitmap, std::vector<unsigned char>& rgb,
                   uint32_t& out_w, uint32_t& out_h) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return false;
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return false;
    }
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return false;
    }
    out_w = info.width;
    out_h = info.height;
    rgb.resize(static_cast<size_t>(out_w) * out_h * 3);
    const auto* src = static_cast<const uint32_t*>(pixels);
    for (size_t i = 0; i < static_cast<size_t>(out_w) * out_h; ++i) {
        const uint32_t px = src[i];
        // Android ARGB_8888 in memory is RGBA byte order on little endian.
        rgb[i * 3 + 0] = static_cast<unsigned char>((px >> 0) & 0xFF);
        rgb[i * 3 + 1] = static_cast<unsigned char>((px >> 8) & 0xFF);
        rgb[i * 3 + 2] = static_cast<unsigned char>((px >> 16) & 0xFF);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return true;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_eu_odysseus_diffusion_SmolVlmEngine_llamaLoadContext(
        JNIEnv* env, jobject /*thiz*/, jstring model_path, jstring mmproj_path) {
    const std::string model_file = jstring_to_std(env, model_path);
    const std::string mmproj_file = jstring_to_std(env, mmproj_path);
    if (model_file.empty() || mmproj_file.empty()) {
        VLM_LOGE("load: empty paths");
        return 0;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_file.c_str(), mparams);
    if (model == nullptr) {
        VLM_LOGE("load: model failed: %s", model_file.c_str());
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = kCtxSize;
    cparams.n_batch = kBatch;
    cparams.n_ubatch = kBatch;
    cparams.n_threads = kThreads;
    cparams.n_threads_batch = kThreads;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        VLM_LOGE("load: context failed");
        llama_model_free(model);
        return 0;
    }

    mtmd_context_params tparams = mtmd_context_params_default();
    tparams.use_gpu = false;
    tparams.print_timings = false;
    tparams.n_threads = kThreads;
    mtmd_context* mctx = mtmd_init_from_file(mmproj_file.c_str(), model, tparams);
    if (mctx == nullptr) {
        VLM_LOGE("load: mtmd failed: %s", mmproj_file.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    auto* handle = new VlmHandle();
    handle->model = model;
    handle->ctx = ctx;
    handle->vocab = llama_model_get_vocab(model);
    handle->mtmd = mctx;
    VLM_LOGI("loaded %s", model_file.c_str());
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_eu_odysseus_diffusion_SmolVlmEngine_llamaReleaseContext(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    VlmHandle* h = as_handle(ptr);
    if (h == nullptr) return;
    if (h->mtmd) mtmd_free(h->mtmd);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

extern "C" JNIEXPORT jstring JNICALL
Java_eu_odysseus_diffusion_SmolVlmEngine_llamaGenerateCaption(
        JNIEnv* env, jobject /*thiz*/, jlong ptr, jobject bitmap,
        jstring prompt_str, jint max_tokens) {
    VlmHandle* h = as_handle(ptr);
    if (h == nullptr || h->ctx == nullptr || h->mtmd == nullptr) {
        VLM_LOGE("generate: handle not ready");
        return nullptr;
    }

    std::vector<unsigned char> rgb;
    uint32_t w = 0, hgt = 0;
    if (!bitmap_to_rgb(env, bitmap, rgb, w, hgt) || w == 0 || hgt == 0) {
        VLM_LOGE("generate: bitmap conversion failed");
        return nullptr;
    }

    const std::string user_prompt = jstring_to_std(env, prompt_str);
    const char* marker = mtmd_get_marker(h->mtmd);
    const std::string marker_str(marker ? marker : mtmd_default_marker());
    const std::string question =
            user_prompt.empty() ? "Describe the image in detail." : user_prompt;

    // A bare completion makes a 500M model continue the dialogue after the
    // answer ("Question: ... Answer: ..."). Use the GGUF's chat template when
    // present so generation stops at the assistant turn.
    std::string text;
    const char* tmpl = llama_model_chat_template(h->model, nullptr);
    bool templated = false;
    if (tmpl != nullptr) {
        const std::string user_content = marker_str + "\n" + question;
        llama_chat_message msg{};
        msg.role = "user";
        msg.content = user_content.c_str();
        std::vector<char> buf(user_content.size() * 2 + 512);
        const int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(),
                                                    static_cast<int32_t>(buf.size()));
        if (n > 0 && n <= static_cast<int32_t>(buf.size())) {
            text.assign(buf.data(), static_cast<size_t>(n));
            templated = true;
        }
    }
    if (!templated) {
        text = marker_str + "\n" + question;
    }
    VLM_LOGI("prompt: template=%s (%zu chars)", templated ? "yes" : "no", text.size());

    mtmd_bitmap* bmp = mtmd_bitmap_init(w, hgt, rgb.data());
    if (bmp == nullptr) {
        VLM_LOGE("generate: mtmd_bitmap_init failed");
        return nullptr;
    }
    const mtmd_bitmap* bitmaps[1] = {bmp};

    mtmd_input_text itext;
    itext.text = text.c_str();
    itext.text_len = text.size();
    itext.add_special = true;
    itext.parse_special = true;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(h->mtmd, chunks, &itext, bitmaps, 1);
    if (rc != 0) {
        VLM_LOGE("generate: tokenize rc=%d", rc);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_pos n_past = 0;
    rc = mtmd_helper_eval_chunks(h->mtmd, h->ctx, chunks, 0, 0, kBatch, true, &n_past);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    if (rc != 0) {
        VLM_LOGE("generate: eval rc=%d", rc);
        return nullptr;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const int limit = max_tokens <= 0 ? 256 : (max_tokens > 512 ? 512 : max_tokens);
    std::string out;
    for (int i = 0; i < limit && n_past < kCtxSize - 4; ++i) {
        const llama_token tok = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) {
            break;
        }
        char buf[256];
        const int n = llama_token_to_piece(h->vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = tok;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        if (llama_decode(h->ctx, batch) != 0) {
            llama_batch_free(batch);
            VLM_LOGE("generate: decode failed at token %d", i);
            break;
        }
        llama_batch_free(batch);
        ++n_past;
    }
    llama_sampler_free(smpl);

    VLM_LOGI("generate: %zu chars", out.size());
    return env->NewStringUTF(out.c_str());
}
