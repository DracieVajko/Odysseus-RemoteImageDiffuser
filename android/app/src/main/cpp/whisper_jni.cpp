// JNI bridge for WhisperLib.kt (eu.odysseus.diffusion.WhisperLib).
//
// Thin layer over the stable whisper.cpp C API (v1.9.5, verified in
// third_party/whisper.cpp/include/whisper.h):
//   whisper_init_from_file / whisper_full / whisper_full_n_segments /
//   whisper_full_get_segment_text / whisper_free
// No extra logic: decoding/resampling is done by AudioDecoder.kt,
// threads are clamped by Kotlin (1..4). Errors are returned as "[error] ..."
// text, never by crashing (contract: bad input = text, not a crash).
#include <jni.h>

#include <string>
#include <vector>

#include "whisper.h"

namespace {

std::string collect_text(struct whisper_context * ctx) {
    const int n = whisper_full_n_segments(ctx);
    std::string out;
    for (int i = 0; i < n; ++i) {
        const char * t = whisper_full_get_segment_text(ctx, i);
        if (t) {
            out += t;
        }
    }
    // Segments keep their own spacing; only trim the edges.
    const auto beg = out.find_first_not_of(" \t\r\n");
    if (beg == std::string::npos) {
        return "";
    }
    const auto end = out.find_last_not_of(" \t\r\n");
    return out.substr(beg, end - beg + 1);
}

std::string collect_segments(struct whisper_context * ctx, int64_t offset_ms) {
    const int n = whisper_full_n_segments(ctx);
    std::string out;
    char line[64];
    for (int i = 0; i < n; ++i) {
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i) * 10 + offset_ms;
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i) * 10 + offset_ms;
        const char * text = whisper_full_get_segment_text(ctx, i);
        if (text == nullptr) {
            continue;
        }
        std::string t(text);
        const auto beg = t.find_first_not_of(" \t\r\n");
        if (beg == std::string::npos) {
            continue;
        }
        const auto end = t.find_last_not_of(" \t\r\n");
        t = t.substr(beg, end - beg + 1);
        snprintf(line, sizeof(line), "%.2f\t%.2f\t", t0 / 1000.0, t1 / 1000.0);
        out += line;
        out += t;
        out += "\n";
    }
    return out;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_eu_odysseus_diffusion_WhisperLib_initContext(JNIEnv * env, jobject /*thiz*/, jstring modelPath) {
    if (modelPath == nullptr) {
        return 0;
    }
    const char * path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) {
        return 0;
    }
    struct whisper_context * ctx = whisper_init_from_file(path);
    env->ReleaseStringUTFChars(modelPath, path);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_eu_odysseus_diffusion_WhisperLib_freeContext(JNIEnv * /*env*/, jobject /*thiz*/, jlong contextPtr) {
    auto * ctx = reinterpret_cast<struct whisper_context *>(contextPtr);
    if (ctx) {
        whisper_free(ctx);
    }
}

JNIEXPORT jstring JNICALL
Java_eu_odysseus_diffusion_WhisperLib_fullTranscribe(
        JNIEnv * env, jobject /*thiz*/, jlong contextPtr, jint numThreads,
        jfloatArray samples, jstring language, jboolean translate) {
    auto * ctx = reinterpret_cast<struct whisper_context *>(contextPtr);
    if (!ctx || samples == nullptr) {
        return env->NewStringUTF("[error] null context or samples");
    }
    const jsize n = env->GetArrayLength(samples);
    if (n <= 0) {
        return env->NewStringUTF("");
    }
    jfloat * data = env->GetFloatArrayElements(samples, nullptr);
    if (!data) {
        return env->NewStringUTF("[error] cannot pin samples");
    }

    std::string lang = "auto";
    if (language != nullptr) {
        const char * l = env->GetStringUTFChars(language, nullptr);
        if (l && *l) {
            lang = l;
        }
        if (l) {
            env->ReleaseStringUTFChars(language, l);
        }
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = numThreads < 1 ? 1 : (numThreads > 8 ? 8 : numThreads);
    params.translate = (translate == JNI_TRUE);
    params.language = lang.c_str();
    params.no_context = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    std::string result;
    if (whisper_full(ctx, params, data, n) != 0) {
        result = "[error] whisper_full failed";
    } else {
        result = collect_text(ctx);
    }
    env->ReleaseFloatArrayElements(samples, data, JNI_ABORT);
    return env->NewStringUTF(result.c_str());
}

// Same as fullTranscribe, but returns "start<TAB>end<TAB>text" per segment so
// the server can build SRT/VTT. offsetMs shifts the timings of a chunk.
JNIEXPORT jstring JNICALL
Java_eu_odysseus_diffusion_WhisperLib_fullTranscribeSegments(
        JNIEnv * env, jobject /*thiz*/, jlong contextPtr, jint numThreads,
        jfloatArray samples, jstring language, jboolean translate, jlong offsetMs) {
    auto * ctx = reinterpret_cast<struct whisper_context *>(contextPtr);
    if (!ctx || samples == nullptr) {
        return env->NewStringUTF("");
    }
    const jsize n = env->GetArrayLength(samples);
    jfloat * data = env->GetFloatArrayElements(samples, nullptr);
    if (!data) {
        return env->NewStringUTF("");
    }
    std::string lang = "auto";
    if (language != nullptr) {
        const char * l = env->GetStringUTFChars(language, nullptr);
        if (l && *l) {
            lang = l;
        }
        if (l) {
            env->ReleaseStringUTFChars(language, l);
        }
    }
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = numThreads < 1 ? 1 : (numThreads > 8 ? 8 : numThreads);
    params.translate = (translate == JNI_TRUE);
    params.language = lang.c_str();
    params.no_context = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    std::string result;
    if (whisper_full(ctx, params, data, n) == 0) {
        result = collect_segments(ctx, static_cast<int64_t>(offsetMs));
    }
    env->ReleaseFloatArrayElements(samples, data, JNI_ABORT);
    return env->NewStringUTF(result.c_str());
}

} // extern "C"
