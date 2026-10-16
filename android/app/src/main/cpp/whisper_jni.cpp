// JNI most pre WhisperLib.kt (eu.odysseus.diffusion.WhisperLib).
//
// Tenká vrstva nad stabilným whisper.cpp C API (v1.9.5, overené v
// third_party/whisper.cpp/include/whisper.h):
//   whisper_init_from_file / whisper_full / whisper_full_n_segments /
//   whisper_full_get_segment_text / whisper_free
// Žiadna logika navyše: dekódovanie/resampling robí AudioDecoder.kt,
// thready clampuje Kotlin (1..4). Chyby sa vracajú ako "[error] ..." text,
// nikdy pádom (kontrakt: zlý vstup = text, nie crash).
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
    // Segmenty nesú vlastné medzery; len orežeme okraje.
    const auto beg = out.find_first_not_of(" \t\r\n");
    if (beg == std::string::npos) {
        return "";
    }
    const auto end = out.find_last_not_of(" \t\r\n");
    return out.substr(beg, end - beg + 1);
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
    params.n_threads = numThreads < 1 ? 1 : (numThreads > 4 ? 4 : numThreads);
    params.translate = (translate == JNI_TRUE);
    params.language = lang.c_str();

    std::string result;
    if (whisper_full(ctx, params, data, n) != 0) {
        result = "[error] whisper_full failed";
    } else {
        result = collect_text(ctx);
    }
    env->ReleaseFloatArrayElements(samples, data, JNI_ABORT);
    return env->NewStringUTF(result.c_str());
}

} // extern "C"
