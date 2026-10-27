// JNI wrapper for libmp3lame: PCM (16-bit interleaved) -> MP3 frames.
//
// Only the encoder is used; LAME is LGPL-2.1 and linked dynamically as its own
// .so, which keeps the app's MIT licence intact (no ffmpeg, no GPL).
#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <vector>

#include "lame.h"

#define MP3_TAG "mp3_jni"
#define MP3_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, MP3_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jbyteArray JNICALL
Java_eu_odysseus_diffusion_Mp3Encoder_nativeEncode(
        JNIEnv* env, jobject /*thiz*/, jshortArray pcm, jint sample_rate,
        jint channels, jint bitrate_kbps) {
    if (pcm == nullptr || sample_rate <= 0 || (channels != 1 && channels != 2)) {
        MP3_LOGE("encode: bad arguments");
        return nullptr;
    }
    const jsize n = env->GetArrayLength(pcm);
    if (n <= 0) return nullptr;
    jshort* samples = env->GetShortArrayElements(pcm, nullptr);
    if (samples == nullptr) return nullptr;

    lame_global_flags* gf = lame_init();
    if (gf == nullptr) {
        env->ReleaseShortArrayElements(pcm, samples, JNI_ABORT);
        return nullptr;
    }
    lame_set_in_samplerate(gf, sample_rate);
    lame_set_num_channels(gf, channels);
    lame_set_brate(gf, bitrate_kbps);
    lame_set_mode(gf, channels == 1 ? MONO : JOINT_STEREO);
    lame_set_quality(gf, 2);  // 0 = slowest/best, 9 = fastest
    if (lame_init_params(gf) < 0) {
        MP3_LOGE("encode: lame_init_params failed");
        lame_close(gf);
        env->ReleaseShortArrayElements(pcm, samples, JNI_ABORT);
        return nullptr;
    }

    std::vector<unsigned char> out;
    const int frames = n / channels;
    const int chunk_frames = 4096;
    std::vector<unsigned char> buf(static_cast<size_t>(chunk_frames) * 1.25f + 7200);

    int remaining = frames;
    jshort* cursor = samples;
    while (remaining > 0) {
        const int take = remaining < chunk_frames ? remaining : chunk_frames;
        int written;
        if (channels == 1) {
            written = lame_encode_buffer(gf, cursor, cursor, take, buf.data(),
                                         static_cast<int>(buf.size()));
        } else {
            written = lame_encode_buffer_interleaved(gf, cursor, take, buf.data(),
                                                     static_cast<int>(buf.size()));
        }
        if (written < 0) {
            MP3_LOGE("encode: lame_encode_buffer %d", written);
            lame_close(gf);
            env->ReleaseShortArrayElements(pcm, samples, JNI_ABORT);
            return nullptr;
        }
        out.insert(out.end(), buf.begin(), buf.begin() + written);
        cursor += static_cast<size_t>(take) * channels;
        remaining -= take;
    }

    int flushed = lame_encode_flush(gf, buf.data(), static_cast<int>(buf.size()));
    if (flushed > 0) out.insert(out.end(), buf.begin(), buf.begin() + flushed);

    // ID3v2 header so players show the encoder tag.
    unsigned char id3[4096];
    int id3_size = lame_get_id3v2_tag(gf, id3, sizeof(id3));
    lame_close(gf);
    env->ReleaseShortArrayElements(pcm, samples, JNI_ABORT);
    if (id3_size > 0) out.insert(out.begin(), id3, id3 + id3_size);

    if (out.empty()) return nullptr;
    jbyteArray result = env->NewByteArray(static_cast<jsize>(out.size()));
    if (result == nullptr) return nullptr;
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(out.size()),
                            reinterpret_cast<const jbyte*>(out.data()));
    return result;
}
