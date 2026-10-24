package eu.odysseus.diffusion

import android.util.Log
import java.io.File

/**
 * JNI wrapper for whisper.cpp (`libwhisper_jni.so`).
 *
* The model context is CACHED: loading 181 MB of weights takes tens of seconds,
* so every request would be unusably slow. The cache holds
* at most 1 model; a request for another model correctly releases the old one.
 *
* If the native library is missing, the fallback returns a message without crashing.
 */
object WhisperLib {
    private const val TAG = "WhisperLib"
    private var isNativeLoaded = false
    private var cachedPath: String? = null
    private var cachedCtx: Long = 0L
    private val lock = Any()

    init {
        try {
// Our JNI bridge (libwhisper_jni.so) — whisper.cpp itself is libwhisper.so
// and is loaded automatically via DT_NEEDED.
            System.loadLibrary("whisper_jni")
            isNativeLoaded = true
            Log.i(TAG, "libwhisper_jni.so loaded")
        } catch (e: UnsatisfiedLinkError) {
            isNativeLoaded = false
            Log.w(TAG, "native lib missing: ${e.message}")
        } catch (e: Exception) {
            isNativeLoaded = false
            Log.w(TAG, "native lib failed: ${e.message}")
        }
    }

    fun isAvailable(): Boolean = isNativeLoaded

    /**
     * Transkripcia / preklad zvuku pomocou whisper.cpp.
* @param modelFile path to the GGML file (e.g. ggml-small-q5_1.bin)
     * @param samples FloatArray PCM samples at 16 kHz mono [-1.0, 1.0]
     * @param language "sk", "en", "auto"
* @param translate true to translate to English (/v1/audio/translations)
     */
    fun transcribe(
        modelFile: File,
        samples: FloatArray,
        language: String = "sk",
        translate: Boolean = false,
    ): String {
        if (!isNativeLoaded) {
            return "[stub] libwhisper_jni.so is not in the APK. Samples: ${samples.size} (~${samples.size / 16000}s)"
        }
        if (!modelFile.exists()) {
            return "[error] model file not found: ${modelFile.absolutePath}"
        }
        if (samples.isEmpty()) {
            return ""
        }

        synchronized(lock) {
            ensureContext(modelFile)
            if (cachedCtx == 0L) {
                return "[error] failed to initialize whisper context"
            }
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
            val t0 = System.currentTimeMillis()
            return try {
                val text = fullTranscribe(cachedCtx, threads, samples, language, translate)
                Log.i(TAG, "transcribed ${samples.size / 16000.0}s in ${System.currentTimeMillis() - t0} ms " +
                        "(threads=$threads, lang=$language)")
                text
            } catch (e: Exception) {
                Log.e(TAG, "inference failed: ${e.message}", e)
                "[error] whisper inference failed: ${e.message}"
            }
        }
    }

/** Load the model into the cache if not already loaded (or a different one). Call under the lock. */
    private fun ensureContext(modelFile: File) {
        val path = modelFile.absolutePath
        if (cachedCtx != 0L && cachedPath == path) return
        if (cachedCtx != 0L) {
            freeContext(cachedCtx)
            cachedCtx = 0L
            cachedPath = null
            Log.i(TAG, "released previous whisper context")
        }
        val t0 = System.currentTimeMillis()
        cachedCtx = initContext(path)
        cachedPath = if (cachedCtx != 0L) path else null
        Log.i(TAG, "model loaded in ${System.currentTimeMillis() - t0} ms (ok=${cachedCtx != 0L})")
    }

/** Release the context (called on model switch / idle unload). */
    fun release() {
        synchronized(lock) {
            if (cachedCtx != 0L) {
                freeContext(cachedCtx)
                Log.i(TAG, "context released")
            }
            cachedCtx = 0L
            cachedPath = null
        }
    }

// --- Native JNI methods ---
    private external fun initContext(modelPath: String): Long
    private external fun freeContext(contextPtr: Long)
    private external fun fullTranscribe(
        contextPtr: Long,
        numThreads: Int,
        samples: FloatArray,
        language: String,
        translate: Boolean,
    ): String
}
