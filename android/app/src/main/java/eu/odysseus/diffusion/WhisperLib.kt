package eu.odysseus.diffusion

import java.io.File

/**
 * JNI obal pre whisper.cpp (`libwhisper.so`).
 *
 * Ak prebuilt `.so` súbor ešte nie je v `jniLibs/arm64-v8a`,
 * fallback bezpečne vráti oznam bez pádov appky.
 */
object WhisperLib {
    private var isNativeLoaded = false

    init {
        try {
            // Náš JNI most (libwhisper_jni.so) — whisper.cpp samotný je libwhisper.so
            // a načíta sa automaticky cez DT_NEEDED.
            System.loadLibrary("whisper_jni")
            isNativeLoaded = true
        } catch (e: UnsatisfiedLinkError) {
            isNativeLoaded = false
        } catch (e: Exception) {
            isNativeLoaded = false
        }
    }

    fun isAvailable(): Boolean = isNativeLoaded

    /**
     * Transkripcia / preklad zvuku pomocou whisper.cpp.
     * @param modelFile cesta ku GGUF/GGML súboru (napr. ggml-small-q5_1.bin)
     * @param samples FloatArray PCM vzozriek pri 16 kHz mono [-1.0, 1.0]
     * @param language "sk", "en", "auto"
     * @param translate true pre preklad do angličtiny (/v1/audio/translations)
     */
    fun transcribe(
        modelFile: File,
        samples: FloatArray,
        language: String = "sk",
        translate: Boolean = false,
    ): String {
        if (!isNativeLoaded) {
            return "[stub] libwhisper.so zatiaľ nie je v jniLibs. Vzorky zvuku: ${samples.size} (~${samples.size / 16000}s)"
        }
        if (!modelFile.exists()) {
            return "[error] model file not found: ${modelFile.absolutePath}"
        }
        if (samples.isEmpty()) {
            return ""
        }

        var ctxPtr = 0L
        try {
            ctxPtr = initContext(modelFile.absolutePath)
            if (ctxPtr == 0L) {
                return "[error] failed to initialize whisper context"
            }
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            return fullTranscribe(ctxPtr, threads, samples, language, translate)
        } catch (e: Exception) {
            return "[error] whisper inference failed: ${e.message}"
        } finally {
            if (ctxPtr != 0L) {
                freeContext(ctxPtr)
            }
        }
    }

    // --- Native JNI metódy ---
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
