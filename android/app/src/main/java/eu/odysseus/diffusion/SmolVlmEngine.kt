package eu.odysseus.diffusion

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File

/**
 * SmolVLM 500M Instruct engine using llama.cpp (GGUF + mmproj).
 *
 * Runs fully on-device using llama.cpp with libmtmd for multimodal support.
 *
 * Weights:
 *  - SmolVLM-500M-Instruct-Q8_0.gguf (model)
 *  - mmproj-SmolVLM-500M-Instruct-Q8_0.gguf (vision projector)
 *
 * Supports:
 *  - Image captioning (image + prompt → text)
 *  - Vision chat (multi-turn with images)
 *  - Max image side 1024px (images are resized if larger).
 *
 * Note: llama.cpp multimodal (mtmd) is experimental; requires libmtmd.so.
 * The GGUF files are downloaded via ModelDownloader (GGUF + mmproj).
 */
object SmolVlmEngine {
    private const val TAG = "SmolVlmEngine"
    private const val DEFAULT_MAX_TOKENS = 256
    private const val MAX_SIDE = 1024

    // llama.cpp context handles (native)
    private var ctxPtr: Long = 0
    private var modelDir: File? = null
    private var isReady = false
    @Volatile var lastError: String = ""

    /**
     * True only when libsmolvlm_jni.so is actually bundled in the APK. The
     * llama.cpp mtmd bridge is not part of every build, so the endpoints must
     * answer 503 instead of crashing with UnsatisfiedLinkError.
     */
    @Volatile var nativeAvailable: Boolean = false
        private set

    /** Initialize the llama.cpp context from GGUF + mmproj. */
    fun init(ctx: Context, dir: File): Boolean {
        if (!nativeAvailable) {
            lastError = "libsmolvlm_jni.so is not bundled in this build (llama.cpp mtmd)"
            Log.w(TAG, lastError)
            return false
        }
        return try {
            modelDir = dir
            if (!dir.exists()) {
                Log.w(TAG, "model dir missing: ${dir.absolutePath}")
                return false
            }
            val modelFile = dir.listFiles()?.firstOrNull { it.name.endsWith(".gguf") && !it.name.contains("mmproj") }
            val mmprojFile = dir.listFiles()?.firstOrNull { it.name.contains("mmproj") && it.name.endsWith(".gguf") }
            if (modelFile == null || mmprojFile == null) {
                Log.w(TAG, "model or mmproj not found in ${dir.absolutePath}")
                return false
            }
            // Load the native context with multimodal support
            ctxPtr = llamaLoadContext(modelFile.absolutePath, mmprojFile.absolutePath)
            if (ctxPtr == 0L) {
                Log.e(TAG, "llamaLoadContext failed")
                return false
            }
            isReady = true
            Log.i(TAG, "SmolVLM llama.cpp engine ready")
            true
        } catch (e: Exception) {
            lastError = "init: ${e.message}"
            Log.e(TAG, lastError, e)
            false
        }
    }

    /** Release native resources. */
    fun release() {
        if (ctxPtr != 0L) {
            llamaReleaseContext(ctxPtr)
            ctxPtr = 0
        }
        isReady = false
        modelDir = null
        lastError = ""
    }

    fun isReady(): Boolean = isReady

    /**
     * Generate a caption for an image.
     * @param image Input bitmap (RGB).
     * @param prompt Optional prompt (default: "Describe the image in detail.").
     * @param maxTokens Maximum tokens to generate.
     * @return Caption text, or null on error (check lastError).
     */
    fun caption(image: Bitmap, prompt: String?, maxTokens: Int): String? {
        if (!nativeAvailable) {
            lastError = "libsmolvlm_jni.so is not bundled in this build (llama.cpp mtmd)"
            return null
        }
        if (!isReady || ctxPtr == 0L) {
            lastError = "engine not ready"
            return null
        }
        val p = prompt?.ifEmpty { null } ?: "Describe the image in detail."
        val tokens = maxTokens.coerceIn(16, 512)
        // Resize image if needed (llama.cpp mtmd handles preprocessing)
        val resized = if (image.width > MAX_SIDE || image.height > MAX_SIDE) {
            Bitmap.createScaledBitmap(image, MAX_SIDE, MAX_SIDE, true)
        } else image
        return try {
            llamaGenerateCaption(ctxPtr, resized, p, tokens)
        } catch (e: Throwable) {
            // UnsatisfiedLinkError is an Error, not an Exception: catching
            // Throwable keeps the request alive and reports a real reason.
            lastError = "generate: ${e.message}"
            Log.e(TAG, lastError, e)
            null
        }
    }

    // Native JNI methods (implemented in cpp/smolvlm_jni.cpp)
    private external fun llamaLoadContext(modelPath: String, mmprojPath: String): Long
    private external fun llamaReleaseContext(ptr: Long)
    private external fun llamaGenerateCaption(ptr: Long, image: Bitmap, prompt: String, maxTokens: Int): String?

    init {
        // Loaded by ApiServer when the VLM model is switched to
        try {
            System.loadLibrary("smolvlm_jni")
            nativeAvailable = true
        } catch (e: UnsatisfiedLinkError) {
            nativeAvailable = false
            Log.w(TAG, "libsmolvlm_jni.so not loaded (llama.cpp mtmd not bundled or NDK missing)")
        }
    }
}