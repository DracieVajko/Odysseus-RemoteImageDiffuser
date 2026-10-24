package eu.odysseus.diffusion

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File

/**
 * Real-ESRGAN x4plus NCNN Vulkan engine on Android.
 *
 * Port of the tiled inference pattern from `unified/upscale.py`.
 * Runs fully on-device using NCNN Vulkan backend.
 *
 * Weights: realesrgan-ncnn-vulkan ZIP from GitHub releases (param + bin).
 * The ZIP extracts to .param and .bin files in the model directory.
 *
 * Supports:
 *  - 2x, 3x, 4x upscale (ESRGAN is native x4; 2x/3x downsample after).
 *  - Tiled inference (tile 256, overlap 16) to keep VRAM low.
 *  - DoS cap 8192px per side.
 */
object RealEsrganNcnnEngine {
    private const val TAG = "RealEsrganNcnn"
    private const val DEFAULT_TILE = 256
    private const val DEFAULT_OVERLAP = 16
    private const val MAX_SIDE = 8192

    // NCNN net handles (native)
    private var netPtr: Long = 0
    private var modelDir: File? = null
    private var isReady = false
    @Volatile var lastError: String = ""

    /** Initialize the NCNN net from the extracted param/bin files. */
    fun init(ctx: Context, dir: File): Boolean {
        return try {
            modelDir = dir
            if (!dir.exists()) {
                Log.w(TAG, "model dir missing: ${dir.absolutePath}")
                return false
            }
            // Look for .param and .bin files (after extracting the ZIP)
            val paramFile = dir.listFiles()?.firstOrNull { it.name.endsWith(".param") }
            val binFile = dir.listFiles()?.firstOrNull { it.name.endsWith(".bin") }
            if (paramFile == null || binFile == null) {
                Log.w(TAG, "param/bin not found in ${dir.absolutePath}")
                return false
            }
            // Load the native net
            netPtr = ncnnLoadNet(paramFile.absolutePath, binFile.absolutePath)
            if (netPtr == 0L) {
                Log.e(TAG, "ncnnLoadNet failed")
                return false
            }
            isReady = true
            Log.i(TAG, "Real-ESRGAN NCNN engine ready")
            true
        } catch (e: Exception) {
            lastError = "init: ${e.message}"
            Log.e(TAG, lastError, e)
            false
        }
    }

    /** Release native resources. */
    fun release() {
        if (netPtr != 0L) {
            ncnnReleaseNet(netPtr)
            netPtr = 0
        }
        isReady = false
        modelDir = null
        lastError = ""
    }

    fun isReady(): Boolean = isReady

    /**
     * Upscale a bitmap using the NCNN net.
     * @param src Input bitmap (RGB).
     * @param scale Target scale (2, 3, or 4).
     * @return Upscaled bitmap, or null on error (check lastError).
     */
    fun upscale(src: Bitmap, scale: Int): Bitmap? {
        if (!isReady || netPtr == 0L) {
            lastError = "engine not ready"
            return null
        }
        if (scale !in 2..4) {
            lastError = "scale must be 2, 3, or 4"
            return null
        }
        val w = src.width
        val h = src.height
        val nw = w * scale
        val nh = h * scale
        if (nw > MAX_SIDE || nh > MAX_SIDE) {
            lastError = "upscaled ${nw}x${nh} exceeds cap ${MAX_SIDE}px"
            return null
        }
        // Convert bitmap to input tensor (RGB, normalized 0..1)
        val input = BitmapToFloatArray(src)
        // Run tiled inference
        val output = ncnnUpscaleTiled(netPtr, input, w, h, scale, DEFAULT_TILE, DEFAULT_OVERLAP)
        if (output == null) {
            lastError = "ncnn inference failed"
            return null
        }
        // Convert output tensor to bitmap
        val outW = if (scale == 4) w * 4 else nw
        val outH = if (scale == 4) h * 4 else nh
        return FloatArrayToBitmap(output, outW, outH)
    }

    // Native JNI methods (implemented in cpp/realesrgan_jni.cpp)
    private external fun ncnnLoadNet(paramPath: String, binPath: String): Long
    private external fun ncnnReleaseNet(ptr: Long)
    private external fun ncnnUpscaleTiled(ptr: Long, input: FloatArray, w: Int, h: Int,
                                        scale: Int, tile: Int, overlap: Int): FloatArray?
    private external fun BitmapToFloatArray(bitmap: Bitmap): FloatArray
    private external fun FloatArrayToBitmap(data: FloatArray, w: Int, h: Int): Bitmap

    init {
        // Loaded by ApiServer when the upscale model is switched to
        try {
            System.loadLibrary("realesrgan_jni")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "librealesrgan_jni.so not loaded (NCNN not bundled or NDK missing)")
        }
    }
}