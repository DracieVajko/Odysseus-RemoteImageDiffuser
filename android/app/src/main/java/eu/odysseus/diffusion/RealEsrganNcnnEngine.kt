package eu.odysseus.diffusion

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File

/**
 * Real-ESRGAN x4plus via ncnn (Vulkan, CPU fallback).
 *
 * Kotlin does the tiling (256 px tiles, 16 px overlap) and bitmap conversion;
 * `librealesrgan_jni.so` runs one tile through ncnn and returns the 4x result.
 *
 * Weights: the `realesrgan-ncnn-vulkan` ZIP from GitHub (param + bin); the
 * downloader extracts it, and the files are found recursively.
 */
object RealEsrganNcnnEngine {
    private const val TAG = "RealEsrganNcnn"
    private const val TILE = 256
    private const val OVERLAP = 16
    private const val MAX_SIDE = 8192

    private var netPtr: Long = 0
    private var modelDir: File? = null
    private var usingVulkan = false
    private var isReady = false
    // Some Vulkan drivers (Mali-G925 observed) return a compressed constant
    // frame for this graph. Remember it so every request does not pay for a
    // failing Vulkan pass before the CPU retry.
    @Volatile private var vulkanBroken = false
    @Volatile var lastError: String = ""


    /**
     * Pick the 4x plus weights. The official zip also ships animevideov3 x2/x3/x4
     * and realesrgan-x4plus-anime; `firstOrNull { .param }` used to pick whichever
     * came first, and a 2x model silently broke the 4x tile math.
     */
    private fun pickWeights(dir: File): Pair<File, File>? {
        val params = dir.walkTopDown().filter { it.isFile && it.name.endsWith(".param") }.toList()
        if (params.isEmpty()) return null
        val param = params.firstOrNull { it.name.equals("realesrgan-x4plus.param", true) }
            ?: params.firstOrNull { it.name.contains("x4plus", true) && !it.name.contains("anime", true) }
            ?: params.firstOrNull { it.name.contains("x4plus", true) }
            ?: params.first()
        val stem = param.nameWithoutExtension
        val bin = dir.walkTopDown().firstOrNull { it.isFile && it.name == "$stem.bin" }
            ?: dir.walkTopDown().firstOrNull { it.isFile && it.name.endsWith(".bin") }
            ?: return null
        return param to bin
    }

    fun init(ctx: Context, dir: File): Boolean {
        return try {
            modelDir = dir
            val picked = pickWeights(dir)
            if (picked == null) {
                lastError = "param/bin not found in ${dir.absolutePath}"
                Log.w(TAG, lastError)
                return false
            }
            val (param, bin) = picked
            Log.i(TAG, "weights: ${param.name} + ${bin.name}")
            val wantVulkan = AppConfig.acceleration.lowercase() != "cpu" && !vulkanBroken
            if (!load(param, bin, wantVulkan)) {
                if (wantVulkan) {
                    Log.w(TAG, "Vulkan init failed, retrying CPU")
                    if (!load(param, bin, false)) return false
                } else {
                    return false
                }
            }
            isReady = true
            Log.i(TAG, "ready (vulkan=$usingVulkan, model=${param.name})")
            true
        } catch (e: Exception) {
            lastError = "init: ${e.message}"
            Log.e(TAG, lastError, e)
            false
        }
    }

    private fun load(param: File, bin: File, vulkan: Boolean): Boolean {
        release()
        netPtr = ncnnLoadNet(param.absolutePath, bin.absolutePath, vulkan)
        if (netPtr == 0L) {
            lastError = "ncnn load failed (vulkan=$vulkan)"
            usingVulkan = false
            return false
        }
        usingVulkan = vulkan
        return true
    }

    fun release() {
        if (netPtr != 0L) {
            ncnnReleaseNet(netPtr)
            netPtr = 0
        }
        isReady = false
        usingVulkan = false
        lastError = ""
    }

    fun isReady(): Boolean = isReady

    /**
     * Upscale a bitmap (2x/3x/4x). The network is native 4x; 2x/3x downsample.
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
        if (w * scale > MAX_SIDE || h * scale > MAX_SIDE) {
            lastError = "upscaled ${w * scale}x${h * scale} exceeds cap ${MAX_SIDE}px"
            return null
        }
        var full = runTiles(src)
        if (full == null && usingVulkan) {
            // Some devices expose Vulkan but fail on this network; retry on CPU.
            val dir = modelDir
            val picked = dir?.let { pickWeights(it) }
            if (picked != null) {
                val (param, bin) = picked
                vulkanBroken = true
                if (load(param, bin, false)) {
                    Log.w(TAG, "Vulkan tile failed, retrying on CPU (sticky for this session)")
                    full = runTiles(src)
                }
            }
        }
        if (full == null) {
            if (lastError.isEmpty()) lastError = "ncnn tile failed"
            return null
        }
        return if (scale == 4) full else {
            val scaled = Bitmap.createScaledBitmap(full, w * scale, h * scale, true)
            if (scaled !== full) full.recycle()
            scaled
        }
    }

    private fun runTiles(src: Bitmap): Bitmap? {
        // Keep the phone responsive: the tiles run on a background-priority
        // thread and ncnn is limited to two threads (see realesrgan_jni.cpp).
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
        val w = src.width
        val h = src.height
        val outW = w * 4
        val outH = h * 4
        val outPx = IntArray(outW * outH)
        var y = 0
        while (y < h) {
            val y1 = maxOf(y - OVERLAP, 0)
            val y2 = minOf(y + TILE + OVERLAP, h)
            var x = 0
            while (x < w) {
                val x1 = maxOf(x - OVERLAP, 0)
                val x2 = minOf(x + TILE + OVERLAP, w)
                val tw = x2 - x1
                val th = y2 - y1
                val crop = Bitmap.createBitmap(src, x1, y1, tw, th)
                val rgb = bitmapToRgb(crop)
                // createBitmap may return the source itself when the subset covers
                // the whole bitmap (small images); recycling it would kill src.
                if (crop !== src) crop.recycle()
                val outBytes = ncnnUpscaleTile(netPtr, rgb, tw, th)
                if (outBytes == null) {
                    lastError = "tile ${x},${y} failed"
                    return null
                }
                val tilePx = rgbToPixels(outBytes, tw * 4, th * 4)
                val ox = (x - x1) * 4
                val oy = (y - y1) * 4
                val cw = (minOf(x + TILE - OVERLAP, w) - x) * 4
                val ch = (minOf(y + TILE - OVERLAP, h) - y) * 4
                for (row in 0 until ch) {
                    System.arraycopy(
                        tilePx, (oy + row) * tw * 4 + ox,
                        outPx, (y * 4 + row) * outW + x * 4,
                        cw,
                    )
                }
                x += TILE - OVERLAP
            }
            y += TILE - OVERLAP
        }
        return Bitmap.createBitmap(outPx, outW, outH, Bitmap.Config.ARGB_8888)
    }

    private fun bitmapToRgb(bmp: Bitmap): ByteArray {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val out = ByteArray(px.size * 3)
        for (i in px.indices) {
            val c = px[i]
            out[i * 3] = ((c shr 16) and 0xFF).toByte()
            out[i * 3 + 1] = ((c shr 8) and 0xFF).toByte()
            out[i * 3 + 2] = (c and 0xFF).toByte()
        }
        return out
    }

    private fun rgbToPixels(data: ByteArray, w: Int, h: Int): IntArray {
        val px = IntArray(w * h)
        for (i in px.indices) {
            val r = data[i * 3].toInt() and 0xFF
            val g = data[i * 3 + 1].toInt() and 0xFF
            val b = data[i * 3 + 2].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return px
    }

    private external fun ncnnLoadNet(paramPath: String, binPath: String, useVulkan: Boolean): Long
    private external fun ncnnReleaseNet(ptr: Long)
    private external fun ncnnUpscaleTile(ptr: Long, input: ByteArray, w: Int, h: Int): ByteArray?

    init {
        try {
            System.loadLibrary("realesrgan_jni")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "librealesrgan_jni.so not loaded (ncnn not bundled)")
        }
    }
}
