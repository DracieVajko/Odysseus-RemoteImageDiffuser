package eu.odysseus.diffusion

import android.graphics.Bitmap
import android.graphics.Color

/**
 * CPU-only image operations (no ONNX). Used by /v1/images/harmonize.
 */
object ImageOps {
    /**
     * Match the masked region's color to the surrounding background and
     * feather the seam. Without a mask, apply a mild auto-level.
     */
    fun harmonize(
        source: Bitmap,
        mask: Bitmap?,
        colorMatch: Boolean = true,
        seamFix: Boolean = true,
        strength: Float = 0.5f,
    ): Bitmap {
        val w = source.width
        val h = source.height
        val src = IntArray(w * h)
        source.getPixels(src, 0, w, 0, 0, w, h)

        if (mask == null) {
            return autoLevel(src, w, h)
        }

        val scaledMask = Bitmap.createScaledBitmap(mask, w, h, true)
        val m = IntArray(w * h)
        scaledMask.getPixels(m, 0, w, 0, 0, w, h)
        val alpha = FloatArray(w * h) { i ->
            val c = m[i]
            val a = Color.alpha(c) / 255f
            val lum = (Color.red(c) * 0.299f + Color.green(c) * 0.587f + Color.blue(c) * 0.114f) / 255f
            (a * lum).coerceIn(0f, 1f)
        }

        val out = IntArray(w * h)
        if (colorMatch) {
            var inR = 0f; var inG = 0f; var inB = 0f; var inN = 0
            var outR = 0f; var outG = 0f; var outB = 0f; var outN = 0
            for (i in src.indices) {
                val c = src[i]
                if (alpha[i] > 0.5f) {
                    inR += Color.red(c); inG += Color.green(c); inB += Color.blue(c); inN++
                } else {
                    outR += Color.red(c); outG += Color.green(c); outB += Color.blue(c); outN++
                }
            }
            if (inN > 0 && outN > 0) {
                val s = strength.coerceIn(0f, 1f)
                val dR = (outR / outN - inR / inN) * s
                val dG = (outG / outN - inG / inN) * s
                val dB = (outB / outN - inB / inN) * s
                for (i in src.indices) {
                    val c = src[i]
                    val a = alpha[i]
                    val r = (Color.red(c) + dR * a).toInt().coerceIn(0, 255)
                    val g = (Color.green(c) + dG * a).toInt().coerceIn(0, 255)
                    val b = (Color.blue(c) + dB * a).toInt().coerceIn(0, 255)
                    out[i] = Color.rgb(r, g, b)
                }
            } else {
                src.copyInto(out)
            }
        } else {
            src.copyInto(out)
        }

        if (seamFix) {
            val blurred = boxBlur(out, w, h, 2)
            for (i in out.indices) {
                val a = alpha[i] * 0.6f
                out[i] = blend(out[i], blurred[i], a)
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun autoLevel(src: IntArray, w: Int, h: Int): Bitmap {
        val histR = IntArray(256); val histG = IntArray(256); val histB = IntArray(256)
        for (c in src) {
            histR[Color.red(c)]++; histG[Color.green(c)]++; histB[Color.blue(c)]++
        }
        val n = src.size
        fun low(hist: IntArray): Int {
            var acc = 0
            for (i in 0..255) { acc += hist[i]; if (acc >= n * 0.02) return i }
            return 0
        }
        fun high(hist: IntArray): Int {
            var acc = 0
            for (i in 255 downTo 0) { acc += hist[i]; if (acc >= n * 0.02) return i }
            return 255
        }
        val loR = low(histR); val hiR = high(histR)
        val loG = low(histG); val hiG = high(histG)
        val loB = low(histB); val hiB = high(histB)
        val out = IntArray(n)
        for (i in src.indices) {
            val c = src[i]
            val r = stretch(Color.red(c), loR, hiR)
            val g = stretch(Color.green(c), loG, hiG)
            val b = stretch(Color.blue(c), loB, hiB)
            out[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun stretch(v: Int, lo: Int, hi: Int): Int =
        if (hi <= lo) v else ((v - lo) * 255 / (hi - lo)).coerceIn(0, 255)

    private fun boxBlur(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        val out = IntArray(src.size)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var r = 0; var g = 0; var b = 0; var n = 0
                for (dy in -radius..radius) {
                    val yy = y + dy
                    if (yy !in 0 until h) continue
                    for (dx in -radius..radius) {
                        val xx = x + dx
                        if (xx !in 0 until w) continue
                        val c = src[yy * w + xx]
                        r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++
                    }
                }
                out[y * w + x] = Color.rgb(r / n, g / n, b / n)
            }
        }
        return out
    }

    private fun blend(base: Int, over: Int, alpha: Float): Int {
        val a = alpha.coerceIn(0f, 1f)
        val r = (Color.red(base) * (1 - a) + Color.red(over) * a).toInt().coerceIn(0, 255)
        val g = (Color.green(base) * (1 - a) + Color.green(over) * a).toInt().coerceIn(0, 255)
        val b = (Color.blue(base) * (1 - a) + Color.blue(over) * a).toInt().coerceIn(0, 255)
        return Color.rgb(r, g, b)
    }
}
