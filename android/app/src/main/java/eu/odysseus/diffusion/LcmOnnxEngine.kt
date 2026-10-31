package eu.odysseus.diffusion

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * LCM-Dreamshaper v7 ONNX pipeline on Android (onnxruntime-android).
 *
 * Port of `unified/lcm_onnx_numpy.py`. Runs fully on-device.
 *
 * Supports:
 *  - txt2img with optional classifier-free guidance (negative prompt),
 *  - img2img (VAE encoder + strength),
 *  - inpaint (mask blend),
 *  - non-AI color replacement.
 */
object LcmOnnxEngine {
    private const val TAG = "LcmOnnxEngine"

    private const val VAE_SCALE = 0.18215f
    private const val DEFAULT_SIZE = 512
    private const val DEFAULT_STEPS = 4
    private const val ORIGINAL_STEPS = 50
    private const val NUM_TRAIN_TIMESTEPS = 1000
    private const val SIGMA_DATA = 0.5f
    private const val TIMESTEP_SCALING = 10.0f
    private const val BOS = 49406
    private const val EOS = 49407
    private const val MAX_LEN = 77

    private var env: OrtEnvironment? = null
    private var unet: OrtSession? = null
    private var textEncoder: OrtSession? = null
    private var vae: OrtSession? = null
    private var vaeEncoder: OrtSession? = null
    private var vocab: Map<String, Int> = emptyMap()
    private var merges: List<Pair<String, String>> = emptyList()
    private var mergeRank: Map<Pair<String, String>, Int> = emptyMap()
    private var alphasCumprod: FloatArray = FloatArray(0)
    private var finalAlphaCumprod: Float = 1.0f
    private var modelDir: File? = null
    private var isReady = false
    @Volatile var lastError: String = ""

    fun init(ctx: Context, dir: File): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            modelDir = dir
            if (!dir.exists()) {
                Log.w(TAG, "model dir missing: ${dir.absolutePath}")
                return false
            }
            val opts = OrtSession.SessionOptions().apply {
                val threads = HardwareProfile.optimalThreads(ctx)
                try { setIntraOpNumThreads(threads) } catch (_: Exception) { /* cleanup */ }
                try { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT) } catch (_: Exception) { /* cleanup */ }
                val accel = AppConfig.acceleration.lowercase()
                var providerOk = false
                if (accel != "cpu") {
                    if (accel == "gpu") {
                        try {
                            addNnapi()
                            providerOk = true
                            Log.i(TAG, "NNAPI (GPU/NPU) provider enabled")
                        } catch (e: Exception) {
                            Log.w(TAG, "NNAPI unavailable: ${e.message}")
                        }
                        if (!providerOk) {
                            try {
                                addXnnpack(java.util.HashMap())
                                providerOk = true
                                Log.i(TAG, "XNNPACK provider enabled (NNAPI fallback)")
                            } catch (e: Exception) {
                                Log.w(TAG, "XNNPACK unavailable: ${e.message}")
                            }
                        }
                    } else {
                        try {
                            addXnnpack(java.util.HashMap())
                            providerOk = true
                            Log.i(TAG, "XNNPACK provider enabled")
                        } catch (e: Exception) {
                            Log.w(TAG, "XNNPACK unavailable: ${e.message}")
                        }
                        if (!providerOk) {
                            try {
                                addNnapi()
                                providerOk = true
                                Log.i(TAG, "NNAPI provider enabled")
                            } catch (e: Exception) {
                                Log.w(TAG, "NNAPI unavailable: ${e.message}")
                            }
                        }
                    }
                }
                Log.i(TAG, "acceleration=$accel providerOk=$providerOk")
            }
            unet = env!!.createSession((dir.resolve("unet/model.onnx")).absolutePath, opts)
            textEncoder = env!!.createSession((dir.resolve("text_encoder/model.onnx")).absolutePath, opts)
            vae = env!!.createSession((dir.resolve("vae_decoder/model.onnx")).absolutePath, opts)
            val enc = dir.resolve("vae_encoder/model.onnx")
            vaeEncoder = if (enc.isFile) env!!.createSession(enc.absolutePath, opts) else null

            loadTokenizer(dir.resolve("tokenizer"))
            buildSchedulerTables()

            isReady = true
            Log.i(TAG, "LCM ONNX ready (threads=${HardwareProfile.optimalThreads(ctx)}, vaeEncoder=${vaeEncoder != null})")
            true
        } catch (e: Exception) {
            lastError = "init: " + (e.message ?: e.javaClass.simpleName)
            Log.e(TAG, "init failed: ${e.message}", e)
            isReady = false
            false
        }
    }

    fun isReady(): Boolean = isReady && unet != null && textEncoder != null && vae != null

    fun release() {
        try { unet?.close() } catch (_: Exception) { /* cleanup */ }
        try { textEncoder?.close() } catch (_: Exception) { /* cleanup */ }
        try { vae?.close() } catch (_: Exception) { /* cleanup */ }
        try { vaeEncoder?.close() } catch (_: Exception) { /* cleanup */ }
        unet = null; textEncoder = null; vae = null; vaeEncoder = null
        env = null
        isReady = false
    }

    // ---------- public API ----------

    fun generate(
        prompt: String,
        negativePrompt: String = "",
        guidance: Float = 1.0f,
        steps: Int = DEFAULT_STEPS,
        size: Int = DEFAULT_SIZE,
        seed: Long = 0L,
    ): Bitmap? {
        if (!isReady()) return null
        val env = this.env ?: return null
        val te = this.textEncoder ?: return null
        val vae = this.vae ?: return null
        val unet = this.unet ?: return null
        return try {
            val t0 = System.currentTimeMillis()
            val pos = encodePrompt(prompt, te, env)
            val neg = if (negativePrompt.isNotBlank() && guidance > 1.0f)
                encodePrompt(negativePrompt, te, env) else null
            val h = size / 8
            val w = size / 8
            val rng = Random(seed)
            var latents = FloatArray(4 * h * w) { rng.nextGaussian().toFloat() }
            latents = denoise(latents, pos, neg, guidance, steps, size, 0)
            val bmp = decodeLatents(latents, vae, env, size)
            Log.i(TAG, "generate total ${System.currentTimeMillis() - t0}ms (g=$guidance)")
            bmp
        } catch (e: Exception) {
            lastError = "generate: " + (e.message ?: e.javaClass.simpleName)
            Log.e(TAG, "generate failed: ${e.message}", e)
            null
        }
    }

    fun img2img(
        source: Bitmap,
        prompt: String,
        negativePrompt: String = "",
        guidance: Float = 1.0f,
        strength: Float = 0.75f,
        steps: Int = DEFAULT_STEPS,
        size: Int = DEFAULT_SIZE,
        seed: Long = 0L,
    ): Bitmap? {
        if (!isReady()) return null
        val env = this.env ?: return null
        val te = this.textEncoder ?: return null
        val vae = this.vae ?: return null
        val enc = this.vaeEncoder ?: run {
            lastError = "vae_encoder/model.onnx is missing"
            return null
        }
        return try {
            val t0 = System.currentTimeMillis()
            val pos = encodePrompt(prompt, te, env)
            val neg = if (negativePrompt.isNotBlank() && guidance > 1.0f)
                encodePrompt(negativePrompt, te, env) else null
            var latents = encodeLatents(source, enc, env, size)
            val ts = computeTimesteps(steps)
            val start = ((1.0f - strength.coerceIn(0.05f, 1.0f)) * steps).toInt().coerceIn(0, steps - 1)
            val a0 = alphasCumprod[ts[start]]
            val rng = Random(seed)
            val noise = FloatArray(latents.size) { rng.nextGaussian().toFloat() }
            for (i in latents.indices) {
                latents[i] = sqrt(a0) * latents[i] + sqrt(1f - a0) * noise[i]
            }
            latents = denoise(latents, pos, neg, guidance, steps, size, start)
            val bmp = decodeLatents(latents, vae, env, size)
            Log.i(TAG, "img2img total ${System.currentTimeMillis() - t0}ms (strength=$strength)")
            bmp
        } catch (e: Exception) {
            lastError = "img2img: " + (e.message ?: e.javaClass.simpleName)
            Log.e(TAG, "img2img failed: ${e.message}", e)
            null
        }
    }

    fun inpaint(
        source: Bitmap,
        mask: Bitmap,
        prompt: String,
        negativePrompt: String = "",
        guidance: Float = 1.0f,
        strength: Float = 0.8f,
        steps: Int = DEFAULT_STEPS,
        size: Int = DEFAULT_SIZE,
        seed: Long = 0L,
        mode: String = "add",
        color: Int = 0xFF0066FF.toInt(),
    ): Bitmap? {
        if (!isReady()) return null
        if (mode == "color") return colorize(source, mask, color)

        val env = this.env ?: return null
        val te = this.textEncoder ?: return null
        val vae = this.vae ?: return null
        val enc = this.vaeEncoder ?: run {
            lastError = "vae_encoder/model.onnx is missing"
            return null
        }
        return try {
            val t0 = System.currentTimeMillis()
            val pos = encodePrompt(prompt, te, env)
            val neg = if (negativePrompt.isNotBlank() && guidance > 1.0f)
                encodePrompt(negativePrompt, te, env) else null
            var latents = encodeLatents(source, enc, env, size)
            val maskLat = maskToLatent(mask, size)
            val ts = computeTimesteps(steps)
            val start = ((1.0f - strength.coerceIn(0.05f, 1.0f)) * steps).toInt().coerceIn(0, steps - 1)
            val a0 = alphasCumprod[ts[start]]
            val rng = Random(seed)
            val noise = FloatArray(latents.size) { rng.nextGaussian().toFloat() }
            for (i in latents.indices) {
                latents[i] = sqrt(a0) * latents[i] + sqrt(1f - a0) * noise[i]
            }
            val original = latents.copyOf()
            latents = denoise(latents, pos, neg, guidance, steps, size, start, maskLat, original)
            val out = decodeLatents(latents, vae, env, size)
                ?: return null
            val scaled = Bitmap.createScaledBitmap(out, source.width, source.height, true)
            val maskScaled = Bitmap.createScaledBitmap(mask, source.width, source.height, true)
            val composited = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
            val srcPx = IntArray(source.width * source.height)
            val outPx = IntArray(source.width * source.height)
            val maskPx = IntArray(source.width * source.height)
            source.getPixels(srcPx, 0, source.width, 0, 0, source.width, source.height)
            scaled.getPixels(outPx, 0, source.width, 0, 0, source.width, source.height)
            maskScaled.getPixels(maskPx, 0, source.width, 0, 0, source.width, source.height)
            val res = IntArray(srcPx.size) { i ->
                val c = maskPx[i]
                val a = (Color.alpha(c) / 255f) *
                    ((Color.red(c) * 0.299f + Color.green(c) * 0.587f + Color.blue(c) * 0.114f) / 255f)
                blend(srcPx[i], outPx[i], a.coerceIn(0f, 1f))
            }
            composited.setPixels(res, 0, source.width, 0, 0, source.width, source.height)
            Log.i(TAG, "inpaint total ${System.currentTimeMillis() - t0}ms")
            composited
        } catch (e: Exception) {
            lastError = "inpaint: " + (e.message ?: e.javaClass.simpleName)
            Log.e(TAG, "inpaint failed: ${e.message}", e)
            null
        }
    }

    fun colorDepth(): Int = if (vaeEncoder != null) 1 else 0

    // ---------- denoise / CFG ----------

    private fun denoise(
        initial: FloatArray,
        posEmb: FloatArray,
        negEmb: FloatArray?,
        guidance: Float,
        steps: Int,
        size: Int,
        startStep: Int,
        mask: FloatArray? = null,
        original: FloatArray? = null,
    ): FloatArray {
        val unet = this.unet ?: return initial
        val env = this.env ?: return initial
        val ts = computeTimesteps(steps)
        val latents = initial.copyOf()
        for (i in startStep until steps) {
            val t = ts[i]
            val pos = runUnet(latents, t, posEmb, guidance, unet, env, size)
            val noise = if (guidance > 1.0f && negEmb != null) {
                val neg = runUnet(latents, t, negEmb, guidance, unet, env, size)
                FloatArray(latents.size) { j -> neg[j] + guidance * (pos[j] - neg[j]) }
            } else pos
            stepScheduler(latents, noise, t, i, ts, steps)
            if (mask != null && original != null) {
                for (j in latents.indices) {
                    latents[j] = mask[j] * latents[j] + (1f - mask[j]) * original[j]
                }
            }
        }
        return latents
    }

    // ---------- tokenizer ----------

    private fun loadTokenizer(tokDir: File) {
        val vocabFile = tokDir.resolve("vocab.json")
        val mergesFile = tokDir.resolve("merges.txt")
        vocab = if (vocabFile.exists()) {
            val json = JSONObject(vocabFile.readText())
            val m = mutableMapOf<String, Int>()
            val keys = json.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                m[k] = json.getInt(k)
            }
            m
        } else emptyMap()

        merges = if (mergesFile.exists()) {
            mergesFile.readLines()
                .drop(1)
                .mapNotNull { line ->
                    val parts = line.trim().split(" ")
                    if (parts.size == 2) parts[0] to parts[1] else null
                }
        } else emptyList()
        mergeRank = merges.mapIndexed { idx, p -> p to idx }.toMap()
    }

    private fun encodePrompt(prompt: String, te: OrtSession, env: OrtEnvironment): FloatArray {
        val ids = tokenize(prompt)
        val inputIds = IntArray(MAX_LEN) { i -> if (i < ids.size) ids[i] else EOS }
        if (inputIds.isNotEmpty()) inputIds[0] = BOS

        val shape = longArrayOf(1, MAX_LEN.toLong())
        val tensor = OnnxTensor.createTensor(env, IntBuffer.wrap(inputIds), shape)
        val names = te.inputNames.toList()
        val feed = mutableMapOf<String, OnnxTensor>()
        feed[names.first { it.contains("input") || it == "input_ids" }] = tensor
        val maskName = names.firstOrNull { it.contains("attention") }
        if (maskName != null) {
            val mask = IntArray(MAX_LEN) { i -> if (i < ids.size) 1 else 0 }
            feed[maskName] = OnnxTensor.createTensor(env, IntBuffer.wrap(mask), shape)
        }
        val out = te.run(feed)
        val result = out[0] as OnnxTensor
        val buf = result.floatBuffer
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        result.close()
        tensor.close()
        feed.values.forEach { try { it.close() } catch (_: Exception) { /* cleanup */ } }
        out.close()
        return arr
    }

    private fun tokenize(text: String): List<Int> {
        if (vocab.isEmpty()) return listOf(BOS, EOS)
        val cleaned = text.lowercase().trim()
        val words = cleaned.split(Regex("\\s+")).filter { it.isNotBlank() }
        val tokens = mutableListOf<String>()
        tokens.add("<|startoftext|>")
        for (word in words) tokens.addAll(bpeEncode(word))
        tokens.add("<|endoftext|>")
        val ids = tokens.map { vocab[it] ?: vocab["<|endoftext|>"] ?: EOS }
        val out = ids.take(MAX_LEN).toMutableList()
        while (out.size < MAX_LEN) out.add(EOS)
        return out
    }

    private fun bpeEncode(word: String): List<String> {
        var symbols = word.map { it.toString() }.toMutableList()
        if (symbols.isEmpty()) return emptyList()
        symbols[symbols.lastIndex] = symbols.last() + "</w>"
        var changed = true
        while (changed && symbols.size > 1) {
            changed = false
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until symbols.size - 1) {
                val pair = symbols[i] to symbols[i + 1]
                val r = mergeRank[pair]
                if (r != null && r < bestRank) {
                    bestRank = r
                    bestIdx = i
                }
            }
            if (bestIdx >= 0) {
                val merged = symbols[bestIdx] + symbols[bestIdx + 1]
                symbols[bestIdx] = merged
                symbols.removeAt(bestIdx + 1)
                changed = true
            }
        }
        return symbols
    }

    // ---------- scheduler ----------

    private fun buildSchedulerTables() {
        val betas = FloatArray(NUM_TRAIN_TIMESTEPS)
        val start = sqrt(0.00085f)
        val end = sqrt(0.012f)
        for (i in 0 until NUM_TRAIN_TIMESTEPS) {
            val v = start + (end - start) * i / (NUM_TRAIN_TIMESTEPS - 1)
            betas[i] = v * v
        }
        val alphas = FloatArray(NUM_TRAIN_TIMESTEPS) { 1f - betas[it] }
        alphasCumprod = FloatArray(NUM_TRAIN_TIMESTEPS)
        var acc = 1f
        for (i in 0 until NUM_TRAIN_TIMESTEPS) {
            acc *= alphas[i]
            alphasCumprod[i] = acc
        }
        finalAlphaCumprod = alphasCumprod[0]
    }

    private fun computeTimesteps(steps: Int): IntArray {
        val k = NUM_TRAIN_TIMESTEPS / ORIGINAL_STEPS
        val origin = IntArray(ORIGINAL_STEPS) { i -> (i + 1) * k - 1 }
        val reversed = origin.reversed().toIntArray()
        val numInf = steps.coerceIn(1, ORIGINAL_STEPS)
        val indices = FloatArray(numInf) { i -> i * ORIGINAL_STEPS.toFloat() / numInf }
        val infIndices = indices.map { kotlin.math.floor(it).toInt() }
        return infIndices.map { reversed[it.coerceIn(0, reversed.lastIndex)] }.toIntArray()
    }

    // ---------- UNet ----------

    private fun runUnet(
        sample: FloatArray, t: Int, embeds: FloatArray, guidance: Float,
        unet: OrtSession, env: OrtEnvironment, size: Int,
    ): FloatArray {
        val h = size / 8
        val w = size / 8
        val sampleShape = longArrayOf(1, 4, h.toLong(), w.toLong())
        val sampleTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), sampleShape)

        val tShape = longArrayOf(1)
        val tLong = if (t < 0) 0L else t.toLong()
        val tTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(tLong)), tShape)

        val embedShape = longArrayOf(1, MAX_LEN.toLong(), 768)
        val embedTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(embeds), embedShape)

        val names = unet.inputNames.toList()
        val feed = mutableMapOf<String, OnnxTensor>()
        val sName = names.firstOrNull { it.contains("sample") } ?: names[0]
        feed[sName] = sampleTensor
        val tName = names.firstOrNull { it.contains("timestep") } ?: names[1]
        feed[tName] = tTensor
        val eName = names.firstOrNull { it.contains("encoder") || it.contains("hidden") } ?: names[2]
        feed[eName] = embedTensor
        val tcName = names.firstOrNull { it.contains("timestep_cond") || it.contains("cond") }
        if (tcName != null) {
            val tc = guidanceEmbedding((guidance - 1.0f).coerceAtLeast(0f), 256)
            val tcShape = longArrayOf(1, 256)
            feed[tcName] = OnnxTensor.createTensor(env, FloatBuffer.wrap(tc), tcShape)
        }

        val out = unet.run(feed)
        val result = out[0] as OnnxTensor
        val buf = result.floatBuffer
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        result.close()
        sampleTensor.close(); tTensor.close(); embedTensor.close()
        feed.values.forEach { try { it.close() } catch (_: Exception) { /* cleanup */ } }
        out.close()
        return arr
    }

    private fun guidanceEmbedding(w: Float, dim: Int): FloatArray {
        val wv = w * 1000.0f
        val half = dim / 2
        val emb = FloatArray(half)
        val log10000 = kotlin.math.ln(10000.0f)
        for (i in 0 until half) {
            emb[i] = exp(i.toFloat() * (-log10000 / (half - 1)))
        }
        val sinCos = FloatArray(dim)
        for (i in 0 until half) {
            val v = wv * emb[i]
            sinCos[i] = kotlin.math.sin(v)
            sinCos[half + i] = kotlin.math.cos(v)
        }
        return sinCos
    }

    // ---------- scheduler step ----------

    private fun stepScheduler(
        latents: FloatArray, noise: FloatArray, t: Int, stepIdx: Int,
        timesteps: IntArray, totalSteps: Int,
    ) {
        val alphaProdT = alphasCumprod[t.coerceIn(0, alphasCumprod.lastIndex)]
        val prevT = if (stepIdx + 1 < timesteps.size) timesteps[stepIdx + 1] else t
        val alphaProdTPrev = if (prevT >= 0 && prevT < alphasCumprod.size) alphasCumprod[prevT] else finalAlphaCumprod
        val betaProdT = 1f - alphaProdT
        val betaProdTPrev = 1f - alphaProdTPrev

        val scaledT = t * TIMESTEP_SCALING
        val cSkip = SIGMA_DATA * SIGMA_DATA / (scaledT * scaledT + SIGMA_DATA * SIGMA_DATA)
        val cOut = scaledT / sqrt(scaledT * scaledT + SIGMA_DATA * SIGMA_DATA)

        val size = latents.size
        val isLast = stepIdx == totalSteps - 1
        val rng = Random(t.toLong() * 7919L + stepIdx)
        for (i in 0 until size) {
            val predX0 = (latents[i] - sqrt(betaProdT) * noise[i]) / sqrt(alphaProdT)
            val denoised = cOut * predX0 + cSkip * latents[i]
            latents[i] = if (isLast) denoised
            else sqrt(alphaProdTPrev) * denoised + sqrt(betaProdTPrev) * rng.nextGaussian().toFloat()
        }
    }

    // ---------- image helpers ----------

    private fun encodeLatents(bmp: Bitmap, enc: OrtSession, env: OrtEnvironment, size: Int): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bmp, size, size, true)
        val px = IntArray(size * size)
        scaled.getPixels(px, 0, size, 0, 0, size, size)
        val input = FloatArray(3 * size * size)
        for (i in px.indices) {
            val c = px[i]
            input[i] = Color.red(c) / 127.5f - 1f
            input[size * size + i] = Color.green(c) / 127.5f - 1f
            input[2 * size * size + i] = Color.blue(c) / 127.5f - 1f
        }
        val shape = longArrayOf(1, 3, size.toLong(), size.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
        val names = enc.inputNames.toList()
        val feed = mapOf((names.firstOrNull { it.contains("sample") } ?: names[0]) to tensor)
        val out = enc.run(feed)
        val result = out[0] as OnnxTensor
        val buf = result.floatBuffer
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        for (i in arr.indices) arr[i] *= VAE_SCALE
        result.close(); tensor.close(); out.close()
        return arr
    }

    private fun maskToLatent(mask: Bitmap, size: Int): FloatArray {
        val h = size / 8
        val w = size / 8
        val scaled = Bitmap.createScaledBitmap(mask, w, h, true)
        val px = IntArray(w * h)
        scaled.getPixels(px, 0, w, 0, 0, w, h)
        val m = FloatArray(w * h) { i ->
            val c = px[i]
            // white = repaint area (alpha or luminance)
            val a = Color.alpha(c) / 255f
            val lum = (Color.red(c) * 0.299f + Color.green(c) * 0.587f + Color.blue(c) * 0.114f) / 255f
            (a * lum).coerceIn(0f, 1f)
        }
        // simple 3x3 box blur for feathering
        val out = FloatArray(m.size)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sum = 0f; var n = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val yy = y + dy; val xx = x + dx
                    if (yy in 0 until h && xx in 0 until w) { sum += m[yy * w + xx]; n++ }
                }
                out[y * w + x] = sum / n
            }
        }
        // The latent has 4 channels; broadcast the single-channel mask.
        return FloatArray(out.size * 4) { i -> out[i % out.size] }
    }

    private fun decodeLatents(latents: FloatArray, vae: OrtSession, env: OrtEnvironment, size: Int): Bitmap? {
        val h = size / 8
        val w = size / 8
        val input = FloatArray(latents.size) { latents[it] / VAE_SCALE }
        val shape = longArrayOf(1, 4, h.toLong(), w.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
        val names = vae.inputNames.toList()
        val name = names.firstOrNull { it.contains("latent") } ?: names[0]
        val out = vae.run(mapOf(name to tensor))
        val result = out[0] as OnnxTensor
        val buf = result.floatBuffer
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        result.close(); tensor.close(); out.close()

        val outH = size
        val outW = size
        val pixels = IntArray(outH * outW)
        for (y in 0 until outH) {
            for (x in 0 until outW) {
                val r = ((arr[y * outW + x] / 2f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()
                val g = ((arr[outH * outW + y * outW + x] / 2f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()
                val b = ((arr[2 * outH * outW + y * outW + x] / 2f + 0.5f).coerceIn(0f, 1f) * 255f).toInt()
                pixels[y * outW + x] = Color.rgb(r, g, b)
            }
        }
        return Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
    }

    private fun colorize(source: Bitmap, mask: Bitmap, color: Int): Bitmap? {
        val w = source.width
        val h = source.height
        val src = IntArray(w * h)
        val m = IntArray(w * h)
        source.getPixels(src, 0, w, 0, 0, w, h)
        Bitmap.createScaledBitmap(mask, w, h, true).getPixels(m, 0, w, 0, 0, w, h)
        val targetHsv = FloatArray(3)
        Color.colorToHSV(color, targetHsv)
        val hsv = FloatArray(3)
        val out = IntArray(src.size)
        for (i in src.indices) {
            val c = src[i]
            Color.colorToHSV(c, hsv)
            hsv[0] = targetHsv[0]
            hsv[1] = targetHsv[1].coerceIn(0.35f, 1f)
            val recolored = Color.HSVToColor(Color.alpha(c), hsv)
            val a = (Color.alpha(m[i]) / 255f) *
                ((Color.red(m[i]) * 0.299f + Color.green(m[i]) * 0.587f + Color.blue(m[i]) * 0.114f) / 255f)
            out[i] = blend(c, recolored, a.coerceIn(0f, 1f))
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun blend(base: Int, over: Int, alpha: Float): Int {
        val a = alpha.coerceIn(0f, 1f)
        val r = (Color.red(base) * (1 - a) + Color.red(over) * a).toInt().coerceIn(0, 255)
        val g = (Color.green(base) * (1 - a) + Color.green(over) * a).toInt().coerceIn(0, 255)
        val b = (Color.blue(base) * (1 - a) + Color.blue(over) * a).toInt().coerceIn(0, 255)
        return Color.rgb(r, g, b)
    }
}


/**
 * Hardware capability detection for adaptive configuration.
 */
object HardwareProfile {
    fun optimalThreads(ctx: Context): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        return cores.coerceIn(1, 8).let { if (it > 1) it - 1 else 1 }.coerceAtMost(8)
    }

    fun maxImageSize(ctx: Context): Int {
        val ramMb = (Runtime.getRuntime().maxMemory() / 1024 / 1024).toInt()
        return when {
            ramMb >= 4096 -> 512
            ramMb >= 2048 -> 384
            else -> 256
        }
    }

    fun canRunModel(modelId: String): Boolean = true
}
