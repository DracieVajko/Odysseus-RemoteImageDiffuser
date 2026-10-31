package eu.odysseus.diffusion

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * On-device replacement for the PC's ffmpeg tools, built on MediaExtractor /
 * MediaMuxer / MediaCodec. No external binary is shipped: Android's codecs
 * cover container remuxing, trimming, audio extraction, AAC/WAV encoding and
 * image conversion. What the platform cannot do (MP3/FLAC/Opus encoders, AVIF,
 * video re-encode without an EGL pipeline) is reported as unsupported instead of
 * pretending success.
 */
object MediaTools {
    private const val TAG = "MediaTools"

    data class Output(val bytes: ByteArray, val name: String, val mime: String)

    class Unsupported(message: String) : Exception(message)

    // ---------------------------------------------------------------- presets
    val videoPresets = linkedMapOf(
        "mp4-h264" to mapOf("desc" to "MP4 H.264 (remux when the source is already H.264)", "ext" to ".mp4", "vcodec" to "h264", "supported" to true),
        "mp4-h265" to mapOf("desc" to "MP4 H.265 (remux when the source is already HEVC)", "ext" to ".mp4", "vcodec" to "h265", "supported" to true),
        "mp4-h264-fast" to mapOf("desc" to "MP4 H.264 fast (remux)", "ext" to ".mp4", "vcodec" to "h264", "supported" to true),
        "mp4-h265-fast" to mapOf("desc" to "MP4 H.265 fast (remux)", "ext" to ".mp4", "vcodec" to "h265", "supported" to true),
        "mp4-small" to mapOf("desc" to "MP4 small (re-encoded to ~5 MB)", "ext" to ".mp4", "supported" to true),
        "webm" to mapOf("desc" to "WebM (VP9 encode)", "ext" to ".webm", "supported" to false,
                        "reason" to "VP9 encoding is not implemented on device"),
    )

    val audioPresets = linkedMapOf(
        "aac-128" to mapOf("desc" to "AAC 128 kbps (Apple/Android compatible)", "ext" to ".m4a", "supported" to true),
        "aac-192" to mapOf("desc" to "AAC 192 kbps", "ext" to ".m4a", "supported" to true),
        "wav-16k" to mapOf("desc" to "WAV 16 kHz mono (for Whisper)", "ext" to ".wav", "supported" to true),
        "mp3-128" to mapOf("desc" to "MP3 128 kbps (LAME)", "ext" to ".mp3", "supported" to true),
        "mp3-192" to mapOf("desc" to "MP3 192 kbps (LAME)", "ext" to ".mp3", "supported" to true),
        "mp3-320" to mapOf("desc" to "MP3 320 kbps (LAME)", "ext" to ".mp3", "supported" to true),
        "opus" to mapOf("desc" to "Opus", "ext" to ".opus", "supported" to false,
                        "reason" to "Ogg/Opus muxing is not available in MediaMuxer on this API level"),
        "flac" to mapOf("desc" to "FLAC (lossless)", "ext" to ".flac", "supported" to false,
                        "reason" to "FLAC container muxing is not available in MediaMuxer"),
    )

    val imagePresets = linkedMapOf(
        "jpg-95" to mapOf("desc" to "JPG quality 95 (max)", "ext" to ".jpg", "supported" to true),
        "jpg-85" to mapOf("desc" to "JPG quality 85 (smaller)", "ext" to ".jpg", "supported" to true),
        "png" to mapOf("desc" to "PNG (lossless)", "ext" to ".png", "supported" to true),
        "webp" to mapOf("desc" to "WebP (small, high quality)", "ext" to ".webp", "supported" to true),
        "avif" to mapOf("desc" to "AVIF (smallest)", "ext" to ".avif", "supported" to false,
                        "reason" to "no AVIF encoder on Android"),
    )

    fun presetsJson(): Map<String, Any> {
        // MP3 depends on the bundled LAME library; report it honestly if it is
        // missing from this build.
        val audio = audioPresets.mapValues { (key, value) ->
            if (key.startsWith("mp3") && !Mp3Encoder.available) {
                value + mapOf("supported" to false,
                              "reason" to "libmp3_jni.so is not bundled in this build")
            } else value
        }
        return mapOf(
            "video" to videoPresets,
            "audio" to audio,
            "image" to imagePresets,
            "resize" to mapOf(
            "480p" to mapOf("width" to 854, "height" to 480, "desc" to "480p (SD)"),
            "720p" to mapOf("width" to 1280, "height" to 720, "desc" to "720p (HD)"),
            "1080p" to mapOf("width" to 1920, "height" to 1080, "desc" to "1080p (Full HD)"),
            "square-512" to mapOf("width" to 512, "height" to 512, "desc" to "512x512 (AI)"),
            "square-768" to mapOf("width" to 768, "height" to 768, "desc" to "768x768 (AI)"),
            "square-1024" to mapOf("width" to 1024, "height" to 1024, "desc" to "1024x1024 (AI)"),
        ),
            "engine" to "android-mediacodec",
        )
    }

    // ------------------------------------------------------------------- info
    fun info(path: File): Map<String, Any> {
        val retriever = MediaMetadataRetriever()
        val out = linkedMapOf<String, Any>()
        try {
            retriever.setDataSource(path.absolutePath)
            out["duration_s"] = (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L) / 1000.0
            out["bitrate"] = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)
                ?.toLongOrNull() ?: 0L
            out["size_bytes"] = path.length()
            out["format_name"] = path.extension.lowercase()
        } finally {
            retriever.release()
        }

        val streams = ArrayList<Map<String, Any>>()
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(path.absolutePath)
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: "?"
                val s = linkedMapOf<String, Any>(
                    "index" to i,
                    "codec_type" to if (mime.startsWith("video/")) "video" else if (mime.startsWith("audio/")) "audio" else "other",
                    "codec_name" to mime,
                )
                if (f.containsKey(MediaFormat.KEY_WIDTH)) s["width"] = f.getInteger(MediaFormat.KEY_WIDTH)
                if (f.containsKey(MediaFormat.KEY_HEIGHT)) s["height"] = f.getInteger(MediaFormat.KEY_HEIGHT)
                if (f.containsKey(MediaFormat.KEY_SAMPLE_RATE)) s["sample_rate"] = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                if (f.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) s["channels"] = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                if (f.containsKey(MediaFormat.KEY_BIT_RATE)) s["bit_rate"] = f.getInteger(MediaFormat.KEY_BIT_RATE)
                if (f.containsKey(MediaFormat.KEY_FRAME_RATE)) s["frame_rate"] = f.getInteger(MediaFormat.KEY_FRAME_RATE)
                streams.add(s)
            }
        } finally {
            extractor.release()
        }
        out["streams"] = streams
        return out
    }

    // ------------------------------------------------------------------ image
    fun convertImage(input: File, preset: String, quality: Int = 0): Output {
        val spec = imagePresets[preset] ?: throw Unsupported("unknown image preset: $preset")
        if (spec["supported"] != true) throw Unsupported(spec["reason"] as String)

        val bmp = BitmapFactory.decodeFile(input.absolutePath)
            ?: throw Unsupported("cannot decode image")
        val out = ByteArrayOutputStream()
        val (format, q) = when (preset) {
            "jpg-95" -> Bitmap.CompressFormat.JPEG to 95
            "jpg-85" -> Bitmap.CompressFormat.JPEG to 85
            "png" -> Bitmap.CompressFormat.PNG to 100
            else -> Bitmap.CompressFormat.WEBP_LOSSY to (if (quality in 1..100) quality else 90)
        }
        bmp.compress(format, q, out)
        bmp.recycle()
        val ext = spec["ext"] as String
        val mime = when (ext) {
            ".jpg" -> "image/jpeg"
            ".png" -> "image/png"
            else -> "image/webp"
        }
        return Output(out.toByteArray(), "converted$ext", mime)
    }

    // ------------------------------------------------------------ trim / mux
    /** Remux (no re-encode) the slice [startS, endS] of the input. */
    fun trim(input: File, startS: Double, endS: Double, outName: String = "trim.mp4"): Output {
        val startUs = (startS * 1_000_000).toLong().coerceAtLeast(0)
        val endUs = if (endS > startS) (endS * 1_000_000).toLong() else Long.MAX_VALUE
        val extractor = MediaExtractor()
        val outFile = File(input.parentFile, "mt_$outName")
        var muxer: MediaMuxer? = null
        try {
            extractor.setDataSource(input.absolutePath)
            muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val trackMap = LinkedHashMap<Int, Int>()
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("audio/") && !mime.startsWith("video/")) continue
                trackMap[i] = muxer.addTrack(f)
                extractor.selectTrack(i)
            }
            if (trackMap.isEmpty()) throw Unsupported("no audio or video track found")
            muxer.start()

            val buffer = ByteBuffer.allocateDirect(1 shl 20)
            val info = MediaCodec.BufferInfo()
            val pending = HashMap<Int, Long>()   // track -> first pts to normalise
            var written = 0L
            while (true) {
                buffer.clear()
                val size = extractor.readSampleData(buffer, 0)
                if (size < 0) break
                val srcTrack = extractor.sampleTrackIndex
                val dstTrack = trackMap[srcTrack] ?: run { extractor.advance(); return@run -1 }
                if (dstTrack < 0) { extractor.advance(); continue }
                val pts = extractor.sampleTime
                if (pts > endUs) break
                if (pts >= startUs) {
                    val base = pending.getOrPut(dstTrack) { pts }
                    info.set(0, size, (pts - base).coerceAtLeast(0),
                             extractor.sampleFlags)
                    muxer.writeSampleData(dstTrack, buffer, info)
                    written += size
                }
                extractor.advance()
            }
            if (written == 0L) throw Unsupported("nothing in the requested time range")
        } finally {
            try { muxer?.stop() } catch (_: Exception) { /* cleanup */ }
            try { muxer?.release() } catch (_: Exception) { /* cleanup */ }
            extractor.release()
        }
        val bytes = outFile.readBytes()
        outFile.delete()
        return Output(bytes, outName, "video/mp4")
    }

    /** Copy the audio track into an .m4a (or .wav for PCM sources). */
    fun extractAudio(input: File, preset: String = "aac-192"): Output {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(input.absolutePath)
            var audioTrack = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) { audioTrack = i; format = f; break }
            }
            if (audioTrack < 0 || format == null) throw Unsupported("no audio track")
            val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
            // Already AAC-in-MP4: remux without touching samples.
            if (mime == MediaFormat.MIMETYPE_AUDIO_AAC) {
                extractor.selectTrack(audioTrack)
                val outFile = File(input.parentFile, "mt_audio.m4a")
                var muxer: MediaMuxer? = null
                try {
                    muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                    val track = muxer.addTrack(format)
                    muxer.start()
                    val buf = ByteBuffer.allocateDirect(1 shl 18)
                    val info = MediaCodec.BufferInfo()
                    while (true) {
                        buf.clear()
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) break
                        info.set(0, size, extractor.sampleTime, extractor.sampleFlags)
                        muxer.writeSampleData(track, buf, info)
                        extractor.advance()
                    }
                } finally {
                    try { muxer?.stop() } catch (_: Exception) { /* cleanup */ }
                    try { muxer?.release() } catch (_: Exception) { /* cleanup */ }
                }
                val bytes = outFile.readBytes()
                outFile.delete()
                if (preset.startsWith("mp3")) {
                    // a copied AAC track cannot become MP3: re-encode instead
                    return toMp3(input, preset)
                }
                return Output(bytes, "audio.m4a", "audio/mp4")
            }
        } finally {
            extractor.release()
        }
        // Other codecs: decode + re-encode.
        return if (preset.startsWith("mp3")) toMp3(input, preset) else transcodeAudio(input, "aac-192")
    }

    // ------------------------------------------------------------- audio ops
    fun convertAudio(input: File, preset: String): Output {
        val spec = audioPresets[preset] ?: throw Unsupported("unknown audio preset: $preset")
        if (spec["supported"] != true) throw Unsupported(spec["reason"] as String)
        return when {
            preset == "wav-16k" -> toWav16k(input)
            preset.startsWith("mp3") -> toMp3(input, preset)
            else -> transcodeAudio(input, preset)
        }
    }

    private data class Pcm(val samples: ShortArray, val rate: Int, val channels: Int)

    private fun decodeToPcm(input: File): Pcm {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(input.absolutePath)
            var track = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) { track = i; format = f; break }
            }
            if (track < 0 || format == null) throw Unsupported("no audio track")
            extractor.selectTrack(track)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            // WAV / raw PCM has no decoder in MediaCodec: read the samples as-is.
            if (mime == MediaFormat.MIMETYPE_AUDIO_RAW) {
                val chunks = ArrayList<ShortArray>()
                var total = 0
                val buf = ByteBuffer.allocateDirect(1 shl 16).order(ByteOrder.LITTLE_ENDIAN)
                while (true) {
                    buf.clear()
                    val size = extractor.readSampleData(buf, 0)
                    if (size < 0) break
                    buf.position(0); buf.limit(size)
                    val shorts = ShortArray(size / 2)
                    buf.asShortBuffer().get(shorts)
                    chunks.add(shorts); total += shorts.size
                    extractor.advance()
                }
                val all = ShortArray(total)
                var pos = 0
                for (c in chunks) { System.arraycopy(c, 0, all, pos, c.size); pos += c.size }
                return Pcm(all, rate, channels)
            }

            val decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(format, null, null, 0)
            decoder.start()
            val chunks = ArrayList<ShortArray>()
            var total = 0
            val info = MediaCodec.BufferInfo()
            var sawInputEos = false
            var sawOutputEos = false
            while (!sawOutputEos) {
                if (!sawInputEos) {
                    val inIdx = decoder.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = decoder!!.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            decoder!!.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEos = true
                        } else {
                            decoder!!.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = decoder!!.dequeueOutputBuffer(info, 10_000)
                if (outIdx >= 0) {
                    val buf = decoder!!.getOutputBuffer(outIdx)
                    if (buf != null && info.size > 0) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val shorts = ShortArray(info.size / 2)
                        buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts)
                        chunks.add(shorts)
                        total += shorts.size
                    }
                    decoder!!.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEos = true
                } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    }
            }
            decoder.stop(); decoder.release()
            val all = ShortArray(total)
            var pos = 0
            for (c in chunks) { System.arraycopy(c, 0, all, pos, c.size); pos += c.size }
            return Pcm(all, rate, channels)
        } finally {
            extractor.release()
        }
    }

    private fun toWav16k(input: File): Output {
        val pcm = decodeToPcm(input)
        // linear resample to 16 kHz mono
        val srcRate = pcm.rate
        val srcCh = pcm.channels
        val frames = pcm.samples.size / srcCh
        val dstRate = 16000
        val outFrames = (frames.toLong() * dstRate / srcRate).toInt()
        val out = ShortArray(outFrames)
        for (i in 0 until outFrames) {
            val srcPos = i.toDouble() * srcRate / dstRate
            val i0 = min(srcPos.toInt(), frames - 1)
            val i1 = min(i0 + 1, frames - 1)
            val frac = srcPos - i0
            var acc = 0.0
            for (c in 0 until srcCh) {
                val a = pcm.samples[i0 * srcCh + c].toDouble()
                val b = pcm.samples[i1 * srcCh + c].toDouble()
                acc += a + (b - a) * frac
            }
            out[i] = (acc / srcCh).toInt().toShort()
        }
        val bytes = ByteArray(44 + out.size * 2)
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + out.size * 2); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(dstRate); bb.putInt(dstRate * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(out.size * 2)
        for (s in out) bb.putShort(s)
        return Output(bytes, "audio16k.wav", "audio/wav")
    }

    private fun transcodeAudio(input: File, preset: String): Output {
        val bitrate = when (preset) {
            "aac-128" -> 128_000
            "aac-192" -> 192_000
            else -> 160_000
        }
        val pcm = decodeToPcm(input)
        return encodePcmToAac(pcm, bitrate)
    }

    /** Encode interleaved 16-bit PCM into an AAC-in-MP4 file. */
    private fun encodePcmToAac(pcm: Pcm, bitrate: Int): Output {
        val outFile = File(System.getProperty("java.io.tmpdir") ?: ".", "mt_aac_${System.nanoTime()}.m4a")
        var muxer: MediaMuxer? = null
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        try {
            val fmt = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC,
                                                    pcm.rate, pcm.channels)
            fmt.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            fmt.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            fmt.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 32768)
            encoder.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()

            val bytesPerFrame = 2 * pcm.channels
            val framesPerChunk = 1024
            val chunkBytes = framesPerChunk * bytesPerFrame
            val info = MediaCodec.BufferInfo()
            var muxTrack = -1
            var muxStarted = false
            var ptsUs = 0L
            val bytesPerSecond = pcm.rate.toLong() * bytesPerFrame

            var offset = 0
            val totalBytes = pcm.samples.size * 2
            var inputDone = false
            var outputDone = false

            fun drain(endOfStream: Boolean) {
                while (!outputDone) {
                    val idx = encoder.dequeueOutputBuffer(info, 0)
                    if (idx == MediaCodec.INFO_TRY_AGAIN_LATER) {
                        if (!endOfStream) return
                        continue
                    }
                    if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                        muxTrack = muxer!!.addTrack(encoder.outputFormat)
                        muxer!!.start()
                        muxStarted = true
                        continue
                    }
                    if (idx >= 0) {
                        val buf = encoder.getOutputBuffer(idx)
                        if (buf != null && info.size > 0 &&
                            info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            muxer?.writeSampleData(muxTrack, buf, info)
                        }
                        encoder.releaseOutputBuffer(idx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                            return
                        }
                    }
                }
            }

            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = encoder.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val inBuf = encoder.getInputBuffer(inIdx)!!
                        inBuf.clear()
                        val remaining = totalBytes - offset
                        if (remaining <= 0) {
                            encoder.queueInputBuffer(inIdx, 0, 0, ptsUs,
                                                     MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val n = min(remaining, chunkBytes)
                            val shorts = ShortArray(n / 2)
                            System.arraycopy(pcm.samples, offset / 2, shorts, 0, shorts.size)
                            val bb = ByteBuffer.allocateDirect(n).order(ByteOrder.LITTLE_ENDIAN)
                            bb.asShortBuffer().put(shorts)
                            bb.position(0)
                            inBuf.put(bb)
                            encoder.queueInputBuffer(inIdx, 0, n, ptsUs, 0)
                            offset += n
                            ptsUs += n * 1_000_000L / bytesPerSecond
                        }
                    }
                }
                drain(false)
            }
            var guard = 0
            while (!outputDone && guard++ < 100) drain(true)

            encoder.stop(); encoder.release()
            if (muxStarted) { try { muxer?.stop() } catch (_: Exception) { /* cleanup */ } }
            muxer?.release(); muxer = null

            val bytes = outFile.readBytes()
            if (bytes.isEmpty()) throw Unsupported("audio encode produced no data")
            return Output(bytes, "audio.m4a", "audio/mp4")
        } catch (e: Unsupported) {
            throw e
        } catch (e: Exception) {
            throw Unsupported("audio encode failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            try { encoder.stop() } catch (_: Exception) { /* cleanup */ }
            try { encoder.release() } catch (_: Exception) { /* cleanup */ }
            try { muxer?.release() } catch (_: Exception) { /* cleanup */ }
            outFile.delete()
        }
    }

    /** Decode any audio source and encode it with LAME. */
    private fun toMp3(input: File, preset: String): Output {
        if (!Mp3Encoder.available) {
            throw Unsupported("libmp3_jni.so is not bundled in this build")
        }
        val bitrate = when (preset) {
            "mp3-128" -> 128
            "mp3-192" -> 192
            "mp3-320" -> 320
            else -> 192
        }
        val pcm = decodeToPcm(input)
        val bytes = Mp3Encoder.encode(pcm.samples, pcm.rate, pcm.channels, bitrate)
            ?: throw Unsupported("MP3 encoding failed")
        return Output(bytes, "audio.mp3", "audio/mpeg")
    }

    // ------------------------------------------------------------- video ops
    /** Remux into MP4 without re-encoding (container/codec compatibility path). */
    fun remux(input: File, preset: String): Output {
        val spec = videoPresets[preset] ?: throw Unsupported("unknown video preset: $preset")
        if (spec["supported"] != true) throw Unsupported(spec["reason"] as String)
        val wantHevc = (spec["vcodec"] as String) == "h265"
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(input.absolutePath)
            var videoMime = ""
            for (i in 0 until extractor.trackCount) {
                val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("video/")) { videoMime = mime; break }
            }
            val isHevc = videoMime == MediaFormat.MIMETYPE_VIDEO_HEVC
            val isAvc = videoMime == MediaFormat.MIMETYPE_VIDEO_AVC
            if (wantHevc && !isHevc) {
                throw Unsupported("source is not H.265; transcoding to H.265 is not implemented on device")
            }
            if (!wantHevc && !isAvc) {
                throw Unsupported("source is not H.264; transcoding is not implemented on device")
            }
        } finally {
            extractor.release()
        }
        val out = trim(input, 0.0, 0.0, "converted.mp4")
        return Output(out.bytes, "converted.mp4", "video/mp4")
    }

    fun compress(input: File, preset: String): Output =
        VideoTranscoder.compress(input, preset)
}
