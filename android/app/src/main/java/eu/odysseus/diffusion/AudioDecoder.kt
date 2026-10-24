package eu.odysseus.diffusion

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
* Audio decoder to 16 kHz, 16-bit mono float PCM array (input for whisper.cpp).
 *
 * Podporuje:
 * 1. Direct WAV parser for 16 kHz 16-bit PCM.
 * 2. Android MediaCodec + MediaExtractor pre M4A, MP3, OGG, FLAC, MP4.
 */
object AudioDecoder {
    const val TARGET_SAMPLE_RATE = 16000

/** Decode audio bytes to a 16 kHz mono FloatArray in the range [-1.0f, 1.0f]. */
    fun decodeToPcmFloats(audioBytes: ByteArray, tempDir: File): FloatArray {
        if (audioBytes.isEmpty()) return FloatArray(0)

// Try the fast WAV parser first
        if (isWavFormat(audioBytes)) {
            val wavFloats = parseWavToFloats(audioBytes)
            if (wavFloats != null) return wavFloats
        }

// If not WAV or a different sample rate, use Android MediaCodec via a temp file
        val tempFile = File(tempDir, "temp_audio_${System.currentTimeMillis()}.tmp")
        try {
            FileOutputStream(tempFile).use { it.write(audioBytes) }
            return decodeFileWithMediaCodec(tempFile)
        } catch (e: Exception) {
            e.printStackTrace()
            return FloatArray(0)
        } finally {
            tempFile.delete()
        }
    }

    private fun isWavFormat(bytes: ByteArray): Boolean {
        if (bytes.size < 12) return false
        val riff = String(bytes, 0, 4)
        val wave = String(bytes, 8, 4)
        return riff == "RIFF" && wave == "WAVE"
    }

    private fun parseWavToFloats(bytes: ByteArray): FloatArray? {
        try {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            var pos = 12 // po RIFF WAVE

            var audioFormat = 1
            var numChannels = 1
            var sampleRate = 16000
            var bitsPerSample = 16
            var dataOffset = -1
            var dataSize = 0

            while (pos < bytes.size - 8) {
                val chunkId = String(bytes, pos, 4)
                val chunkSize = buffer.getInt(pos + 4)
                pos += 8

                if (chunkId == "fmt ") {
                    audioFormat = buffer.getShort(pos).toInt()
                    numChannels = buffer.getShort(pos + 2).toInt()
                    sampleRate = buffer.getInt(pos + 4)
                    bitsPerSample = buffer.getShort(pos + 14).toInt()
                } else if (chunkId == "data") {
                    dataOffset = pos
                    dataSize = chunkSize
                    break
                }
                pos += chunkSize
            }

            if (dataOffset == -1 || audioFormat != 1 || bitsPerSample != 16) {
return null // Unsupported WAV shape, fall back to MediaCodec
            }

            val numSamples = dataSize / 2
            val rawShorts = ShortArray(numSamples)
            val pcmBuf = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until numSamples) {
                rawShorts[i] = pcmBuf.short
            }

        // Downmix to mono if stereo
            val monoShorts = if (numChannels > 1) {
                val frames = numSamples / numChannels
                ShortArray(frames) { i ->
                    var sum = 0
                    for (ch in 0 until numChannels) {
                        sum += rawShorts[i * numChannels + ch]
                    }
                    (sum / numChannels).toShort()
                }
            } else {
                rawShorts
            }

// Resample to 16 kHz if the sample rate differs
            val finalShorts = if (sampleRate != TARGET_SAMPLE_RATE) {
                resampleLinear(monoShorts, sampleRate, TARGET_SAMPLE_RATE)
            } else {
                monoShorts
            }

        // Convert short to normalized float [-1.0, 1.0]
            return FloatArray(finalShorts.size) { i -> finalShorts[i] / 32768.0f }
        } catch (e: Exception) {
            return null
        }
    }

    private fun decodeFileWithMediaCodec(file: File): FloatArray {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        val pcmBytes = mutableListOf<Byte>()
        var sampleRate = TARGET_SAMPLE_RATE
        var channels = 1

        try {
            extractor.setDataSource(file.absolutePath)
            var audioTrackIndex = -1
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    break
                }
            }
            if (audioTrackIndex == -1) return FloatArray(0)

            extractor.selectTrack(audioTrackIndex)
            val format = extractor.getTrackFormat(audioTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
            sampleRate = if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) format.getInteger(MediaFormat.KEY_SAMPLE_RATE) else 16000
            channels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val info = MediaCodec.BufferInfo()
            var isEOS = false

            while (!isEOS) {
                val inIndex = codec.dequeueInputBuffer(10000)
                if (inIndex >= 0) {
                    val buffer = codec.getInputBuffer(inIndex)
                    val sampleSize = extractor.readSampleData(buffer!!, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        isEOS = true
                    } else {
                        val presentationTimeUs = extractor.sampleTime
                        codec.queueInputBuffer(inIndex, 0, sampleSize, presentationTimeUs, 0)
                        extractor.advance()
                    }
                }

                var outIndex = codec.dequeueOutputBuffer(info, 10000)
                while (outIndex >= 0) {
                    val outBuffer = codec.getOutputBuffer(outIndex)
                    val chunk = ByteArray(info.size)
                    outBuffer?.get(chunk)
                    outBuffer?.clear()
                    pcmBytes.addAll(chunk.toList())
                    codec.releaseOutputBuffer(outIndex, false)
                    outIndex = codec.dequeueOutputBuffer(info, 0)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            codec?.stop()
            codec?.release()
            extractor.release()
        }

        if (pcmBytes.isEmpty()) return FloatArray(0)

// Convert decoded 16-bit PCM bytes to shorts
        val byteArr = pcmBytes.toByteArray()
        val numShorts = byteArr.size / 2
        val shorts = ShortArray(numShorts)
        ByteBuffer.wrap(byteArr).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts)

        // Downmix to mono if stereo
        val monoShorts = if (channels > 1) {
            val frames = numShorts / channels
            ShortArray(frames) { i ->
                var sum = 0
                for (ch in 0 until channels) {
                    sum += shorts[i * channels + ch]
                }
                (sum / channels).toShort()
            }
        } else {
            shorts
        }

// Linear resampling to 16 kHz if needed
        val resampled = if (sampleRate != TARGET_SAMPLE_RATE) {
            resampleLinear(monoShorts, sampleRate, TARGET_SAMPLE_RATE)
        } else {
            monoShorts
        }

        return FloatArray(resampled.size) { i -> resampled[i] / 32768.0f }
    }

    private fun resampleLinear(input: ShortArray, fromRate: Int, toRate: Int): ShortArray {
        if (fromRate == toRate || input.isEmpty()) return input
        val ratio = fromRate.toDouble() / toRate.toDouble()
        val outputSize = (input.size / ratio).toInt()
        val output = ShortArray(outputSize)

        for (i in 0 until outputSize) {
            val srcPos = i * ratio
            val srcIndex = srcPos.toInt()
            val frac = srcPos - srcIndex

            if (srcIndex >= input.size - 1) {
                output[i] = input.last()
            } else {
                val s1 = input[srcIndex]
                val s2 = input[srcIndex + 1]
                output[i] = (s1 + frac * (s2 - s1)).toInt().coerceIn(-32768, 32767).toShort()
            }
        }
        return output
    }
}
