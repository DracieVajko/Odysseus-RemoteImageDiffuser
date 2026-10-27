package eu.odysseus.diffusion

import android.util.Log

/**
 * MP3 encoding through libmp3lame (LGPL, shipped as libmp3_jni.so).
 *
 * Android's MediaCodec has no MP3 encoder, and bundling ffmpeg just for this
 * would add tens of megabytes and a GPL-3 obligation. LAME is ~300 KB per ABI
 * and covers the encoder.
 */
object Mp3Encoder {
    private const val TAG = "Mp3Encoder"

    @Volatile
    var available: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("mp3_jni")
            available = true
            Log.i(TAG, "libmp3_jni.so loaded")
        } catch (e: Throwable) {
            available = false
            Log.w(TAG, "libmp3_jni.so not available: ${e.message}")
        }
    }

    /**
     * Encode interleaved 16-bit PCM to MP3.
     * @param bitrateKbps 128, 192 or 320
     * @return MP3 bytes or null on failure/unsupported build.
     */
    fun encode(pcm: ShortArray, sampleRate: Int, channels: Int, bitrateKbps: Int): ByteArray? {
        if (!available) return null
        if (pcm.isEmpty() || sampleRate <= 0 || (channels != 1 && channels != 2)) return null
        return try {
            nativeEncode(pcm, sampleRate, channels, bitrateKbps)
        } catch (e: Throwable) {
            Log.e(TAG, "encode failed: ${e.message}")
            null
        }
    }

    private external fun nativeEncode(
        pcm: ShortArray,
        sampleRate: Int,
        channels: Int,
        bitrateKbps: Int,
    ): ByteArray?
}
