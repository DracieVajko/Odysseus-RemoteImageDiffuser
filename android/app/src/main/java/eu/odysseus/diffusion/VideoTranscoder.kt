package eu.odysseus.diffusion

import android.graphics.SurfaceTexture
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.view.Surface
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * On-device video re-encode (compress) using MediaCodec + EGL/GL bridge.
 *
 * Best-effort: any failure is converted into [MediaTools.Unsupported] so the
 * API answers an honest 503 instead of crashing the server process.
 */
object VideoTranscoder {

    private const val VERTEX_SHADER = """
        attribute vec4 aPosition;
        attribute vec4 aTexCoord;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord.xy;
        }
    """

    private const val FRAGMENT_SHADER = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;
        varying vec2 vTexCoord;
        uniform samplerExternalOES uTexture;
        void main() {
            gl_FragColor = texture2D(uTexture, vTexCoord);
        }
    """

    private val QUAD_VERTICES = floatArrayOf(
        -1f, -1f, 0f, 0f,
         1f, -1f, 1f, 0f,
        -1f,  1f, 0f, 1f,
         1f,  1f, 1f, 1f,
    )

    fun compress(input: File, preset: String): MediaTools.Output {
        val targetSizeMb = when (preset) {
            "web-720p", "mp4-small" -> 5f
            else -> 10f
        }

        val extractor = MediaExtractor()
        var muxer: MediaMuxer? = null
        var decoder: MediaCodec? = null
        var encoder: MediaCodec? = null
        var encoderInputSurface: Surface? = null
        var decoderOutputSurface: Surface? = null
        var surfaceTexture: SurfaceTexture? = null
        var eglDisplay: EGLDisplay? = null
        var eglContext: EGLContext? = null
        var eglSurface: EGLSurface? = null
        val outFile = File(input.parentFile, "mt_compress.mp4")

        try {
            extractor.setDataSource(input.absolutePath)

            var videoTrackIndex = -1
            var videoFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("video/")) { videoTrackIndex = i; videoFormat = f; break }
            }
            if (videoTrackIndex < 0 || videoFormat == null)
                throw MediaTools.Unsupported("no video track")

            val mime = videoFormat.getString(MediaFormat.KEY_MIME)!!
            val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
            val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
            val frameRate = try { videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE) } catch (_: Exception) { 30 }

            var durationUs = try { videoFormat.getLong(MediaFormat.KEY_DURATION) } catch (_: Exception) { 0L }
            if (durationUs <= 0) {
                extractor.selectTrack(videoTrackIndex)
                extractor.seekTo(Long.MAX_VALUE, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                val lastPts = extractor.sampleTime
                if (lastPts > 0) durationUs = lastPts
            }
            if (durationUs <= 0) throw MediaTools.Unsupported("could not determine video duration")

            val durationS = durationUs / 1_000_000.0
            val targetBits = targetSizeMb * 1024 * 1024 * 8
            val audioBitrate = 128000
            val videoBitrate = ((targetBits / durationS) - audioBitrate).toInt().coerceAtLeast(100000)

            extractor.selectTrack(videoTrackIndex)

            // --- decoder ---
            val textures = IntArray(1)
            android.opengl.GLES20.glGenTextures(1, textures, 0)
            val textureId = textures[0]
            android.opengl.GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
            android.opengl.GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, android.opengl.GLES20.GL_TEXTURE_MIN_FILTER, android.opengl.GLES20.GL_LINEAR)
            android.opengl.GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, android.opengl.GLES20.GL_TEXTURE_MAG_FILTER, android.opengl.GLES20.GL_LINEAR)
            android.opengl.GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, android.opengl.GLES20.GL_TEXTURE_WRAP_S, android.opengl.GLES20.GL_CLAMP_TO_EDGE)
            android.opengl.GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, android.opengl.GLES20.GL_TEXTURE_WRAP_T, android.opengl.GLES20.GL_CLAMP_TO_EDGE)

            surfaceTexture = SurfaceTexture(textureId)
            decoderOutputSurface = Surface(surfaceTexture)
            decoder = MediaCodec.createDecoderByType(mime)
            decoder!!.configure(videoFormat, decoderOutputSurface, null, 0)

            // --- encoder ---
            val encoderFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
            encoderFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            encoderFormat.setInteger(MediaFormat.KEY_BIT_RATE, videoBitrate)
            encoderFormat.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            encoderFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            encoder!!.configure(encoderFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoderInputSurface = encoder!!.createInputSurface()

            // --- muxer ---
            muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            // --- EGL ---
            eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val version = IntArray(2)
            if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1))
                throw MediaTools.Unsupported("EGL init failed")

            val attribList = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, 1, numConfigs, 0) || numConfigs[0] == 0)
                throw MediaTools.Unsupported("EGL config failed")
            val eglConfig = configs[0]!!

            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            if (eglContext == EGL14.EGL_NO_CONTEXT) throw MediaTools.Unsupported("EGL context failed")

            eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, encoderInputSurface, intArrayOf(EGL14.EGL_NONE), 0)
            if (eglSurface == EGL14.EGL_NO_SURFACE) throw MediaTools.Unsupported("EGL window surface failed")
            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
                throw MediaTools.Unsupported("EGL makeCurrent failed")

            // Compile shaders.
            val program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
            if (program == 0) throw MediaTools.Unsupported("GL shader compile failed")
            GLES20.glUseProgram(program)
            val posLoc = GLES20.glGetAttribLocation(program, "aPosition")
            val texLoc = GLES20.glGetAttribLocation(program, "aTexCoord")
            val texUniform = GLES20.glGetUniformLocation(program, "uTexture")

            val vertexBuffer: FloatBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.size * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
            vertexBuffer.put(QUAD_VERTICES).position(0)

            // --- start codecs ---
            decoder!!.start()
            encoder!!.start()

            var videoMuxTrack = -1
            var muxStarted = false
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var frameAvailable = false

            surfaceTexture!!.setOnFrameAvailableListener { frameAvailable = true }

            while (!outputDone) {
                // Feed input to decoder.
                if (!inputDone) {
                    val inIdx = decoder!!.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = decoder!!.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            decoder!!.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val pts = extractor.sampleTime
                            decoder!!.queueInputBuffer(inIdx, 0, size, pts, 0)
                            extractor.advance()
                        }
                    }
                }

                // Get decoder output.
                val decInfo = MediaCodec.BufferInfo()
                val decIdx = decoder!!.dequeueOutputBuffer(decInfo, 10_000)
                if (decIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // ignore
                } else if (decIdx >= 0) {
                    val doRender = decInfo.size > 0
                    decoder!!.releaseOutputBuffer(decIdx, doRender)
                    if (doRender) {
                        var waited = 0
                        while (!frameAvailable && waited < 100) { Thread.sleep(10); waited++ }
                        frameAvailable = false
                        surfaceTexture!!.updateTexImage()

                        GLES20.glViewport(0, 0, width, height)
                        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
                        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
                        GLES20.glUniform1i(texUniform, 0)

                        vertexBuffer.position(0)
                        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer)
                        GLES20.glEnableVertexAttribArray(posLoc)
                        vertexBuffer.position(2)
                        GLES20.glVertexAttribPointer(texLoc, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer)
                        GLES20.glEnableVertexAttribArray(texLoc)

                        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
                        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
                    }
                    if (decInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        encoder!!.signalEndOfInputStream()
                    }
                }

                // Get encoder output.
                val encIdx = encoder!!.dequeueOutputBuffer(bufferInfo, 10_000)
                if (encIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val newFormat = encoder!!.outputFormat
                    videoMuxTrack = muxer!!.addTrack(newFormat)
                    muxer!!.start()
                    muxStarted = true
                } else if (encIdx >= 0) {
                    val buf = encoder!!.getOutputBuffer(encIdx)!!
                    if (muxStarted && bufferInfo.size > 0) {
                        buf.position(bufferInfo.offset)
                        buf.limit(bufferInfo.offset + bufferInfo.size)
                        muxer!!.writeSampleData(videoMuxTrack, buf, bufferInfo)
                    }
                    encoder!!.releaseOutputBuffer(encIdx, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
            }

            val bytes = outFile.readBytes()
            outFile.delete()
            return MediaTools.Output(bytes, "compressed.mp4", "video/mp4")

        } catch (e: Exception) {
            try { outFile.delete() } catch (_: Exception) {}
            if (e is MediaTools.Unsupported) throw e
            throw MediaTools.Unsupported("video re-encode failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            try { decoder?.stop() } catch (_: Exception) {}
            try { decoder?.release() } catch (_: Exception) {}
            try { encoder?.stop() } catch (_: Exception) {}
            try { encoder?.release() } catch (_: Exception) {}
            try { muxer?.stop() } catch (_: Exception) {}
            try { muxer?.release() } catch (_: Exception) {}
            try { encoderInputSurface?.release() } catch (_: Exception) {}
            try { decoderOutputSurface?.release() } catch (_: Exception) {}
            try { surfaceTexture?.release() } catch (_: Exception) {}
            try { extractor.release() } catch (_: Exception) {}
            try { if (eglDisplay != null && eglSurface != null) EGL14.eglDestroySurface(eglDisplay, eglSurface) } catch (_: Exception) {}
            try { if (eglDisplay != null && eglContext != null) EGL14.eglDestroyContext(eglDisplay, eglContext) } catch (_: Exception) {}
            try { if (eglDisplay != null) EGL14.eglTerminate(eglDisplay) } catch (_: Exception) {}
        }
    }

    private fun createProgram(vertexShader: String, fragmentShader: String): Int {
        val vs = loadShader(GLES20.GL_VERTEX_SHADER, vertexShader)
        val fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShader)
        if (vs == 0 || fs == 0) return 0
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glLinkProgram(program)
        val linkStatus = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] == 0) {
            GLES20.glDeleteProgram(program)
            return 0
        }
        return program
    }

    private fun loadShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            GLES20.glDeleteShader(shader)
            return 0
        }
        return shader
    }
}
