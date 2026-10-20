package eu.odysseus.diffusion

import android.util.Log
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.contentType
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.server.request.queryString
import io.ktor.server.request.receiveChannel
import io.ktor.server.response.respondBytes
import io.ktor.utils.io.readRemaining
import io.ktor.utils.io.core.readBytes
import java.net.HttpURLConnection
import java.net.URL

/**
 * Server-side remote fallback: keď lokálny engine na mobile nevie danú
 * operáciu (image gen / upscale / caption / ffmpeg), pošle request na PC
 * a vráti jeho odpoveď. Audio ide vždy lokálne (whisper JNI).
 *
 * Multipart sa forwarduje ako raw bajty s pôvodným Content-Type.
 */
object ProxyForward {
    private const val TAG = "ProxyForward"

    // Endpointy, ktoré mobil lokálne nevie (alebo sú PC-only).
    private val PROXY_PREFIXES = listOf(
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/images/inpaint",
        "/v1/images/harmonize",
        "/v1/images/upscale",
        "/v1/images/caption",
        "/v1/chat/",
        "/v1/ffmpeg/",
    )

    fun shouldProxy(path: String, method: String): Boolean {
        if (method != "POST") return false
        val mode = AppConfig.mode
        if (mode !in setOf("REMOTE", "AUTO")) return false
        if (AppConfig.remoteBase.isBlank()) return false
        return PROXY_PREFIXES.any { path.startsWith(it) }
    }

    /** Vráti true, ak request bol forwardovaný (a odpoveď už odoslaná). */
    suspend fun forward(call: ApplicationCall): Boolean {
        val base = AppConfig.remoteBase.trimEnd('/')
        if (base.isBlank()) return false
        val path = call.request.path()
        val query = call.request.queryString()
        val target = base + path + if (query.isNotBlank()) "?$query" else ""
        val method = call.request.httpMethod.value

        val body: ByteArray = try {
            call.receiveChannel().readRemaining().readBytes()
        } catch (e: Exception) {
            ByteArray(0)
        }
        val contentType = call.request.headers[HttpHeaders.ContentType]

        return try {
            val conn = URL(target).openConnection() as HttpURLConnection
            conn.requestMethod = method
            conn.connectTimeout = 5000
            conn.readTimeout = 600000
            conn.doOutput = body.isNotEmpty()
            if (contentType != null) conn.setRequestProperty("Content-Type", contentType)
            if (body.isNotEmpty()) {
                conn.outputStream.use { it.write(body) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val out = stream?.use { it.readBytes() } ?: ByteArray(0)
            val respType = conn.contentType?.let {
                runCatching { ContentType.parse(it) }.getOrNull()
            } ?: ContentType.Application.Json
            conn.disconnect()
            Log.i(TAG, "proxied $method $path -> $code (${out.size} B)")
            call.respondBytes(out, respType, io.ktor.http.HttpStatusCode.fromValue(code))
            true
        } catch (e: Exception) {
            Log.w(TAG, "proxy failed for $path: ${e.message}")
            false
        }
    }
}
