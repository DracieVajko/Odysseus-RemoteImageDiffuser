package eu.odysseus.diffusion

import android.content.Context
import android.util.Log
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Remote proxy — keď je telefón v Remote/Auto režime a lokálny model
 * nie je pripravený (503 will-not-load), preposiela request na PC server.
 * Používa ktor client + JSON. V Auto režime sa najprv skúsi Local
 * (api.ktor), pri 503 sa transparentne prepnú na Remote.
 */
object RemoteProxy {
    private const val TAG = "RemoteProxy"
    private val client = HttpClient(OkHttp) { expectSuccess = false }

    fun health(remoteBase: String, scope: CoroutineScope, onResult: (Boolean) -> Unit) {
        scope.launch(Dispatchers.IO) {
            try {
                val r = client.get(remoteBase.trimEnd('/') + "/health")
                onResult(r.status.value in 200..299)
            } catch (e: Exception) {
                Log.w(TAG, "health failed: ${e.message}")
                onResult(false)
            }
        }
    }

    suspend fun get(remoteBase: String, path: String): String? = try {
        client.get(remoteBase.trimEnd('/') + path).bodyAsText()
    } catch (e: Exception) {
        Log.w(TAG, "get $path failed: ${e.message}")
        null
    }

    suspend fun postJson(remoteBase: String, path: String, body: String): String? = try {
        client.post(remoteBase.trimEnd('/') + path) {
            contentType(ContentType.Application.Json)
            setBody(body)
        }.bodyAsText()
    } catch (e: Exception) {
        Log.w(TAG, "post $path failed: ${e.message}")
        null
    }
}
