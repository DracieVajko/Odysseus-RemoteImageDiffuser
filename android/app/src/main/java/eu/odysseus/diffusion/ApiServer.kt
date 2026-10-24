package eu.odysseus.diffusion

import io.ktor.http.HttpStatusCode
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import io.ktor.serialization.gson.gson
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.netty.NettyApplicationEngine
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.request.receive
import io.ktor.server.request.path
import io.ktor.server.request.httpMethod
import io.ktor.server.request.receiveMultipart
import io.ktor.server.request.receiveText
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.http.ContentType
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.http.content.PartData
import io.ktor.http.content.forEachPart
import io.ktor.utils.io.core.readBytes
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

// ---------------------------------------------------------------------------
// Contract: same paths + same 503 shapes as docs/ODYSSEUS-CONTRACT.md.
// Base: http://<phone>:8101
//
// Rozdiely (mobile V1, research verdikty):
// - quality→steps is capped to 4..8 (LCM mode); more = OOM on Adreno/Mali.
// - backend NCNN + Vulkan (not NNAPI).
// Stav inferencie (mobile V1):
// - audio: REAL (AudioDecoder → WhisperLib JNI → libwhisper.so from the CMake build).
// - image: REAL via ONNX LCM (QDQ UNet); upscale/caption/chat are experimental.
//   See docs/ONDEVICE-PLAN.md for the on-device roadmap.
// ---------------------------------------------------------------------------

const val SERVER_PORT = 8101
const val MOBILE_STEPS_MIN = 4
const val MOBILE_STEPS_MAX = 8
const val MAX_CANVAS_PX = 8192
const val MAX_AUDIO_BYTES = 100L * 1024L * 1024L // 100 MB, same as WhisperApi

/** quality → steps, then clamp to [MOBILE_STEPS_MIN, MOBILE_STEPS_MAX]. */
fun mobileSteps(quality: String?, steps: Int?): Int {
    val base = when (steps) {
        null -> when (quality ?: "medium") {
            "low" -> 4
            "high" -> 30
            "auto" -> 12
        else -> 25 // medium = profile default (mirrors the server)
        }
        else -> steps
    }
    return base.coerceIn(MOBILE_STEPS_MIN, MOBILE_STEPS_MAX)
}

@Serializable
data class GenerationReq(
    val model: String = "",
    val prompt: String = "",
    val negative_prompt: String = "",
    val guidance: Float = 1.0f,
    val n: Int = 1,
    val size: String = "512x512",
    val quality: String = "medium",
    val steps: Int? = null,
    val request_id: String = "",
)

@Serializable
data class InpaintReq(
    val image: String = "",
    val mask: String = "",
    val prompt: String = "",
    val negative_prompt: String = "",
    val guidance: Float = 1.0f,
    val mode: String = "add",
    val color: String = "#0066FF",
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int? = null,
    val strength: Double = 0.8,
    val feather: Int = 16,
    val model: String = "",
    val request_id: String = "",
)

@Serializable
data class HarmonizeReq(
    val image: String = "",
    val prompt: String = "",
    val color_match: Boolean = true,
    val seam_fix: Boolean = true,
    val body_mask: String? = null,
    val seam_mask: String? = null,
    val mask: String? = null,
    val strength: Double = 0.6,
    val max_side: Int = 1024,
    val steps: Int? = null,
    val model: String = "",
    val request_id: String = "",
)

@Serializable
data class UpscaleReq(
    val image: String = "",
    val scale: Int = 2,
    val model: String = "realesrgan-x4plus-ncnn",
    val request_id: String = "",
)

@Serializable
data class CaptionReq(
    val image: String = "",
    val prompt: String = "Describe the image in detail.",
    val max_tokens: Int = 256,
    val model: String = "",
    val request_id: String = "",
)

enum class JobState { RUNNING, LOADING, DONE, ERROR, UNKNOWN }

data class JobProgress(
    val id: String,
    var status: JobState = JobState.RUNNING,
    var step: Int = 0,
    var total: Int = MOBILE_STEPS_MAX,
    var percent: Int = 0,
)

/** Safe JSON response: Map<String, Any> cannot be serialized by kotlinx-serialization
*  (Any has no serializer) and Ktor then returns a silent 500.
*  jsonOf() recursively converts Map/List/primitives to Gson JsonElement.
*  NOTE: we use Gson tree types (not just the Gson serializer) because Gson cannot
*  serialize kotlinx JsonElement — via reflection it would produce
*  {"content": "...", "isString": true} instead of the raw JSON shape. */
fun anyToJson(v: Any?): JsonElement = when (v) {
    null -> JsonNull.INSTANCE
    is JsonElement -> v
    is Boolean -> JsonPrimitive(v)
    is Number -> JsonPrimitive(v)
    is String -> JsonPrimitive(v)
    is Map<*, *> -> JsonObject().apply {
        v.forEach { (k, vv) -> add(k.toString(), anyToJson(vv)) }
    }
    is Iterable<*> -> JsonArray().apply {
        v.forEach { add(anyToJson(it)) }
    }
    else -> JsonPrimitive(v.toString())
}

fun jsonOf(vararg pairs: Pair<String, Any?>): JsonObject =
    JsonObject().apply {
        pairs.forEach { (k, v) -> add(k, anyToJson(v)) }
    }

/** Request body: kotlinx-serialization (defaults + ignoreUnknownKeys), not Gson.
*  Gson ignores Kotlin defaults via reflection and leaves missing String fields null. */
private val wireJson = Json {
    ignoreUnknownKeys = true
    isLenient = true
    coerceInputValues = true
}

class ApiServer(
    private val manager: ModelManager,
    private val downloader: ModelDownloader,
    private val appContext: android.content.Context? = null,
) {
    private val progress = ConcurrentHashMap<String, JobProgress>()
    private var engine: NettyApplicationEngine? = null

    fun start(wait: Boolean = false): NettyApplicationEngine {
// Explicit 0.0.0.0: the default bind may be IPv6-only ([::]), which adb forward
// (IPv4 127.0.0.1) cannot reach — requests then hang/500.
        val eng = embeddedServer(Netty, host = "0.0.0.0", port = SERVER_PORT) { module() }
        engine = eng
        eng.start(wait = wait)
        return eng
    }

    fun stop() {
        engine?.stop(1000, 2000)
        engine = null
    }

    private fun Application.module() {
        // Remote fallback: endpoint handlers handle auto-forward via try-catch on ModelWillNotLoadError
// Audio stays local (JNI). Image generation is local too (ONNX).
        install(ContentNegotiation) {
            gson {
                setPrettyPrinting()
                setLenient()
            }
        }
        install(StatusPages) {
            exception<Throwable> { call, e ->
                android.util.Log.e("ApiServer", "unhandled: " + e.message, e)
                call.respond(
                    HttpStatusCode.InternalServerError,
                    jsonOf("error" to (e.message ?: e.javaClass.simpleName)),
                )
            }
            exception<ModelLoadingError> { call, e ->
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    jsonOf(
                        "error" to "loading model",
                        "loading_model" to e.loadingModel,
                        "retry_after" to e.retryAfterSec,
                    ),
                )
            }
            exception<ModelWillNotLoadError> { call, e ->
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    jsonOf(
                        "error" to "will not load",
                        "reason" to e.reason,
                        "needed_mb" to e.neededMb,
                        "free_mb" to e.freeMb,
                    ),
                )
            }
        }


        // Track in-flight image/audio requests for /health busy/queue.
        intercept(io.ktor.server.application.ApplicationCallPipeline.Monitoring) {
            val path = call.request.path()
            val track = path.startsWith("/v1/images/") || path.startsWith("/v1/audio/")
            if (track) manager.beginRequest()
            try {
                proceed()
            } finally {
                if (track) manager.endRequest()
            }
        }
        routing {
            get("/health") {
                val st = manager.status()
                val all = downloader.allAssets()
                call.respond(
                    jsonOf(
                        "status" to "ok",
                        "model" to (st.loadedId ?: ""),
                        "current_model" to (st.loadedId ?: ""),
                        "device" to "vulkan-ncnn",
                        "available_models" to all.filter { downloader.isReady(it.id) }.map { it.id },
                        "catalog_models" to all.map { it.id },
                        "loading" to st.loading,
                        "loading_model" to st.loadingModel,
                        "busy" to st.busy,
                        "queue" to st.queue,
                    ),
                )
            }
            get("/v1/models") {
                val st = manager.status()
                val all = downloader.allAssets()
                val data = all.map { a ->
                    jsonOf(
                        "id" to a.id,
                        "object" to "model",
                        "owned_by" to "odysseus-local",
                        "kind" to a.kind,
                        "label" to a.label,
                        "engine" to a.engine,
                        "available" to downloader.isReady(a.id),
                        "has_source" to a.hasSource,
                        "runs_on_device" to a.runsOnDevice,
                        "installed_mb" to downloader.installedSizeMb(a.id),
                    )
                }
                call.respond(
                    jsonOf(
                        "object" to "list",
                        "data" to data,
                        "currently_loaded_model" to (st.loadedId ?: ""),
                        "loading" to st.loading,
                        "loading_model" to st.loadingModel,
// warning only for models that HAVE a source and are not downloaded
                        "missing_models" to all
                            .filter { it.hasSource && !downloader.isReady(it.id) }
                            .map { it.id },
                        "default_audio_model" to "whisper-large-v3-turbo-q5_0",
                        "audio_models" to all.filter { it.kind == "audio" }.map { it.id },
                        "kinds" to ModelCatalog.kinds.map { k ->
                            jsonOf("id" to k, "label" to ModelCatalog.kindLabel(k))
                        },
                    ),
                )
            }
            get("/v1/images/progress/{id}") {
                val id = call.parameters["id"] ?: ""
                val j = progress[id]
                if (j == null) {
                    call.respond(jsonOf("id" to id, "status" to "unknown", "step" to 0, "total" to 0, "percent" to 0))
                } else {
                    call.respond(
                        jsonOf(
                            "id" to j.id,
                            "status" to j.status.name.lowercase(),
                            "step" to j.step,
                            "total" to j.total,
                            "percent" to j.percent,
                        ),
                    )
                }
            }
            get("/v1/models/find") {
                val q = call.request.queryParameters["q"] ?: ""
                val limit = (call.request.queryParameters["limit"] ?: "15").toIntOrNull() ?: 15
                if (q.isBlank()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "missing q"))
                    return@get
                }
                try {
                    val url = java.net.URL(
                        "https://huggingface.co/api/models?search=" +
                            java.net.URLEncoder.encode(q, "UTF-8") + "&limit=" + limit
                    )
                    val conn = url.openConnection() as java.net.HttpURLConnection
                    conn.connectTimeout = 8000
                    conn.readTimeout = 12000
                    val body = conn.inputStream.bufferedReader().use { it.readText() }
                    conn.disconnect()
                    val arr = org.json.JSONArray(body)
                    val out = ArrayList<JsonElement>()
                    for (i in 0 until arr.length()) {
                        val o = arr.getJSONObject(i)
                        out.add(
                            jsonOf(
                                "id" to o.optString("id"),
                                "downloads" to o.optInt("downloads"),
                                "likes" to o.optInt("likes"),
                                "pipeline_tag" to o.optString("pipeline_tag"),
                                "library" to o.optString("library_name"),
                            )
                        )
                    }
                    call.respond(jsonOf("query" to q, "count" to out.size, "models" to out))
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadGateway, jsonOf("error" to (e.message ?: "hf search failed")))
                }
            }
// Custom model: the user provides a direct URL (GGUF/ONNX/bin/safetensors)
// and the model is stored in models/<name>/ and downloaded immediately.
            post("/v1/models/add") {
                val body = call.receiveText()
                fun field(k: String): String? =
                    Regex("\"$k\"\\s*:\\s*\"([^\"]*)\"").find(body)?.groupValues?.get(1)
                val url = field("url")
                if (url.isNullOrBlank()) {
                    call.respond(
                        HttpStatusCode.BadRequest,
                        jsonOf("error" to "missing url (direct file URL, e.g. HF resolve/main/...)"),
                    )
                    return@post
                }
                val name = field("name")
                    ?: url.substringAfterLast('/').substringBefore('?').substringBeforeLast('.')
                val kind = field("kind") ?: "custom"
                try {
                    val asset = downloader.addCustom(name, kind, url, field("label") ?: "")
                    call.respond(
                        jsonOf(
                            "id" to asset.id,
                            "name" to asset.id,
                            "kind" to asset.kind,
                            "state" to "downloading",
                            "path" to File(downloader.modelsDir(), asset.id).absolutePath,
                        ),
                    )
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to (e.message ?: "add failed")))
                }
            }
            get("/v1/models/custom") {
                val root = downloader.modelsDir()
                val rows = downloader.customAssets().map { a ->
                    jsonOf(
                        "id" to a.id,
                        "name" to a.id,
                        "kind" to a.kind,
                        "url" to a.url,
                        "size_mb" to downloader.installedSizeMb(a.id),
                        "installed" to downloader.isReady(a.id),
                        "state" to (downloader.progress(a.id)?.state?.name?.lowercase() ?: "idle"),
                        "path" to File(root, a.id).absolutePath,
                    )
                }
                call.respond(jsonOf("models" to rows, "models_dir" to root.absolutePath))
            }
            post("/v1/models/delete") {
                val body = call.receiveText()
                val name = Regex("\"name\"\\s*:\\s*\"([^\"]+)\"").find(body)?.groupValues?.get(1)
                    ?: Regex("\"id\"\\s*:\\s*\"([^\"]+)\"").find(body)?.groupValues?.get(1)
                if (name.isNullOrBlank() || name.contains("/") || name.contains("\\") || name == "..") {
                    return@post call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "invalid model name"))
                }
                if (ModelCatalog.find(name) != null) {
                    return@post call.respond(
                        HttpStatusCode.BadRequest,
                        jsonOf("error" to "catalog model: use uninstall instead"),
                    )
                }
                if (downloader.findAsset(name) == null) {
                    return@post call.respond(HttpStatusCode.NotFound, jsonOf("error" to "model not found: $name"))
                }
                val ok = downloader.uninstall(name)
                call.respond(jsonOf("name" to name, "removed" to ok))
            }
            get("/v1/models/catalog") {
                val rows = downloader.allAssets().map { a ->
                    val p = downloader.progress(a.id)
                    jsonOf(
                        "id" to a.id,
                        "kind" to a.kind,
                        "kind_label" to ModelCatalog.kindLabel(a.kind),
                        "label" to a.label,
                        "engine" to a.engine,
                        "note" to a.note,
                        "custom" to a.custom,
                        "runs_on_device" to a.runsOnDevice,
                        "size_mb" to a.sizeMb,
                        "size_gb" to (a.sizeMb / 1024.0),
                        "files" to a.files.map { it.path },
                        "installed" to downloader.isReady(a.id),
                        "installed_gb" to (downloader.installedSizeMb(a.id) / 1024.0),
                        "install_state" to (p?.state?.name?.lowercase() ?: if (downloader.isReady(a.id)) "done" else "idle"),
                        "install_progress" to (p?.percent ?: 0),
                        "install_error" to (p?.error ?: ""),
                        "url" to a.url,
                        "has_source" to a.hasSource,
                        "downloadable" to a.hasSource,
                    )
                }
                call.respond(
                    jsonOf(
                        "models" to rows,
                        "kinds" to ModelCatalog.kinds.map { k ->
                            jsonOf("id" to k, "label" to ModelCatalog.kindLabel(k))
                        },
                    ),
                )
            }
            get("/v1/models/progress/{id}") {
                val id = call.parameters["id"] ?: ""
                val p = downloader.progress(id)
                    ?: return@get call.respond(
                        jsonOf("id" to id, "state" to "idle", "percent" to 0, "bytes_done" to 0, "bytes_total" to 0),
                    )
                call.respond(
                    jsonOf(
                        "id" to id,
                        "state" to p.state.name.lowercase(),
                        "percent" to p.percent,
                        "bytes_done" to p.bytesDone,
                        "bytes_total" to p.bytesTotal,
                        "file_index" to p.fileIndex,
                        "file_count" to p.fileCount,
                        "error" to p.error,
                    ),
                )
            }
            post("/v1/models/cancel") {
                val body = call.receiveText()
                val id = Regex("\"id\"\\s*:\\s*\"([^\"]+)\"").find(body)?.groupValues?.get(1)
                    ?: return@post call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "missing id"))
                downloader.cancel(id)
                call.respond(jsonOf("id" to id, "state" to "cancelled"))
            }
            post("/v1/models/install") {
                val body = call.receiveText()
                val id = Regex("\"id\"\\s*:\\s*\"([^\"]+)\"").find(body)?.groupValues?.get(1)
                    ?: return@post call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "missing id"))
                val asset = downloader.findAsset(id)
                    ?: return@post call.respond(HttpStatusCode.NotFound, jsonOf("error" to "unknown model id: $id"))
                if (!asset.hasSource) {
                    return@post call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "no verified source for $id"))
                }
                try {
                    downloader.enqueue(asset)
                    call.respond(jsonOf("id" to id, "state" to "downloading", "size_mb" to asset.sizeMb))
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to (e.message ?: "enqueue failed")))
                }
            }
            post("/v1/models/uninstall") {
                val body = call.receiveText()
                val id = Regex("\"id\"\\s*:\\s*\"([^\"]+)\"").find(body)?.groupValues?.get(1)
                    ?: return@post call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "missing id"))
                if (downloader.findAsset(id) == null) {
                    return@post call.respond(HttpStatusCode.NotFound, jsonOf("error" to "unknown model id: $id"))
                }
                val ok = downloader.uninstall(id)
                call.respond(jsonOf("id" to id, "removed" to ok))
            }
            get("/v1/settings") {
                val rt = Runtime.getRuntime()
                call.respond(
                    jsonOf(
                        "version" to "android-0.1.0",
                        "port" to SERVER_PORT,
                        "whisper_url" to "local (JNI libwhisper_jni.so)",
                        "models_dir" to downloader.modelsDir().absolutePath,
                        "default_image_model" to "lcm-dreamshaper-int8-onnx",
                        "audio_models" to downloader.allAssets().filter { it.kind == "audio" }.map { it.id },
                        "engines" to jsonOf(
                            "torch" to false,
                            "cuda" to false,
                            "onnxruntime" to true,
                            "whisper_native" to WhisperLib.isAvailable(),
                            "java_heap_free_mb" to (rt.freeMemory() / 1048576),
                            "java_heap_max_mb" to (rt.maxMemory() / 1048576),
                        ),
                    ),
                )
            }
// FFmpeg Tools: not available on device yet (MediaCodec covers audio
// decoding for whisper).
            get("/v1/ffmpeg/presets") {
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    jsonOf("error" to "ffmpeg tools run on PC (:8101) or via Remote mode"),
                )
            }
            post("/v1/ffmpeg/{op}") {
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    jsonOf("error" to "ffmpeg tools run on PC (:8101) or via Remote mode"),
                )
            }
            post("/v1/images/generations") {
                val req = wireJson.decodeFromString<GenerationReq>(call.receiveText())
                if (req.n !in 1..4) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "n must be 1..4"))
                    return@post
                }
                val (w, h) = parseSize(req.size) ?: run {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "bad size, expected WxH"))
                    return@post
                }
                if (maxOf(w, h) > MAX_CANVAS_PX) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "canvas too large"))
                    return@post
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val steps = mobileSteps(req.quality, req.steps).coerceIn(1, 8)
                val modelId = req.model.ifEmpty { "lcm-dreamshaper-int8-onnx" }
                if (!downloader.isReady(modelId)) {
                    throw ModelWillNotLoadError("model $modelId not downloaded (see ModelDownloader)", -1, -1)
                }
                val ctx = appContext
                if (ctx == null) {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val modelDir = downloader.modelDir(modelId)
                WhisperLib.release()
                if (!LcmOnnxEngine.isReady()) {
                    manager.setLoading(modelId, true)
                    try {
                        if (!LcmOnnxEngine.init(ctx, modelDir)) {
                            throw ModelWillNotLoadError("LCM engine init failed: " + LcmOnnxEngine.lastError, -1, -1)
                        }
                    } finally {
                        manager.setLoading(modelId, false)
                    }
                }
                // The ONNX engine is square; use the larger side and report the
                // actual size (the requested WxH may be non-square or > 512).
                val side = maxOf(w, h).coerceIn(256, 512)
                val images = ArrayList<String>(req.n)
                for (i in 0 until req.n) {
                    val bmp = LcmOnnxEngine.generate(
                        req.prompt, req.negative_prompt, req.guidance, steps, side,
                        seed = System.nanoTime() + i,
                    )
                    if (bmp == null) {
                        LcmOnnxEngine.release()
                        throw ModelWillNotLoadError("LCM generation failed: " + LcmOnnxEngine.lastError, -1, -1)
                    }
                    val baos = java.io.ByteArrayOutputStream()
                    bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, baos)
                    images.add(android.util.Base64.encodeToString(baos.toByteArray(), android.util.Base64.NO_WRAP))
                    bmp.recycle()
                }
                // Unload model after generation to free RAM (re-init on next request).
                LcmOnnxEngine.release()
                call.respond(
                    jsonOf(
                        "created" to System.currentTimeMillis() / 1000,
                        "data" to images.map { jsonOf("b64_json" to it) },
                        "request_id" to rid,
                        "steps" to steps,
                        "size" to "${side}x${side}",
                        "requested_size" to "${w}x${h}",
                    ),
                )
            }
            post("/v1/images/edits") {
// Multipart: image (binary) + prompt, model, n, size, quality, strength, request_id.
                var prompt = ""
                var model = ""
                var n = 1
                var size = "512x512"
                var quality = "medium"
                var requestId = ""
                var hasImage = false
                var imageBytes = ByteArray(0)
                call.receiveMultipart().forEachPart { part ->
                    when (part) {
                        is PartData.FormItem -> when (part.name) {
                            "prompt" -> prompt = part.value
                            "model" -> model = part.value
                            "n" -> n = part.value.toIntOrNull() ?: 1
                            "size" -> size = part.value
                            "quality" -> quality = part.value
                            "request_id" -> requestId = part.value
                        }
                        is PartData.FileItem -> if (part.name == "image") {
                            hasImage = true
                            imageBytes = part.provider().readBytes()
                        }
                        else -> {}
                    }
                    part.dispose()
                }
                if (!hasImage) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "missing image part"))
                    return@post
                }
                if (n !in 1..4) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "n must be 1..4"))
                    return@post
                }
                val rid = requestId.ifEmpty { UUID.randomUUID().toString() }
                val steps = mobileSteps(quality, null)
                val modelId = "lcm-dreamshaper-int8-onnx"
                if (!downloader.isReady(modelId)) {
                    throw ModelWillNotLoadError("model $modelId not downloaded", -1, -1)
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                if (!LcmOnnxEngine.isReady()) {
                    manager.setLoading(modelId, true)
                    try { LcmOnnxEngine.init(ctx, downloader.modelDir(modelId)) }
                    finally { manager.setLoading(modelId, false) }
                }
                val src = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: run {
                        call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image"))
                        return@post
                    }
                val (w, h) = parseSize(size) ?: (512 to 512)
                val out = LcmOnnxEngine.img2img(
                    src, prompt, steps = steps, size = maxOf(w, h).coerceAtMost(512),
                )
                LcmOnnxEngine.release()
                if (out == null) {
                    throw ModelWillNotLoadError("img2img failed: " + LcmOnnxEngine.lastError, -1, -1)
                }
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(
                    jsonOf(
                        "created" to System.currentTimeMillis() / 1000,
                        "data" to List(n.coerceIn(1, 4)) { jsonOf("b64_json" to b64) },
                        "request_id" to rid,
                    ),
                )
            }
            post("/v1/images/inpaint") {
                val req = wireJson.decodeFromString<InpaintReq>(call.receiveText())
                if (req.image.isEmpty() || req.mask.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image and mask required (base64)"))
                    return@post
                }
                if (maxOf(req.width, req.height) > MAX_CANVAS_PX) {
                // DoS guard like the server: absurd canvas → 503 will-not-load, not a hang.
                    throw ModelWillNotLoadError("canvas $req.width×$req.height exceeds $MAX_CANVAS_PX")
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val modelId = "lcm-dreamshaper-int8-onnx"
                if (!downloader.isReady(modelId)) {
                    throw ModelWillNotLoadError("model $modelId not downloaded", -1, -1)
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                if (!LcmOnnxEngine.isReady()) {
                    manager.setLoading(modelId, true)
                    try { LcmOnnxEngine.init(ctx, downloader.modelDir(modelId)) }
                    finally { manager.setLoading(modelId, false) }
                }
                val src = decodeB64Image(req.image)
                val mask = decodeB64Image(req.mask)
                if (src == null || mask == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image/mask"))
                    return@post
                }
                val steps = (req.steps ?: 6).coerceIn(1, 8)
                val side = maxOf(req.width, req.height).coerceIn(256, 512)
                val colorInt = parseHexColor(req.color)
                val t0 = System.currentTimeMillis()
                val out = LcmOnnxEngine.inpaint(
                    src, mask, req.prompt,
                    negativePrompt = req.negative_prompt,
                    guidance = req.guidance,
                    strength = req.strength.toFloat(),
                    steps = steps,
                    size = side,
                    mode = req.mode,
                    color = colorInt,
                )
                LcmOnnxEngine.release()
                if (out == null) {
                    throw ModelWillNotLoadError("inpaint failed: " + LcmOnnxEngine.lastError, -1, -1)
                }
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(jsonOf("image" to b64, "elapsed" to elapsed, "request_id" to rid, "engine" to "onnx"))
            }
            post("/v1/images/harmonize") {
                val req = wireJson.decodeFromString<HarmonizeReq>(call.receiveText())
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image required (base64)"))
                    return@post
                }
                if (req.max_side > MAX_CANVAS_PX) {
                    throw ModelWillNotLoadError("max_side ${req.max_side} exceeds $MAX_CANVAS_PX")
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val src = decodeB64Image(req.image)
                if (src == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image"))
                    return@post
                }
                val maskB64 = req.mask ?: req.body_mask ?: req.seam_mask
                val mask = maskB64?.let { decodeB64Image(it) }
                val t0 = System.currentTimeMillis()
                val out = ImageOps.harmonize(
                    src, mask,
                    colorMatch = req.color_match,
                    seamFix = req.seam_fix,
                    strength = req.strength.toFloat(),
                )
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(
                    jsonOf(
                        "image" to b64,
                        "elapsed" to elapsed,
                        "request_id" to rid,
                        "engine" to "cpu-harmonize",
                    ),
                )
            }
            post("/v1/images/upscale") {
                val req = wireJson.decodeFromString<UpscaleReq>(call.receiveText())
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image required (base64)"))
                    return@post
                }
                if (req.scale !in 2..4) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "scale must be 2..4"))
                    return@post
                }
                switchTo("upscale", req.model)
                requireModelReady("realesrgan-x4plus-ncnn")
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                // TODO(NCNN): Real-ESRGAN-x4v3 tile processing (Vulkan).
                call.respond(jsonOf("image" to STUB_PNG_B64, "elapsed" to 0.0, "request_id" to rid))
            }
            post("/v1/images/caption") {
                val req = wireJson.decodeFromString<CaptionReq>(call.receiveText())
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image required (base64)"))
                    return@post
                }
// Mobile: SmolVLM runtime is not bundled yet (see ModelCatalog) → 503, not stub text.
                throw ModelWillNotLoadError("vlm smolvlm-500m not available on device (PC :8101 /v1/images/caption works)")
            }
            post("/v1/chat/completions") {
                // V2 mobile: vision chat goes through the PC; error shape like /v1/images/caption.
                throw ModelWillNotLoadError("vlm smolvlm-500m not available on device (PC :8101 /v1/chat/completions works)")
            }
            post("/v1/audio/transcriptions") {
                handleAudio(call, translation = false)
            }
            post("/v1/audio/translations") {
                handleAudio(call, translation = true)
            }
        }
    }

    private suspend fun handleAudio(call: io.ktor.server.application.ApplicationCall, translation: Boolean) {
        var filename = ""
        var bytes = ByteArray(0)
        var model = "whisper-large-v3-turbo-q5_0"
        var language = "sk"
        var autoSplit = true
        call.receiveMultipart().forEachPart { part ->
            when (part) {
                is PartData.FileItem -> if (part.name == "audio" || part.name == "file") {
                    filename = part.originalFileName ?: "audio"
                    bytes = part.provider().readBytes()
                }
                is PartData.FormItem -> when (part.name) {
                    "model" -> model = part.value
                    "language" -> language = part.value
                    "auto_split" -> autoSplit = part.value.lowercase() in setOf("true", "1", "yes", "on")
                }
                else -> {}
            }
            part.dispose()
        }
        if (bytes.isEmpty()) {
// Empty / bad file → 400/415, never 500 (contract).
            call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "empty audio file"))
            return
        }
// No hard size limit: long files are split into chunks (auto-split).
        val ext = filename.substringAfterLast('.', "").lowercase()
        if (ext.isNotEmpty() && ext !in ALLOWED_AUDIO_EXT) {
            call.respond(HttpStatusCode.UnsupportedMediaType, jsonOf("error" to "unsupported extension .$ext"))
            return
        }
// If the requested model is not installed, use the smallest installed
// audio model (fast) — otherwise every default request would hit 503.
        val installedAudio = downloader.allAssets()
            .filter { it.kind == "audio" && downloader.isReady(it.id) }
        val effectiveModel = when {
            downloader.isReady(model) -> model
            installedAudio.isNotEmpty() -> installedAudio.first().id
            else -> model
        }
        if (effectiveModel != model) {
            android.util.Log.i("ApiServer", "audio: $model not installed, using $effectiveModel")
        }

switchTo("audio", effectiveModel) // 503 loading while switching image→audio
        requireModelReady(effectiveModel)

        val tempDir = downloader.modelsDir()
        val pcmFloats = AudioDecoder.decodeToPcmFloats(bytes, tempDir)
        val modelFile = downloader.getModelFile(effectiveModel)
            ?: throw ModelWillNotLoadError("$effectiveModel model file missing", -1, -1)

// Auto-split long recordings (e.g. 45 min): whisper.cpp on mobile
// handles short segments; 30 s chunks are safe for RAM and time.
        val chunkSize = 30 * 16000
        val chunks = if (autoSplit || pcmFloats.size > chunkSize) {
            (0 until pcmFloats.size step chunkSize).map { i ->
                pcmFloats.copyOfRange(i, (i + chunkSize).coerceAtMost(pcmFloats.size))
            }.filter { it.isNotEmpty() }
        } else {
            listOf(pcmFloats)
        }

        val texts = mutableListOf<String>()
        for ((idx, chunk) in chunks.withIndex()) {
            android.util.Log.i("ApiServer", "transcribing chunk ${idx + 1}/${chunks.size} (${chunk.size / 16000}s)")
            val t = WhisperLib.transcribe(
                modelFile = modelFile,
                samples = chunk,
                language = language.ifEmpty { "sk" },
                translate = translation,
            )
            texts.add(t)
        }
        val text = texts.joinToString(" ").trim()

        call.respond(
            jsonOf(
                "text" to text,
                "language" to language,
                "model" to effectiveModel,
                "requested_model" to model,
                "duration" to (pcmFloats.size / 16000.0),
                "auto_split" to (chunks.size > 1),
                "chunks" to chunks.size,
            )
        )
    }

/** Switch the resident model; release the whisper context when leaving audio. */
    private fun switchTo(kind: String, id: String): String {
        val prev = manager.status().loadedKind
        if (prev == "audio" && kind != "audio") WhisperLib.release()
        return manager.ensure(kind, id)
    }

    private fun requireModelReady(assetId: String) {
        if (!downloader.isReady(assetId)) {
            throw ModelWillNotLoadError("model $assetId not downloaded (see ModelDownloader)", -1, -1)
        }
    }


    private fun decodeB64Image(b64: String): android.graphics.Bitmap? {
        return try {
            val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (_: Exception) {
            null
        }
    }

    private fun bitmapToB64(bmp: android.graphics.Bitmap): String {
        val baos = java.io.ByteArrayOutputStream()
        bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, baos)
        return android.util.Base64.encodeToString(baos.toByteArray(), android.util.Base64.NO_WRAP)
    }

    private fun parseHexColor(hex: String): Int {
        return try {
            val h = hex.trim().removePrefix("#")
            when (h.length) {
                6 -> (0xFF000000.toInt()) or h.toLong(16).toInt()
                3 -> {
                    val r = h[0].toString() + h[0]
                    val g = h[1].toString() + h[1]
                    val b = h[2].toString() + h[2]
                    (0xFF000000.toInt()) or (r + g + b).toLong(16).toInt()
                }
                else -> 0xFF0066FF.toInt()
            }
        } catch (_: Exception) {
            0xFF0066FF.toInt()
        }
    }

    companion object {
/** 1×1 PNG stub until the ONNX engine is wired. */
        const val STUB_PNG_B64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

        val ALLOWED_AUDIO_EXT = setOf(
            "wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4", "mov", "avi", "mkv",
        )

        fun parseSize(size: String): Pair<Int, Int>? {
            val m = Regex("""^(\d+)x(\d+)$""").matchEntire(size.trim()) ?: return null
            val w = m.groupValues[1].toInt()
            val h = m.groupValues[2].toInt()
            if (w < 64 || h < 64) return null
            return w to h
        }
    }
}
