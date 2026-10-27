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
import io.ktor.server.response.respondFile
import io.ktor.server.response.respondText
import io.ktor.http.ContentType
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.http.content.PartData
import io.ktor.http.content.forEachPart
import io.ktor.utils.io.core.readBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray as KxJsonArray
import kotlinx.serialization.json.JsonObject as KxJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
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

/** Expand the canvas and let the inpaint engine paint the new border. */
@Serializable
data class OutpaintReq(
    val image: String = "",
    val prompt: String = "",
    val negative_prompt: String = "",
    val model: String = "",
    val left: Int = 0,
    val right: Int = 0,
    val top: Int = 0,
    val bottom: Int = 0,
    val margin: Double = 0.0,
    val guidance: Float = 1.0f,
    val strength: Double = 0.99,
    val steps: Int? = null,
    val request_id: String = "",
)

@Serializable
data class BatchReq(
    val jobs: List<Map<String, String>> = emptyList(),
    val stop_on_error: Boolean = true,
    val request_id: String = "",
)

data class BatchState(
    val id: String,
    val total: Int,
    val stopOnError: Boolean,
) {
    var state: String = "running"
    var done: Int = 0
    var percent: Int = 0
    var message: String = ""
    var error: String = ""
    @Volatile var cancelled: Boolean = false
    val results: java.util.concurrent.CopyOnWriteArrayList<Map<String, Any>> =
        java.util.concurrent.CopyOnWriteArrayList()
}

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
private val batchState = ConcurrentHashMap<String, BatchState>()
    private var engine: NettyApplicationEngine? = null

    // Heavy inference is serialized: two concurrent ORT runs crashed with
    // SIGSEGV inside libonnxruntime on Android. limitedParallelism(1) queues
    // requests without blocking the Ktor/Netty event loop.
    private val heavyDispatcher = Dispatchers.IO.limitedParallelism(1)
    // The LCM engine now stays resident between requests (repeated ORT
    // session init/release churn was crashing the process); it is released
    // after 10 minutes without image work.
    @Volatile private var lastImageUse = 0L
    // Any heavy engine counts here: on a phone that also runs Instagram, Outlook
    // and friends, residency beyond a minute is what got this process killed
    // with reason=LOW_MEMORY (twice) and restarted the launcher with it.
    @Volatile private var lastHeavyUse = 0L

    /**
     * Run a single batch job on the heavy dispatcher. Kept as a member so the
     * private helpers (parseSize, decodeB64Image, switchTo, requireModelReady,
     * mobileSteps, OutputStore) are reachable without qualifying `this`.
     */
    private suspend fun runBatchJob(bid: String, index: Int, job: Map<String, String>,
                                   ctx: android.content.Context) {
        val type = (job["type"] ?: job["kind"] ?: "").lowercase()
        val bs = batchState[bid] ?: return
        try {
            when (type) {
                "generate" -> {
                    val modelId = job.getOrElse("model") { "lcm-dreamshaper-int8-onnx" }
                    if (!downloader.isReady(modelId)) throw ModelWillNotLoadError("model $modelId not downloaded", -1, -1)
                    if (!LcmOnnxEngine.isReady()) {
                        manager.setLoading(modelId, true)
                        try {
                            if (!LcmOnnxEngine.init(ctx, downloader.modelDir(modelId))) {
                                throw ModelWillNotLoadError("LCM init failed: " + LcmOnnxEngine.lastError, -1, -1)
                            }
                        } finally { manager.setLoading(modelId, false) }
                    }
                    val (w, h) = parseSize(job.getOrElse("size") { "512x512" }) ?: Pair(512, 512)
                    val side = maxOf(w, h).coerceIn(256, 512)
                    val steps = mobileSteps(job["quality"], job["steps"]?.toIntOrNull())
                    val bmp = LcmOnnxEngine.generate(
                        job.getOrElse("prompt") { "" },
                        job.getOrElse("negative_prompt") { "" },
                        (job.getOrElse("guidance") { "1.0" }).toFloat(),
                        steps, side,
                        seed = System.nanoTime() + index,
                    ) ?: throw ModelWillNotLoadError("generate failed: " + LcmOnnxEngine.lastError, -1, -1)
                    val baos = java.io.ByteArrayOutputStream()
                    bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, baos)
                    bmp.recycle()
                    val meta = OutputStore.store(ctx, "image", ".png", baos.toByteArray(), bid)
                    bs.results.add(mapOf("index" to index, "type" to "generate", "ok" to true,
                        "size" to "${side}x${side}", "engine" to "onnx-lcm",
                        "outputs" to listOf(meta)))
                }
                "upscale" -> {
                    val src = decodeB64Image(job.getOrElse("image") { "" })
                        ?: throw ModelWillNotLoadError("bad image base64", -1, -1)
                    switchTo("upscale", job.getOrElse("model") { "realesrgan-x4plus-ncnn" })
                    requireModelReady("realesrgan-x4plus-ncnn")
                    val scale = (job.getOrElse("scale") { "2" }).toIntOrNull() ?: 2
                    if (src.width * scale > 1536 || src.height * scale > 1536) {
                        src.recycle()
                        throw ModelWillNotLoadError("upscaled side exceeds the mobile cap 1536", -1, -1)
                    }
                    val out = withContext(heavyDispatcher) {
                        if (!RealEsrganNcnnEngine.isReady()) {
                            manager.setLoading("realesrgan-x4plus-ncnn", true)
                            try {
                                if (!RealEsrganNcnnEngine.init(ctx, downloader.modelDir("realesrgan-x4plus-ncnn"))) {
                                    throw ModelWillNotLoadError("ncnn init failed: " + RealEsrganNcnnEngine.lastError, -1, -1)
                                }
                            } finally { manager.setLoading("realesrgan-x4plus-ncnn", false) }
                        }
                        RealEsrganNcnnEngine.upscale(src, scale)
                    }
                    src.recycle()
                    if (out == null) throw ModelWillNotLoadError("upscale failed: " + RealEsrganNcnnEngine.lastError, -1, -1)
                    val baos = java.io.ByteArrayOutputStream()
                    out.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, baos)
                    val bytes = baos.toByteArray()
                    out.recycle()
                    val meta = OutputStore.store(ctx, "image", ".png", bytes, bid)
                    bs.results.add(mapOf("index" to index, "type" to "upscale", "ok" to true,
                        "size" to "${out.width}x${out.height}", "engine" to "realesrgan-ncnn",
                        "outputs" to listOf(meta)))
                }
                else -> throw IllegalArgumentException("unknown job type: $type")
            }
        } catch (e: Exception) {
            bs.results.add(mapOf("index" to index, "type" to type, "ok" to false,
                "error" to (e.message ?: e.javaClass.simpleName)))
            if (bs.stopOnError) { bs.state = "error"; bs.error = e.message ?: ""; return }
        }
        bs.done = index + 1
        bs.percent = (100 * bs.done) / bs.total
        bs.message = "job ${index + 1}/${bs.total}"
    }

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
        launch {
            // Idle unload keeps the RAM budget honest without re-initialising
            // the ONNX sessions on every request.
            while (isActive) {
                delay(15_000)
                val idleMs = System.currentTimeMillis() - maxOf(lastImageUse, lastHeavyUse)
                val anyReady = LcmOnnxEngine.isReady() || RealEsrganNcnnEngine.isReady() ||
                    SmolVlmEngine.isReady() || WhisperLib.isAvailable()
                if (anyReady && idleMs > 45_000) {
                    withContext(heavyDispatcher) {
                        LcmOnnxEngine.release()
                        RealEsrganNcnnEngine.release()
                        SmolVlmEngine.release()
                        WhisperLib.release()
                    }
                    android.util.Log.i("ApiServer", "idle unload: all engines released")
                }
                // Memory watchdog. The system killed the process with
                // reason=LOW_MEMORY after a run that had LCM + whisper + ncnn
                // resident (pid 28143, 2026-10-27 10:41). Native engines are
                // cheap to rebuild and expensive to lose the whole server for.
                try {
                    val am = appContext?.getSystemService(android.content.Context.ACTIVITY_SERVICE)
                            as? android.app.ActivityManager
                    if (am != null) {
                        val mi = android.app.ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        val availMb = mi.availMem / 1048576
                        if (availMb < 1536) {
                            withContext(heavyDispatcher) {
                                LcmOnnxEngine.release()
                                RealEsrganNcnnEngine.release()
                                SmolVlmEngine.release()
                                WhisperLib.release()
                            }
                            android.util.Log.w("ApiServer",
                                "low memory ($availMb MB free): released all engines")
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.w("ApiServer", "memory watchdog failed: ${e.message}")
                }
            }
        }
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
            get("/v1/models/status/{id}") {
                val id = call.parameters["id"] ?: ""
                val asset = downloader.findAsset(id)
                    ?: return@get call.respond(HttpStatusCode.NotFound,
                        jsonOf("error" to "unknown model id: $id"))
                call.respond(
                    jsonOf(
                        "id" to asset.id,
                        "kind" to asset.kind,
                        "kind_label" to ModelCatalog.kindLabel(asset.kind),
                        "engine" to asset.engine,
                        "label" to asset.label,
                        "installed" to downloader.isReady(id),
                        "installed_mb" to downloader.installedSizeMb(id),
                        "path" to downloader.modelDir(id).absolutePath,
                        "state" to (downloader.progress(id)?.state?.name?.lowercase() ?: "idle"),
                        "error" to (downloader.progress(id)?.error ?: ""),
                    ),
                )
            }
            post("/v1/models/verify/{id}") {
                val id = call.parameters["id"] ?: ""
                val asset = downloader.findAsset(id)
                    ?: return@post call.respond(HttpStatusCode.NotFound,
                        jsonOf("error" to "unknown model id: $id"))
                call.respond(
                    jsonOf(
                        "id" to asset.id,
                        "verified" to downloader.isReady(id),
                        "method" to "READY manifest: every listed file exists and is non-empty",
                        "installed_mb" to downloader.installedSizeMb(id),
                    ),
                )
            }
            post("/v1/models/cancel") {                val body = call.receiveText()
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
            post("/v1/batch") {
                val body = call.receiveText()
                val req = try { wireJson.decodeFromString<BatchReq>(body) } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "body must be JSON"))
                    return@post
                }
                if (req.jobs.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "no jobs"))
                    return@post
                }
                val bid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val bs = BatchState(bid, req.jobs.size, req.stop_on_error)
                batchState[bid] = bs
                launch(heavyDispatcher) {
                    for ((i, job) in req.jobs.withIndex()) {
                        if (bs.cancelled) { bs.state = "cancelled"; break }
                        runBatchJob(bid, i, job, ctx)
                        if (bs.state == "error" && bs.stopOnError) break
                    }
                    if (bs.state == "running") bs.state = "done"
                    bs.percent = 100
                }
                call.respond(jsonOf("id" to bid, "state" to "running", "total" to req.jobs.size,
                    "request_id" to bid))
            }
            get("/v1/batch/{id}") {
                val id = call.parameters["id"] ?: ""
                val bs = batchState[id] ?: return@get call.respond(
                    HttpStatusCode.NotFound, jsonOf("error" to "unknown batch"))
                call.respond(jsonOf(
                    "id" to bs.id, "state" to bs.state, "total" to bs.total, "done" to bs.done,
                    "percent" to bs.percent, "message" to bs.message, "error" to bs.error,
                    "results" to bs.results,
                ))
            }
            post("/v1/batch/{id}/cancel") {
                val id = call.parameters["id"] ?: ""
                val bs = batchState[id] ?: return@post call.respond(
                    HttpStatusCode.NotFound, jsonOf("error" to "unknown batch"))
                bs.cancelled = true
                call.respond(jsonOf("id" to id, "state" to "cancelling", "cancelled" to true))
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
            get("/v1/outputs") {
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError,
                        jsonOf("error" to "no app context"))
                    return@get
                }
                val limit = (call.request.queryParameters["limit"] ?: "200").toIntOrNull() ?: 200
                val files = OutputStore.list(ctx, limit.coerceIn(1, 1000))
                call.respond(
                    jsonOf(
                        "dir" to OutputStore.root(ctx).absolutePath,
                        "count" to files.size,
                        "engine" to "android-outputs",
                        "files" to files.map { f ->
                            val kind = f.parentFile?.name ?: ""
                            jsonOf(
                                "name" to f.name,
                                "kind" to kind,
                                "size" to f.length(),
                                "mtime" to f.lastModified() / 1000,
                                "path" to "$kind/${f.name}",
                                "url" to "/v1/outputs/$kind/${f.name}",
                            )
                        },
                    ),
                )
            }
            get("/v1/outputs/{rel...}") {
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError,
                        jsonOf("error" to "no app context"))
                    return@get
                }
                val rel = call.parameters.getAll("rel")?.joinToString("/") ?: ""
                val rootDir = OutputStore.root(ctx).canonicalFile
                val target = File(rootDir, rel).canonicalFile
                if (!target.path.startsWith(rootDir.path) || !target.isFile) {
                    call.respond(HttpStatusCode.NotFound, jsonOf("error" to "not found"))
                    return@get
                }
                call.respondFile(target)
            }
            delete("/v1/outputs/{rel...}") {
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError,
                        jsonOf("error" to "no app context"))
                    return@delete
                }
                val rel = call.parameters.getAll("rel")?.joinToString("/") ?: ""
                val rootDir = OutputStore.root(ctx).canonicalFile
                val target = File(rootDir, rel).canonicalFile
                if (!target.path.startsWith(rootDir.path) || !target.isFile) {
                    call.respond(HttpStatusCode.NotFound, jsonOf("error" to "not found"))
                    return@delete
                }
                val ok = OutputStore.delete(target)
                call.respond(jsonOf("removed" to rel, "ok" to ok))
            }
            get("/v1/ffmpeg/presets") {
                call.respond(MediaTools.presetsJson())
            }
            post("/v1/ffmpeg/{op}") {
                val op = call.parameters["op"] ?: ""
                var bytes = ByteArray(0)
                var filename = "input"
                var preset = ""
                var startS = ""
                var endS = ""
                var quality = "0"
                call.receiveMultipart().forEachPart { part ->
                    when (part) {
                        is PartData.FileItem -> if (part.name == "file") {
                            filename = part.originalFileName ?: "input"
                            bytes = part.provider().readBytes()
                        }
                        is PartData.FormItem -> when (part.name) {
                            "preset" -> preset = part.value
                            "start" -> startS = part.value
                            "end" -> endS = part.value
                            "quality" -> quality = part.value
                            else -> {}
                        }
                        else -> {}
                    }
                    part.dispose()
                }
                if (bytes.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "Empty file"))
                    return@post
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val suffix = "." + filename.substringAfterLast('.', "bin").lowercase()
                val tmp = File(ctx.cacheDir, "ffmpeg_in_${System.currentTimeMillis()}$suffix")
                val httpRid = UUID.randomUUID().toString()
                try {
                    tmp.writeBytes(bytes)
                    lastHeavyUse = System.currentTimeMillis()
                    val result = withContext(heavyDispatcher) {
                        when (op) {
                            "info" -> null to MediaTools.info(tmp)
                            "convert-image" -> MediaTools.convertImage(
                                tmp, preset.ifEmpty { "jpg-95" }, quality.toIntOrNull() ?: 0) to null
                            "convert-audio" -> MediaTools.convertAudio(
                                tmp, preset.ifEmpty { "aac-192" }) to null
                            "convert-video" -> MediaTools.remux(
                                tmp, preset.ifEmpty { "mp4-h264" }) to null
                            "extract-audio" -> MediaTools.extractAudio(
                                tmp, preset.ifEmpty { "aac-192" }) to null
                            "trim" -> MediaTools.trim(
                                tmp,
                                startS.toDoubleOrNull() ?: 0.0,
                                endS.toDoubleOrNull() ?: 0.0) to null
                            "compress" -> MediaTools.compress(tmp, preset.ifEmpty { "web-720p" }) to null
                            else -> throw MediaTools.Unsupported("unknown operation: $op")
                        }
                    }
                    val (fileOut, infoOut) = result
                    if (fileOut == null) {
                        call.respond(jsonOf("info" to infoOut, "request_id" to httpRid))
                        return@post
                    }
                    val info = mapOf(
                        "duration" to null,
                        "size" to fileOut.bytes.size,
                        "format" to fileOut.mime,
                    )
                    call.respond(
                        jsonOf(
                            "file" to android.util.Base64.encodeToString(fileOut.bytes, android.util.Base64.NO_WRAP),
                            "filename" to fileOut.name,
                            "elapsed" to 0,
                            "request_id" to httpRid,
                            "engine" to "android-mediacodec",
                            "outputs" to outputList(
                                runCatching {
                                    OutputStore.store(ctx, "image", "." + fileOut.name.substringAfterLast('.'),
                                                      fileOut.bytes, httpRid)
                                }.getOrNull()),
                            "info" to info,
                        ),
                    )
                } catch (e: MediaTools.Unsupported) {
                    call.respond(HttpStatusCode.ServiceUnavailable, jsonOf(
                        "error" to "will not load", "detail" to (e.message ?: "unsupported operation")))
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf(
                        "error" to "ffmpeg operation failed: ${e.message}"))
                } finally {
                    try { tmp.delete() } catch (_: Exception) {}
                }
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
                lastImageUse = System.currentTimeMillis()
                lastHeavyUse = lastImageUse
                // Serialize the heavy path: init + UNet steps.
                val generated = withContext(heavyDispatcher) {
                    WhisperLib.release()
                    if (!LcmOnnxEngine.isReady()) {
                        requireMemory(1400, "the LCM image engine")
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
                    val side = lcmSize(maxOf(w, h))
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
                    // Keep the engine resident; the idle unloader releases it.
                    side to images
                }
                val (side, images) = generated
                call.respond(
                    jsonOf(
                        "created" to System.currentTimeMillis() / 1000,
                        "data" to images.map { jsonOf("b64_json" to it) },
                        "request_id" to rid,
                        "steps" to steps,
                        "size" to "${side}x${side}",
                        "requested_size" to "${w}x${h}",
                        "engine" to "onnx-lcm",
                        "outputs" to images.mapNotNull { storeB64("image", ".png", it, rid) },
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
                val src = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: run {
                        call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image"))
                        return@post
                    }
                val (w, h) = parseSize(size) ?: (512 to 512)
                lastImageUse = System.currentTimeMillis()
                lastHeavyUse = lastImageUse
                val out = withContext(heavyDispatcher) {
                    if (!LcmOnnxEngine.isReady()) {
                        requireMemory(1400, "the LCM image engine")
                        manager.setLoading(modelId, true)
                        try { LcmOnnxEngine.init(ctx, downloader.modelDir(modelId)) }
                        finally { manager.setLoading(modelId, false) }
                    }
                    LcmOnnxEngine.img2img(
                        src, prompt, steps = steps, size = maxOf(w, h).coerceAtMost(512),
                    )
                }
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
                        "engine" to "onnx-lcm",
                        "outputs" to outputList(storeB64("image", ".png", b64, rid)),
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
                val src = decodeB64Image(req.image)
                val mask = decodeB64Image(req.mask)
                if (src == null || mask == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image/mask"))
                    return@post
                }
                val steps = (req.steps ?: 6).coerceIn(1, 8)
                val side = lcmSize(maxOf(req.width, req.height))
                val colorInt = parseHexColor(req.color)
                val t0 = System.currentTimeMillis()
                lastImageUse = t0
                lastHeavyUse = t0
                val out = withContext(heavyDispatcher) {
                    if (!LcmOnnxEngine.isReady()) {
                        requireMemory(1400, "the LCM image engine")
                        manager.setLoading(modelId, true)
                        try { LcmOnnxEngine.init(ctx, downloader.modelDir(modelId)) }
                        finally { manager.setLoading(modelId, false) }
                    }
                    LcmOnnxEngine.inpaint(
                        src, mask, req.prompt,
                        negativePrompt = req.negative_prompt,
                        guidance = req.guidance,
                        strength = req.strength.toFloat(),
                        steps = steps,
                        size = side,
                        mode = req.mode,
                        color = colorInt,
                    )
                }
                if (out == null) {
                    throw ModelWillNotLoadError("inpaint failed: " + LcmOnnxEngine.lastError, -1, -1)
                }
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(jsonOf("image" to b64, "elapsed" to elapsed, "request_id" to rid,
                    "engine" to "onnx",
                    "outputs" to outputList(storeB64("image", ".png", b64, rid))))
            }
            post("/v1/images/outpaint") {
                val req = wireJson.decodeFromString<OutpaintReq>(call.receiveText())
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image required (base64)"))
                    return@post
                }
                val src = decodeB64Image(req.image)
                if (src == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "cannot decode image"))
                    return@post
                }
                val short = minOf(src.width, src.height)
                var l = req.left.coerceAtLeast(0)
                var r = req.right.coerceAtLeast(0)
                var t = req.top.coerceAtLeast(0)
                var b = req.bottom.coerceAtLeast(0)
                if (req.margin > 0.0) {
                    val m = (short * req.margin).toInt()
                    if (l == 0) l = m
                    if (r == 0) r = m
                    if (t == 0) t = m
                    if (b == 0) b = m
                }
                if (l + r + t + b == 0) {
                    call.respond(HttpStatusCode.BadRequest,
                        jsonOf("error" to "nothing to expand: pass left/right/top/bottom pixels or margin"))
                    return@post
                }
                val newW = src.width + l + r
                val newH = src.height + t + b
                if (maxOf(newW, newH) > 2048 || newW * newH > 4_000_000) {
                    call.respond(HttpStatusCode.BadRequest,
                        jsonOf("error" to "result too large: ${newW}x${newH} (max side 2048, 4 MP)"))
                    return@post
                }
                val modelId = "lcm-dreamshaper-int8-onnx"
                if (!downloader.isReady(modelId)) {
                    throw ModelWillNotLoadError("model $modelId not downloaded", -1, -1)
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val steps = (req.steps ?: 6).coerceIn(1, 8)
                val side = lcmSize(maxOf(newW, newH))
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val inW = src.width
                val inH = src.height
                val t0 = System.currentTimeMillis()
                lastImageUse = t0
                lastHeavyUse = t0
                val out = withContext(heavyDispatcher) {
                    if (!LcmOnnxEngine.isReady()) {
                        requireMemory(1400, "the LCM image engine")
                        manager.setLoading(modelId, true)
                        try { LcmOnnxEngine.init(ctx, downloader.modelDir(modelId)) }
                        finally { manager.setLoading(modelId, false) }
                    }
                    val padded = padEdge(src, l, r, t, b)
                    val mask = borderMask(newW, newH, l, r, t, b)
                    try {
                        LcmOnnxEngine.inpaint(
                            padded, mask, req.prompt,
                            negativePrompt = req.negative_prompt,
                            guidance = req.guidance,
                            strength = req.strength.toFloat(),
                            steps = steps,
                            size = side,
                            mode = "add",
                        )
                    } finally {
                        padded.recycle()
                        mask.recycle()
                        src.recycle()
                    }
                }
                if (out == null) {
                    throw ModelWillNotLoadError("outpaint failed: " + LcmOnnxEngine.lastError, -1, -1)
                }
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(
                    jsonOf(
                        "image" to b64,
                        "elapsed" to (System.currentTimeMillis() - t0) / 1000.0,
                        "request_id" to rid,
                        "engine" to "onnx-outpaint",
                        "outputs" to outputList(storeB64("image", ".png", b64, rid)),
                        "input_size" to listOf(inW, inH),
                        "output_size" to listOf(newW, newH),
                        "margins" to mapOf("left" to l, "right" to r, "top" to t, "bottom" to b),
                    ),
                )
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
                lastHeavyUse = System.currentTimeMillis()
                this@ApiServer.requireModelReady("realesrgan-x4plus-ncnn")
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val src = decodeB64Image(req.image)
                if (src == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "invalid image base64"))
                    return@post
                }
                // 512 -> 2048 on CPU kept a phone busy for >6 minutes and the
                // system started thrashing swap. 384 -> 1536 is the mobile cap.
                val outSide = maxOf(src.width, src.height) * req.scale
                if (outSide > 1536) {
                    src.recycle()
                    call.respond(HttpStatusCode.BadRequest, jsonOf(
                        "error" to "upscaled side $outSide exceeds the mobile cap 1536 " +
                            "(use scale 2 for large photos)"))
                    return@post
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val t0 = System.currentTimeMillis()
                val out = withContext(heavyDispatcher) {
                    if (!RealEsrganNcnnEngine.isReady()) {
                        requireMemory(400, "the Real-ESRGAN engine")
                        manager.setLoading("realesrgan-x4plus-ncnn", true)
                        try {
                            if (!RealEsrganNcnnEngine.init(ctx, downloader.modelDir("realesrgan-x4plus-ncnn"))) {
                                throw ModelWillNotLoadError("Real-ESRGAN NCNN init failed: " + RealEsrganNcnnEngine.lastError, -1, -1)
                            }
                        } finally {
                            manager.setLoading("realesrgan-x4plus-ncnn", false)
                        }
                    }
                    RealEsrganNcnnEngine.upscale(src, req.scale)
                }
                src.recycle()
                if (out == null) {
                    throw ModelWillNotLoadError("Real-ESRGAN NCNN upscale failed: " + RealEsrganNcnnEngine.lastError, -1, -1)
                }
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                val b64 = bitmapToB64(out)
                out.recycle()
                call.respond(jsonOf(
                    "image" to b64,
                    "elapsed" to elapsed,
                    "request_id" to rid,
                    "engine" to "realesrgan-ncnn",
                    "outputs" to outputList(storeB64("image", ".png", b64, rid)),
                ))
            }
            post("/v1/images/caption") {
                val req = wireJson.decodeFromString<CaptionReq>(call.receiveText())
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "image required (base64)"))
                    return@post
                }
                switchTo("vlm", req.model)
                lastHeavyUse = System.currentTimeMillis()
                if (!SmolVlmEngine.nativeAvailable) {
                    call.respond(HttpStatusCode.ServiceUnavailable, jsonOf(
                        "error" to "will not load",
                        "detail" to "VLM runtime (libsmolvlm_jni.so / llama.cpp mtmd) is not bundled in this build"))
                    return@post
                }
                requireModelReady("smolvlm-500m-gguf")
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val src = decodeB64Image(req.image)
                if (src == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "invalid image base64"))
                    return@post
                }
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val t0 = System.currentTimeMillis()
                val caption = withContext(heavyDispatcher) {
                    if (!SmolVlmEngine.isReady()) {
                        requireMemory(900, "the SmolVLM engine")
                        manager.setLoading("smolvlm-500m-gguf", true)
                        try {
                            if (!SmolVlmEngine.init(ctx, downloader.modelDir("smolvlm-500m-gguf"))) {
                                throw ModelWillNotLoadError("SmolVLM llama.cpp init failed: " + SmolVlmEngine.lastError, -1, -1)
                            }
                        } finally {
                            manager.setLoading("smolvlm-500m-gguf", false)
                        }
                    }
                    SmolVlmEngine.caption(src, req.prompt, req.max_tokens ?: 256)
                }
                src.recycle()
                if (caption == null) {
                    throw ModelWillNotLoadError("SmolVLM llama.cpp caption failed: " + SmolVlmEngine.lastError, -1, -1)
                }
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                call.respond(jsonOf(
                    "caption" to caption,
                    "outputs" to outputList(storeText(caption, rid)),
                    "elapsed" to elapsed,
                    "request_id" to rid,
                    "engine" to "smolvlm-llama",
                ))
            }
            post("/v1/chat/completions") {
                // Vision chat on device: the same SmolVLM engine that captions,
                // fed with the conversation text + the first image in the thread.
                val raw = call.receiveText()
                val root: KxJsonObject = try {
                    Json.parseToJsonElement(raw).jsonObject
                } catch (e: Exception) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "body must be JSON"))
                    return@post
                }
                val model = root["model"]?.jsonPrimitive?.contentOrNull ?: ""
                val maxTokens = root["max_tokens"]?.jsonPrimitive?.intOrNull ?: 256
                val messages: KxJsonArray = root["messages"]?.jsonArray ?: KxJsonArray(emptyList())

                var prompt = ""
                var imageB64 = ""
                for (m in messages) {
                    val msg = m.jsonObject
                    val role = msg["role"]?.jsonPrimitive?.contentOrNull ?: "user"
                    val content = msg["content"] ?: continue
                    if (content is kotlinx.serialization.json.JsonPrimitive) {
                        if (role != "assistant") prompt = content.contentOrNull ?: prompt
                    } else if (content is KxJsonArray) {
                        for (part in content) {
                            val obj = part as? KxJsonObject ?: continue
                            val type = obj["type"]?.jsonPrimitive?.contentOrNull ?: ""
                            if (type == "text") {
                                prompt = obj["text"]?.jsonPrimitive?.contentOrNull ?: prompt
                            } else if (type == "image_url") {
                                val iu = obj["image_url"] ?: continue
                                val url = (iu as? KxJsonObject)?.get("url")?.jsonPrimitive?.contentOrNull
                                    ?: iu.jsonPrimitive.contentOrNull
                                    ?: ""
                                if (url.startsWith("data:") && imageB64.isEmpty()) {
                                    imageB64 = url.substringAfter(",", "")
                                }
                            }
                        }
                    }
                }
                if (imageB64.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf(
                        "error" to "missing image: on-device chat needs an image_url data: URL (SmolVLM is a VLM)"))
                    return@post
                }
                val src = decodeB64Image(imageB64)
                if (src == null) {
                    call.respond(HttpStatusCode.BadRequest, jsonOf("error" to "invalid image base64"))
                    return@post
                }
                switchTo("vlm", model)
                lastHeavyUse = System.currentTimeMillis()
                if (!SmolVlmEngine.nativeAvailable) {
                    call.respond(HttpStatusCode.ServiceUnavailable, jsonOf(
                        "error" to "will not load",
                        "detail" to "VLM runtime (libsmolvlm_jni.so / llama.cpp mtmd) is not bundled in this build"))
                    return@post
                }
                requireModelReady("smolvlm-500m-gguf")
                val ctx = appContext ?: run {
                    call.respond(HttpStatusCode.InternalServerError, jsonOf("error" to "no app context"))
                    return@post
                }
                val t0 = System.currentTimeMillis()
                val answer = withContext(heavyDispatcher) {
                    if (!SmolVlmEngine.isReady()) {
                        requireMemory(900, "the SmolVLM engine")
                        manager.setLoading("smolvlm-500m-gguf", true)
                        try {
                            if (!SmolVlmEngine.init(ctx, downloader.modelDir("smolvlm-500m-gguf"))) {
                                throw ModelWillNotLoadError(
                                    "SmolVLM llama.cpp init failed: " + SmolVlmEngine.lastError, -1, -1)
                            }
                        } finally {
                            manager.setLoading("smolvlm-500m-gguf", false)
                        }
                    }
                    SmolVlmEngine.caption(src, prompt.ifEmpty { "Describe the image in detail." }, maxTokens)
                }
                src.recycle()
                if (answer == null) {
                    throw ModelWillNotLoadError("vlm chat failed: " + SmolVlmEngine.lastError, -1, -1)
                }
                call.respond(
                    jsonOf(
                        "id" to "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").take(12),
                        "object" to "chat.completion",
                        "created" to System.currentTimeMillis() / 1000,
                        "model" to "smolvlm-500m-gguf",
                        "choices" to listOf(
                            mapOf(
                                "index" to 0,
                                "message" to mapOf("role" to "assistant", "content" to answer),
                                "finish_reason" to "stop",
                            )
                        ),
                        "engine" to "smolvlm-gguf",
                        "outputs" to outputList(storeText(answer, "")),
                        "elapsed" to (System.currentTimeMillis() - t0) / 1000.0,
                    ),
                )
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
        var responseFormat = "json"
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
                    "response_format" -> responseFormat = part.value.lowercase()
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

        val wantSubs = responseFormat == "srt" || responseFormat == "vtt" ||
            responseFormat == "verbose_json"
        val texts = mutableListOf<String>()
        val cues = mutableListOf<Triple<Double, Double, String>>()
        var chunkOffset = 0.0
        for ((idx, chunk) in chunks.withIndex()) {
            android.util.Log.i("ApiServer", "transcribing chunk ${idx + 1}/${chunks.size} (${chunk.size / 16000}s)")
            if (wantSubs) {
                val raw = WhisperLib.transcribeSegments(
                    modelFile = modelFile,
                    samples = chunk,
                    language = language.ifEmpty { "sk" },
                    translate = translation,
                    offsetSeconds = chunkOffset,
                )
                for (line in raw.lineSequence()) {
                    val parts = line.split('\t')
                    if (parts.size >= 3) {
                        val a = parts[0].toDoubleOrNull() ?: continue
                        val b = parts[1].toDoubleOrNull() ?: continue
                        val t = parts.drop(2).joinToString("\t").trim()
                        if (t.isNotEmpty()) cues.add(Triple(a, b, t))
                    }
                }
                texts.add(cues.filter { it.first >= chunkOffset }
                    .joinToString(" ") { it.third })
            } else {
                val t = WhisperLib.transcribe(
                    modelFile = modelFile,
                    samples = chunk,
                    language = language.ifEmpty { "sk" },
                    translate = translation,
                )
                texts.add(t)
            }
            chunkOffset += chunk.size / 16000.0
        }
        val text = texts.joinToString(" ").trim().replace(Regex(" +"), " ")

        if (responseFormat == "srt" || responseFormat == "vtt") {
            val body = if (responseFormat == "srt") buildSrt(cues) else buildVtt(cues)
            val ctx = appContext
            val meta = if (ctx != null) OutputStore.store(
                ctx, "text", ".$responseFormat", body.toByteArray(), "") else null
            call.respondText(
                body,
                contentType = ContentType.parse("text/plain; charset=utf-8"),
                status = HttpStatusCode.OK,
            )
            if (meta != null) android.util.Log.i("ApiServer", "subtitles -> ${meta["path"]}")
            return
        }

        call.respond(
            jsonOf(
                "text" to text,
                "outputs" to outputList(storeText(text, "")),
                "language" to language,
                "model" to effectiveModel,
                "requested_model" to model,
                "duration" to (pcmFloats.size / 16000.0),
                "auto_split" to (chunks.size > 1),
                "chunks" to chunks.size,
                "segments" to if (responseFormat == "verbose_json") cues.map {
                    mapOf("start" to it.first, "end" to it.second, "text" to it.third)
                } else emptyList<Map<String, Any>>(),
            )
        )
    }

/** Switch the resident model; release engines when leaving their kind. */
    private fun switchTo(kind: String, id: String): String {
        val prev = manager.status().loadedKind
        if (prev == "audio" && kind != "audio") WhisperLib.release()
        if (prev == "upscale" && kind != "upscale") RealEsrganNcnnEngine.release()
        if (prev == "vlm" && kind != "vlm") SmolVlmEngine.release()
        // Image stays resident for the common generate -> inpaint flow and is
        // freed as soon as another heavy engine needs the RAM. Audio was missing
        // here: LCM (~2-3 GB) plus whisper made the system kill the process with
        // LOW_MEMORY during a transcription run (pid 21358, 2026-10-27 10:33).
        if (prev == "image" && kind != "image") {
            LcmOnnxEngine.release()
        }
        if (kind == "audio" && RealEsrganNcnnEngine.isReady()) {
            RealEsrganNcnnEngine.release()
        }
        return manager.ensure(kind, id)
    }


    /**
     * LCM UNet works on size/8 latents and halves them three times, so the
     * pixel size must be a multiple of 64. 288 used to slip through and ORT
     * failed inside /up_blocks.1/Concat.
     */
    private fun lcmSize(requested: Int): Int {
        val clamped = requested.coerceIn(256, 512)
        val rounded = (clamped + 32) / 64 * 64
        return rounded.coerceIn(256, 512)
    }

    /** Free RAM in MB as Android reports it (available, not free). */
    private fun availableMb(): Long {
        return try {
            val am = appContext?.getSystemService(android.content.Context.ACTIVITY_SERVICE)
                    as? android.app.ActivityManager ?: return Long.MAX_VALUE
            val mi = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            mi.availMem / 1048576
        } catch (e: Exception) {
            Long.MAX_VALUE
        }
    }

    /**
     * Refuse to load a native engine when the system is already starved.
     * LMKD killed this process twice with reason=LOW_MEMORY while the suite had
     * two engines resident plus other user apps; an honest 503 keeps the server
     * (and the phone) alive.
     */
    private fun requireMemory(neededMb: Long, what: String) {
        val avail = availableMb()
        if (avail < neededMb) {
            throw ModelWillNotLoadError(
                "low memory: ${avail} MB available, $what needs ~${neededMb} MB. " +
                    "Close other apps or use a smaller model.", -1, -1)
        }
    }

    private fun tsSrt(t: Double): String {
        val ms = Math.round(t * 1000)
        val h = ms / 3_600_000
        val m = (ms % 3_600_000) / 60_000
        val s = (ms % 60_000) / 1000
        val r = ms % 1000
        return String.format("%02d:%02d:%02d,%03d", h, m, s, r)
    }

    private fun tsVtt(t: Double): String = tsSrt(t).replace(',', '.')

    private fun buildSrt(cues: List<Triple<Double, Double, String>>): String {
        if (cues.isEmpty()) return ""
        val sb = StringBuilder()
        cues.forEachIndexed { i, c ->
            sb.append(i + 1).append('\n')
            sb.append(tsSrt(c.first)).append(" --> ").append(tsSrt(c.second)).append('\n')
            sb.append(c.third).append("\n\n")
        }
        return sb.toString()
    }

    private fun buildVtt(cues: List<Triple<Double, Double, String>>): String {
        if (cues.isEmpty()) return "WEBVTT\n"
        val sb = StringBuilder("WEBVTT\n\n")
        cues.forEach { c ->
            sb.append(tsVtt(c.first)).append(" --> ").append(tsVtt(c.second)).append('\n')
            sb.append(c.third).append("\n\n")
        }
        return sb.toString()
    }

    private fun requireModelReady(assetId: String) {
        if (!downloader.isReady(assetId)) {
            throw ModelWillNotLoadError("model $assetId not downloaded (see ModelDownloader)", -1, -1)
        }
    }


    // ---------- output store (shared with the UI) ----------
    private fun storeB64(kind: String, ext: String, b64: String, rid: String): Map<String, Any>? =
        runCatching {
            val ctx = appContext ?: return null
            val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
            OutputStore.store(ctx, kind, ext, bytes, rid)
        }.getOrNull()

    private fun storeText(text: String, rid: String, ext: String = ".txt"): Map<String, Any>? =
        runCatching {
            val ctx = appContext ?: return null
            OutputStore.store(ctx, "text", ext, text.toByteArray(), rid)
        }.getOrNull()

    private fun outputList(v: Map<String, Any>?): List<Map<String, Any>> =
        if (v == null) emptyList() else listOf(v)

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

    private fun encodeB64(bmp: android.graphics.Bitmap): String {
        val baos = java.io.ByteArrayOutputStream()
        bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, baos)
        return android.util.Base64.encodeToString(baos.toByteArray(), android.util.Base64.NO_WRAP)
    }

    /** Replicate the edge pixels outward so the inpaint model has real context. */
    private fun padEdge(src: android.graphics.Bitmap, l: Int, r: Int, t: Int, b: Int): android.graphics.Bitmap {
        val w = src.width
        val h = src.height
        val out = android.graphics.Bitmap.createBitmap(w + l + r, h + t + b, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(out)
        val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        val full = android.graphics.Rect(0, 0, w, h)
        canvas.drawBitmap(src, full, android.graphics.Rect(l, t, l + w, t + h), paint)
        if (l > 0) {
            canvas.drawBitmap(src, android.graphics.Rect(0, 0, 1, h), android.graphics.Rect(0, t, l, t + h), paint)
            canvas.drawBitmap(src, android.graphics.Rect(0, 0, 1, 1), android.graphics.Rect(0, 0, l, t), paint)
            canvas.drawBitmap(src, android.graphics.Rect(0, h - 1, 1, h), android.graphics.Rect(0, t + h, l, t + h + b), paint)
        }
        if (r > 0) {
            canvas.drawBitmap(src, android.graphics.Rect(w - 1, 0, w, h), android.graphics.Rect(l + w, t, l + w + r, t + h), paint)
            canvas.drawBitmap(src, android.graphics.Rect(w - 1, 0, w, 1), android.graphics.Rect(l + w, 0, l + w + r, t), paint)
            canvas.drawBitmap(src, android.graphics.Rect(w - 1, h - 1, w, h), android.graphics.Rect(l + w, t + h, l + w + r, t + h + b), paint)
        }
        if (t > 0) {
            canvas.drawBitmap(src, android.graphics.Rect(0, 0, w, 1), android.graphics.Rect(l, 0, l + w, t), paint)
        }
        if (b > 0) {
            canvas.drawBitmap(src, android.graphics.Rect(0, h - 1, w, h), android.graphics.Rect(l, t + h, l + w, t + h + b), paint)
        }
        return out
    }

    /** White border (repaint area) on a black canvas. */
    private fun borderMask(w: Int, h: Int, l: Int, r: Int, t: Int, b: Int): android.graphics.Bitmap {
        val mask = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(mask)
        canvas.drawColor(android.graphics.Color.BLACK)
        val paint = android.graphics.Paint().apply { color = android.graphics.Color.WHITE }
        if (l > 0) canvas.drawRect(0f, 0f, l.toFloat(), h.toFloat(), paint)
        if (r > 0) canvas.drawRect((w - r).toFloat(), 0f, w.toFloat(), h.toFloat(), paint)
        if (t > 0) canvas.drawRect(0f, 0f, w.toFloat(), t.toFloat(), paint)
        if (b > 0) canvas.drawRect(0f, (h - b).toFloat(), w.toFloat(), h.toFloat(), paint)
        return mask
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
