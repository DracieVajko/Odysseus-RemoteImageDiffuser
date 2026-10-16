package eu.odysseus.diffusion

import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.netty.NettyApplicationEngine
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.request.receive
import io.ktor.server.request.receiveMultipart
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.http.content.PartData
import io.ktor.http.content.forEachPart
import io.ktor.utils.io.core.readBytes
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

// ---------------------------------------------------------------------------
// Zmluva: rovnaké cesty + rovnaké 503 tvary ako docs/ODYSSEUS-CONTRACT.md.
// Base: http://<phone>:8101
//
// Rozdiely (mobile V1, research verdikty):
// - quality→steps sa capuje na 4..8 (LCM režim); viac = OOM na Adreno/Mali.
// - backend NCNN + Vulkan (nie NNAPI).
// Stav inferencie (mobile V1):
// - audio: REÁLNE (AudioDecoder → WhisperLib JNI → libwhisper.so z CMake buildu).
// - image/upscale: STUB (1×1 PNG) + 503 will-not-load bez váh; NCNN hooky TODO(NCNN).
//   Dôvod: overený NCNN zdroj pre LCM neexistuje (pozri ModelCatalog).
// ---------------------------------------------------------------------------

const val SERVER_PORT = 8101
const val MOBILE_STEPS_MIN = 4
const val MOBILE_STEPS_MAX = 8
const val MAX_CANVAS_PX = 8192
const val MAX_AUDIO_BYTES = 100L * 1024L * 1024L // 100 MB, ako WhisperApi

/** quality → steps, potom clamp do [MOBILE_STEPS_MIN, MOBILE_STEPS_MAX]. */
fun mobileSteps(quality: String?, steps: Int?): Int {
    val base = when (steps) {
        null -> when (quality ?: "medium") {
            "low" -> 4
            "high" -> 30
            "auto" -> 12
            else -> 25 // medium = profile default (zrkadlo servera)
        }
        else -> steps
    }
    return base.coerceIn(MOBILE_STEPS_MIN, MOBILE_STEPS_MAX)
}

@Serializable
data class GenerationReq(
    val model: String = "",
    val prompt: String = "",
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

enum class JobState { RUNNING, LOADING, DONE, ERROR, UNKNOWN }

data class JobProgress(
    val id: String,
    var status: JobState = JobState.RUNNING,
    var step: Int = 0,
    var total: Int = MOBILE_STEPS_MAX,
    var percent: Int = 0,
)

class ApiServer(
    private val manager: ModelManager,
    private val downloader: ModelDownloader,
) {
    private val progress = ConcurrentHashMap<String, JobProgress>()
    private var engine: NettyApplicationEngine? = null

    fun start(wait: Boolean = false): NettyApplicationEngine {
        val eng = embeddedServer(Netty, port = SERVER_PORT) { module() }
        engine = eng
        eng.start(wait = wait)
        return eng
    }

    fun stop() {
        engine?.stop(1000, 2000)
        engine = null
    }

    private fun Application.module() {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true; explicitNulls = false })
        }
        install(StatusPages) {
            exception<ModelLoadingError> { call, e ->
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    mapOf(
                        "error" to "loading model",
                        "loading_model" to e.loadingModel,
                        "retry_after" to e.retryAfterSec,
                    ),
                )
            }
            exception<ModelWillNotLoadError> { call, e ->
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    mapOf(
                        "error" to "will not load",
                        "reason" to e.reason,
                        "needed_mb" to e.neededMb,
                        "free_mb" to e.freeMb,
                    ),
                )
            }
        }
        routing {
            get("/health") {
                val st = manager.status()
                call.respond(
                    mapOf(
                        "status" to "ok",
                        "model" to (st.loadedId ?: ""),
                        "current_model" to (st.loadedId ?: ""),
                        "device" to "vulkan-ncnn",
                        "available_models" to listOf("lcm-dreamshaper-int8", "whisper-small-q5_1"),
                        "loading" to st.loading,
                        "loading_model" to st.loadingModel,
                        "busy" to st.busy,
                        "queue" to st.queue,
                    ),
                )
            }
            get("/v1/models") {
                val st = manager.status()
                call.respond(
                    mapOf(
                        "data" to listOf(
                            mapOf(
                                "id" to "lcm-dreamshaper-int8",
                                "object" to "model",
                                "owned_by" to "odysseus-local",
                                "available" to downloader.isReady("lcm-dreamshaper-int8"),
                                "guessed" to false,
                            ),
                            mapOf(
                                "id" to "whisper-small-q5_1",
                                "object" to "model",
                                "owned_by" to "odysseus-local",
                                "available" to downloader.isReady("whisper-small-q5_1"),
                                "guessed" to false,
                            ),
                        ),
                        "currently_loaded_model" to (st.loadedId ?: ""),
                        "loading" to st.loading,
                        "loading_model" to st.loadingModel,
                        "missing_models" to ModelCatalog.assets.filter { !downloader.isReady(it.id) }.map { it.id },
                    ),
                )
            }
            get("/v1/images/progress/{id}") {
                val id = call.parameters["id"] ?: ""
                val j = progress[id]
                if (j == null) {
                    call.respond(mapOf("id" to id, "status" to "unknown", "step" to 0, "total" to 0, "percent" to 0))
                } else {
                    call.respond(
                        mapOf(
                            "id" to j.id,
                            "status" to j.status.name.lowercase(),
                            "step" to j.step,
                            "total" to j.total,
                            "percent" to j.percent,
                        ),
                    )
                }
            }
            post("/v1/images/generations") {
                val req = call.receive<GenerationReq>()
                if (req.n !in 1..4) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "n must be 1..4"))
                    return@post
                }
                val (w, h) = parseSize(req.size) ?: run {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "bad size, expected WxH"))
                    return@post
                }
                if (maxOf(w, h) > MAX_CANVAS_PX) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "canvas too large"))
                    return@post
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                val steps = mobileSteps(req.quality, req.steps)
                val modelId = req.model.ifEmpty { "lcm-dreamshaper-int8" }
                manager.ensure("image", modelId) // 503 loading → StatusPages
                requireModelReady("lcm-dreamshaper-int8") // 503 will-not-load bez váh
                val job = JobProgress(rid, JobState.RUNNING, 0, steps, 0)
                progress[rid] = job
                // TODO(NCNN): Vulkan LCM-Dreamshaper INT8 inferencia, n=req.n, w×h, steps.
                job.status = JobState.DONE
                job.step = steps
                job.percent = 100
                call.respond(
                    mapOf(
                        "created" to System.currentTimeMillis() / 1000,
                        "data" to List(req.n) { mapOf("b64_json" to STUB_PNG_B64) },
                        "request_id" to rid,
                    ),
                )
            }
            post("/v1/images/edits") {
                // Multipart: image (binárne) + prompt, model, n, size, quality, strength, request_id.
                var prompt = ""
                var model = ""
                var n = 1
                var size = "512x512"
                var quality = "medium"
                var requestId = ""
                var hasImage = false
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
                            part.provider().readBytes() // TODO(NCNN): dekódovať do pipeline
                        }
                        else -> {}
                    }
                    part.dispose()
                }
                if (!hasImage) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "missing image part"))
                    return@post
                }
                if (n !in 1..4) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "n must be 1..4"))
                    return@post
                }
                val rid = requestId.ifEmpty { UUID.randomUUID().toString() }
                val steps = mobileSteps(quality, null)
                manager.ensure("image", model.ifEmpty { "lcm-dreamshaper-int8" })
                requireModelReady("lcm-dreamshaper-int8")
                val job = JobProgress(rid, JobState.DONE, steps, steps, 100)
                progress[rid] = job
                @Suppress("UNUSED_VARIABLE")
                val unused = prompt + size
                // TODO(NCNN): img2img strength 0.1..1 clamp, Vulkan path.
                call.respond(
                    mapOf(
                        "created" to System.currentTimeMillis() / 1000,
                        "data" to List(n) { mapOf("b64_json" to STUB_PNG_B64) },
                        "request_id" to rid,
                    ),
                )
            }
            post("/v1/images/inpaint") {
                val req = call.receive<InpaintReq>()
                if (req.image.isEmpty() || req.mask.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "image and mask required (base64)"))
                    return@post
                }
                if (maxOf(req.width, req.height) > MAX_CANVAS_PX) {
                    // DoS guard ako server: absurd canvas → 503 will-not-load, nie hang.
                    throw ModelWillNotLoadError("canvas $req.width×$req.height exceeds $MAX_CANVAS_PX")
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                manager.ensure("image", req.model.ifEmpty { "lcm-dreamshaper-int8" })
                requireModelReady("lcm-dreamshaper-int8")
                val steps = (req.steps ?: 8).coerceIn(MOBILE_STEPS_MIN, MOBILE_STEPS_MAX)
                val job = JobProgress(rid, JobState.DONE, steps, steps, 100)
                progress[rid] = job
                // TODO(NCNN): mask blur feather + inpaint pipeline.
                call.respond(mapOf("image" to STUB_PNG_B64, "elapsed" to 0.0, "request_id" to rid))
            }
            post("/v1/images/harmonize") {
                val req = call.receive<HarmonizeReq>()
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "image required (base64)"))
                    return@post
                }
                if (req.max_side > MAX_CANVAS_PX) {
                    throw ModelWillNotLoadError("max_side ${req.max_side} exceeds $MAX_CANVAS_PX")
                }
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                manager.ensure("image", req.model.ifEmpty { "lcm-dreamshaper-int8" })
                requireModelReady("lcm-dreamshaper-int8")
                val job = JobProgress(rid, JobState.DONE, 1, 1, 100)
                progress[rid] = job
                // TODO: color-match + seam-fix (CPU, OpenCV alebo NCNN).
                call.respond(mapOf("image" to STUB_PNG_B64, "elapsed" to 0.0, "request_id" to rid))
            }
            post("/v1/images/upscale") {
                val req = call.receive<UpscaleReq>()
                if (req.image.isEmpty()) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "image required (base64)"))
                    return@post
                }
                if (req.scale !in 2..4) {
                    call.respond(HttpStatusCode.BadRequest, mapOf("error" to "scale must be 2..4"))
                    return@post
                }
                manager.ensure("upscale", req.model)
                requireModelReady("realesrgan-x4plus-ncnn")
                val rid = req.request_id.ifEmpty { UUID.randomUUID().toString() }
                // TODO(NCNN): Real-ESRGAN-x4v3 tile processing (Vulkan).
                call.respond(mapOf("image" to STUB_PNG_B64, "elapsed" to 0.0, "request_id" to rid))
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
        var model = "whisper-small-q5_1"
        var language = "sk"
        call.receiveMultipart().forEachPart { part ->
            when (part) {
                is PartData.FileItem -> if (part.name == "audio" || part.name == "file") {
                    filename = part.originalFileName ?: "audio"
                    bytes = part.provider().readBytes()
                }
                is PartData.FormItem -> when (part.name) {
                    "model" -> model = part.value
                    "language" -> language = part.value
                }
                else -> {}
            }
            part.dispose()
        }
        if (bytes.isEmpty()) {
            // Prázdny / zlý súbor → 400/415, nikdy 500 (kontrakt).
            call.respond(HttpStatusCode.BadRequest, mapOf("error" to "empty audio file"))
            return
        }
        if (bytes.size > MAX_AUDIO_BYTES) {
            call.respond(HttpStatusCode.BadRequest, mapOf("error" to "file too large (max 100 MB)"))
            return
        }
        val ext = filename.substringAfterLast('.', "").lowercase()
        if (ext.isNotEmpty() && ext !in ALLOWED_AUDIO_EXT) {
            call.respond(HttpStatusCode.UnsupportedMediaType, mapOf("error" to "unsupported extension .$ext"))
            return
        }
        manager.ensure("audio", model) // 503 loading počas prepnutia image→audio
        requireModelReady("whisper-small-q5_1")

        val tempDir = downloader.modelsDir()
        val pcmFloats = AudioDecoder.decodeToPcmFloats(bytes, tempDir)
        val modelFile = downloader.getModelFile("whisper-small-q5_1")
            ?: throw ModelWillNotLoadError("whisper-small-q5_1 model file missing", -1, -1)

        val text = WhisperLib.transcribe(
            modelFile = modelFile,
            samples = pcmFloats,
            language = language.ifEmpty { "sk" },
            translate = translation,
        )

        call.respond(
            mapOf(
                "text" to text,
                "language" to language,
                "duration" to (pcmFloats.size / 16000.0),
            )
        )
    }

    private fun requireModelReady(assetId: String) {
        if (!downloader.isReady(assetId)) {
            throw ModelWillNotLoadError("model $assetId not downloaded (see ModelDownloader)", -1, -1)
        }
    }

    companion object {
        /** 1×1 PNG stub, kým nie je zapojená NCNN inferencia. */
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
