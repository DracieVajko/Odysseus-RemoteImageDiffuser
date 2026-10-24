package eu.odysseus.diffusion

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Single-resident gate — mirrors `unified/model_manager.py`.
 *
 * Mobile budget is ~3–4 GB (SD INT8 ~2.5 GB + Whisper small ~350 MB +
* Real-ESRGAN ~150 MB are NEVER held at once). Switch kind/id =
* unload + load. During load the caller gets [ModelLoadingError]
* (HTTP 503 `loading model`, retry_after 30), never corrupt output.
 */
class ModelLoadingError(
    val loadingModel: String,
    val retryAfterSec: Int = 30,
) : Exception("loading model")

class ModelWillNotLoadError(
    val reason: String,
    val neededMb: Long = -1,
    val freeMb: Long = -1,
) : Exception("will not load: $reason")

data class ManagerStatus(
    val loadedKind: String?,
    val loadedId: String?,
    val loading: Boolean,
    val loadingModel: String,
    val busy: Boolean,
    val queue: Int,
    val active: Int,
)

class ModelManager {
    private val lock = ReentrantLock()
    private var loadedKind: String? = null
    private var loadedId: String? = null
    private var loading = false
    private var loadingModel = ""
    private var busy = false
    private var queue = 0
    private var active = 0
    var loadCount = 0
        private set

    fun setLoading(modelId: String, active: Boolean) = lock.withLock {
        loading = active
        loadingModel = if (active) modelId else ""
    }

    /**
* Ensure a resident model. Returns the effective id.
* @throws ModelLoadingError during cold load (→ 503 loading)
     */
    fun ensure(kind: String, modelId: String): String = lock.withLock {
        if (loading) throw ModelLoadingError(loadingModel.ifEmpty { modelId })
        if (loadedKind == kind && loadedId == modelId) return modelId
// Switch = unload the previous one first (frees ~GB of RAM).
        if (loadedKind != null) doUnloadLocked("switch")
        loadedKind = kind
        loadedId = modelId
        loadCount++
        return modelId
    }

    fun unload(reason: String = "idle"): Boolean = lock.withLock {
        doUnloadLocked(reason)
    }

    private fun doUnloadLocked(@Suppress("UNUSED_PARAMETER") reason: String): Boolean {
        val had = loadedKind != null
        loadedKind = null
        loadedId = null
// Native handles are released by the caller (WhisperLib/LcmOnnxEngine)
        // before this call; no System.gc() here (it can stall the Netty loop).
        return had
    }

    /** Called for every tracked image/audio request (Ktor intercept). */
    fun beginRequest() = lock.withLock {
        active++
        busy = true
        queue = (active - 1).coerceAtLeast(0)
    }

    fun endRequest() = lock.withLock {
        active = (active - 1).coerceAtLeast(0)
        busy = active > 0
        queue = (active - 1).coerceAtLeast(0)
    }

    fun markBusy(active: Boolean, queuedDelta: Int = 0) = lock.withLock {
        busy = active
        queue = (queue + queuedDelta).coerceAtLeast(0)
    }

    fun status(): ManagerStatus = lock.withLock {
        ManagerStatus(
            loadedKind = loadedKind,
            loadedId = loadedId,
            loading = loading,
            loadingModel = loadingModel,
            busy = busy,
            queue = queue,
            active = active,
        )
    }
}
