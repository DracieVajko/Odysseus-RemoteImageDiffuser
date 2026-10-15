package eu.odysseus.diffusion

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Single-resident gate — zrkadlo `unified/model_manager.py`.
 *
 * Na mobile je budget ~3–4 GB (SD INT8 ~2.5 GB + Whisper small ~350 MB +
 * Real-ESRGAN ~150 MB sa NIKDY nedržia naraz). Prepnutie kind/id =
 * unload + load. Počas loadu volajúci dostane [ModelLoadingError]
 * (HTTP 503 `loading model`, retry_after 30), nikdy corrupt výstup.
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
)

class ModelManager {
    private val lock = ReentrantLock()
    private var loadedKind: String? = null
    private var loadedId: String? = null
    private var loading = false
    private var loadingModel = ""
    private var busy = false
    private var queue = 0
    var loadCount = 0
        private set

    fun setLoading(modelId: String, active: Boolean) = lock.withLock {
        loading = active
        loadingModel = if (active) modelId else ""
    }

    /**
     * Zaistí rezidentný model. Vráti efektívne id.
     * @throws ModelLoadingError počas cold loadu (→ 503 loading)
     */
    fun ensure(kind: String, modelId: String): String = lock.withLock {
        if (loading) throw ModelLoadingError(loadingModel.ifEmpty { modelId })
        if (loadedKind == kind && loadedId == modelId) return modelId
        // Prepnutie = najprv unload predchádzajúceho (uvoľní ~GB RAM).
        if (loadedKind != null) doUnloadLocked("switch")
        loadedKind = kind
        loadedId = modelId
        loadCount++
        return modelId
    }

    fun unload(reason: String = "idle"): Boolean = lock.withLock {
        doUnloadLocked(reason)
    }

    private fun doUnloadLocked(reason: String): Boolean {
        @Suppress("UNUSED_PARAMETER")
        val ignored = reason
        val had = loadedKind != null
        loadedKind = null
        loadedId = null
        // JVM: hint GC; natívne NCNN/whisper handlery uvoľňuje volajúci
        // cez InferenceHandle.close() pred unload() — viď ApiServer.
        System.gc()
        return had
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
        )
    }
}
