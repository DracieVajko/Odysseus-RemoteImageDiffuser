package eu.odysseus.diffusion

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import java.io.File

/**
 * Lazy download váh — modely NIE sú v APK (licencie + veľkosť).
 *
 * Cieľ: `app/files/models/<id>/<súbory>`. DownloadManager s resume,
 * po dokončení SHA-256 checksum. Bez stiahnutého modelu vracia
 * ApiServer `503 will-not-load` (nie hang, nie 500).
 *
 * Hlavný kandidát (research verdikt): LCM-Dreamshaper INT8 (NCNN/Vulkan).
 * Whisper minimum pre SK: small Q5_0 (whisper.cpp GGUF).
 * SD-Turbo (nekomerčná licencia) NIE je v zozname — len opt-in.
 * SSD-1B / XL / PaliGemma / SmolVLM / Moondream2 = PC-only / V2, nesťahovať.
 */
data class ModelAsset(
    val id: String,
    val kind: String, // "image" | "audio" | "upscale"
    val url: String, // TODO(PIN): doplniť finálne URL po schválení mirroru
    val sha256: String, // TODO(PIN): doplniť checksum
    val sizeMb: Long,
)

object ModelCatalog {
    val assets = listOf(
        ModelAsset(
            id = "lcm-dreamshaper-int8",
            kind = "image",
            url = "TODO_PIN_URL_lcm_dreamshaper_int8_ncnn_zip",
            sha256 = "TODO_PIN_SHA256",
            sizeMb = 2500,
        ),
        ModelAsset(
            id = "whisper-small-q5_0",
            kind = "audio",
            url = "TODO_PIN_URL_whisper_small_q5_0_gguf",
            sha256 = "TODO_PIN_SHA256",
            sizeMb = 350,
        ),
        ModelAsset(
            id = "realesrgan-x4v3-ncnn",
            kind = "upscale",
            url = "TODO_PIN_URL_realesrgan_x4v3_ncnn_zip",
            sha256 = "TODO_PIN_SHA256",
            sizeMb = 150,
        ),
    )

    fun forImageModel(registryId: String): ModelAsset =
        assets.first { it.kind == "image" } // registryId → asset mapovanie TODO po NCNN konverzii
}

class ModelDownloader(private val ctx: Context) {
    fun modelsDir(): File = File(ctx.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun isReady(assetId: String): Boolean {
        val marker = File(modelsDir(), "$assetId.READY")
        return marker.exists()
    }

    /** Zaradí sťahovanie do fronty, vráti system download id. */
    fun enqueue(asset: ModelAsset): Long {
        val dm = ctx.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val req = DownloadManager.Request(Uri.parse(asset.url))
            .setTitle("Odysseus: sťahujem ${asset.id}")
            .setDescription("Model ${asset.id} (~${asset.sizeMb} MB), nevypínajte appku")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationInExternalFilesDir(ctx, Environment.DIRECTORY_DOWNLOADS, "models/${asset.id}.zip")
            .setAllowedOverMetered(true)
        return dm.enqueue(req)
    }

    /** Overí SHA-256 a vytvorí .READY marker. Volá sa po dokončení downloadu. */
    fun verifyAndMark(asset: ModelAsset, zipFile: File): Boolean {
        // TODO: stream SHA-256 + rozbalenie do models/<id>/ + .READY marker.
        // Bez overeného checksumu model NIKDY nehlásime ako pripravený.
        return false // stub: vždy "nepripravené" → server vracia 503 will-not-load
    }
}
