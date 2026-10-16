package eu.odysseus.diffusion

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/**
 * Lazy download váh — modely NIE sú v APK (licencie + veľkosť).
 *
 * Cieľ: `app/files/models/<id>/<súbory>`. DownloadManager s resume,
 * po dokončení SHA-256 checksum. Bez stiahnutého modelu vracia
 * ApiServer `503 will-not-load` (nie hang, nie 500).
 *
 * Hlavný kandidát (research verdikt): LCM-Dreamshaper INT8 (NCNN/Vulkan).
 * Whisper minimum pre SK: small q5_1 (whisper.cpp GGML, PIN v katalógu).
 * SD-Turbo (nekomerčná licencia) NIE je v zozname — len opt-in.
 * SSD-1B / XL / PaliGemma / SmolVLM / Moondream2 = PC-only / V2, nesťahovať.
 */
data class ModelAsset(
    val id: String,
    val kind: String, // "image" | "audio" | "upscale"
    val url: String,
    val sha256: String,
    val sizeMb: Long,
)

object ModelCatalog {
    // PINy overené 2026-10-16 priamym downloadom z oficiálnych zdrojov
    // (whisper SHA + magic "ggml" + veľkosť; realesr zip SHA + vnútorné modely).
    // Pravidlo: žiadne URL bez HEAD/API dôkazu —insky 404 (viď lcm nižšie).
    val assets = listOf(
        ModelAsset(
            id = "lcm-dreamshaper-int8",
            kind = "image",
            // NEOVERENÉ: https://huggingface.co/ncnn/lcm-dreamshaper-v7-int8-ncnn
            // neexistuje (HF API 401/404). URL sa doplní až po nájdení reálneho
            // NCNN buildu alebo po vlastnej pnnx konverzii na PC. Dovtedy image
            // endpointy na mobile odpovedajú 503 will-not-load (alebo Remote proxy).
            url = "",
            sha256 = "",
            sizeMb = 2500,
        ),
        ModelAsset(
            id = "whisper-small-q5_1",
            kind = "audio",
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
            sha256 = "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
            sizeMb = 181,
        ),
        ModelAsset(
            id = "realesrgan-x4plus-ncnn",
            kind = "upscale",
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
            sha256 = "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96",
            sizeMb = 45,
            // Balík neobsahuje x4v3/general — pre foto použi vnútorné
            // models/realesrgan-x4plus.param + .bin (33 MB, všeobecné foto x4).
        ),
    )

    fun forImageModel(registryId: String): ModelAsset =
        assets.firstOrNull { it.kind == "image" && it.id == registryId }
            ?: assets.first { it.kind == "image" }
}

class ModelDownloader(private val ctx: Context) {
    fun modelsDir(): File = File(ctx.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun isReady(assetId: String): Boolean {
        val marker = File(modelsDir(), "$assetId.READY")
        return marker.exists()
    }

    fun modelDir(assetId: String): File = File(modelsDir(), assetId)

    /** Vráti cestu k hlavnému súboru modelu v adresári `models/<assetId>/`. */
    fun getModelFile(assetId: String): File? {
        val dir = modelDir(assetId)
        if (!dir.exists() || !dir.isDirectory) return null
        return dir.listFiles()?.firstOrNull { file ->
            val ext = file.extension.lowercase()
            ext == "bin" || ext == "gguf" || ext == "param" || ext == "onnx"
        } ?: dir.listFiles()?.firstOrNull()
    }

    /** Zaradí sťahovanie do fronty, vráti system download id. */
    fun enqueue(asset: ModelAsset): Long {
        val dm = ctx.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val filename = asset.url.substringAfterLast('/')
        val req = DownloadManager.Request(Uri.parse(asset.url))
            .setTitle("Odysseus: sťahujem ${asset.id}")
            .setDescription("Model ${asset.id} (~${asset.sizeMb} MB), nevypínajte appku")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationInExternalFilesDir(ctx, Environment.DIRECTORY_DOWNLOADS, "models/$filename")
            .setAllowedOverMetered(true)
        return dm.enqueue(req)
    }

    /** Overí SHA-256 a vytvorí .READY marker. Volá sa po dokončení downloadu. */
    fun verifyAndMark(asset: ModelAsset, downloadedFile: File): Boolean {
        if (!downloadedFile.exists() || downloadedFile.length() == 0L) {
            return false
        }

        // Ak je zadaný SHA-256 checksum, skontroluj ho.
        if (asset.sha256.isNotBlank() && !asset.sha256.startsWith("TODO")) {
            val calculatedSha = calculateSha256(downloadedFile)
            if (!calculatedSha.equals(asset.sha256, ignoreCase = true)) {
                downloadedFile.delete()
                return false
            }
        }

        val targetDir = modelDir(asset.id).apply { mkdirs() }

        try {
            if (downloadedFile.name.endsWith(".zip", ignoreCase = true)) {
                ZipInputStream(FileInputStream(downloadedFile)).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        val outFile = File(targetDir, entry.name)
                        if (entry.isDirectory) {
                            outFile.mkdirs()
                        } else {
                            outFile.parentFile?.mkdirs()
                            FileOutputStream(outFile).use { out -> zip.copyTo(out) }
                        }
                        entry = zip.nextEntry
                    }
                }
            } else {
                val targetFile = File(targetDir, downloadedFile.name)
                downloadedFile.copyTo(targetFile, overwrite = true)
            }

            // Vytvor .READY marker súbor
            val readyMarker = File(modelsDir(), "${asset.id}.READY")
            readyMarker.writeText("OK\nsha256=${asset.sha256}\nsize=${downloadedFile.length()}")

            downloadedFile.delete()
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    private fun calculateSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (input.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
