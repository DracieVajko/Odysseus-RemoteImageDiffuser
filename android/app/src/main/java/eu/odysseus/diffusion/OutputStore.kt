package eu.odysseus.diffusion

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Default output folder for every job on the phone.
 *
 * Files live under `Android/data/<pkg>/files/output/<kind>/` so the server can
 * list and serve them, and "Save to Downloads" copies them into the public
 * `Download/Prism` folder where any file manager can open them.
 */
object OutputStore {
    private const val TAG = "OutputStore"

    fun root(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "output").apply { mkdirs() }

    fun kindDir(ctx: Context, kind: String): File =
        File(root(ctx), kind).apply { mkdirs() }

    /** Write one artifact; returns the descriptor used in API responses. */
    fun store(ctx: Context, kind: String, ext: String, data: ByteArray, rid: String = ""): Map<String, Any> {
        val safeKind = if (kind in setOf("image", "video", "audio", "text")) kind else "text"
        val safeExt = if (ext.startsWith(".")) ext else ".$ext"
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val safeRid = rid.filter { it.isLetterOrDigit() || it == '-' || it == '_' }
            .take(24).trim('-', '_')
        val base = "$stamp-${safeRid.ifEmpty { System.currentTimeMillis().toString(36) }}"
        var target = File(kindDir(ctx, safeKind), "$base$safeExt")
        var n = 1
        while (target.exists()) {
            target = File(kindDir(ctx, safeKind), "$base-$n$safeExt")
            n++
        }
        target.writeBytes(data)
        return mapOf(
            "name" to target.name,
            "kind" to safeKind,
            "size" to data.size,
            "path" to "$safeKind/${target.name}",
            "url" to "/v1/outputs/$safeKind/${target.name}",
        )
    }

    fun list(ctx: Context, limit: Int = 200): List<File> {
        val all = ArrayList<File>()
        for (kind in listOf("image", "video", "audio", "text")) {
            kindDir(ctx, kind).listFiles()?.let { all.addAll(it) }
        }
        return all.filter { it.isFile }
            .sortedByDescending { it.lastModified() }
            .take(limit)
    }

    fun total(ctx: Context): Pair<Int, Long> {
        val files = list(ctx, 1000)
        return files.size to files.sumOf { it.length() }
    }

    /** Copy every output into the public Download/Prism folder (MediaStore). */
    fun saveAllToDownloads(ctx: Context): Int {
        var saved = 0
        for (f in list(ctx, 1000)) {
            try {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, f.name)
                    put(MediaStore.MediaColumns.MIME_TYPE, mimeOf(f))
                    put(MediaStore.MediaColumns.RELATIVE_PATH,
                        "${android.os.Environment.DIRECTORY_DOWNLOADS}/Prism")
                }
                val uri: Uri? = ctx.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    ctx.contentResolver.openOutputStream(uri)?.use { out ->
                        f.inputStream().use { it.copyTo(out) }
                    }
                    saved++
                }
            } catch (e: Exception) {
                Log.w(TAG, "save ${f.name} failed: ${e.message}")
            }
        }
        return saved
    }

    fun delete(file: File): Boolean = runCatching { file.delete() }.getOrDefault(false)

    private fun mimeOf(f: File): String = when (f.extension.lowercase()) {
        "png" -> "image/png"
        "jpg", "jpeg" -> "image/jpeg"
        "webp" -> "image/webp"
        "mp4" -> "video/mp4"
        "m4a" -> "audio/mp4"
        "mp3" -> "audio/mpeg"
        "wav" -> "audio/wav"
        "txt" -> "text/plain"
        else -> "application/octet-stream"
    }
}
