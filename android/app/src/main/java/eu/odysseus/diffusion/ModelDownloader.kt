package eu.odysseus.diffusion

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.zip.ZipInputStream

/**
// ---------- custom models (own URL) ----------
 *
* Why not DownloadManager:
*  - DownloadManager cannot handle multi-file models (LCM DreamShaper has 7 files),
*  - its result cannot be verified when the Activity is dead,
*  - it cannot name the target subdirectory after the model.
 *
* Guarantees:
 *  - resume via HTTP Range (`<file>.part`),
*  - staging `models/.staging/<id>/` → atomic move after completion,
*  - `.READY` manifest with sizes and SHA-256 (when known),
*  - progress available through both API and UI.
 */
enum class InstallState { IDLE, QUEUED, DOWNLOADING, VERIFYING, DONE, FAILED, CANCELLED }

data class InstallProgress(
    val assetId: String,
    var state: InstallState = InstallState.IDLE,
    var bytesDone: Long = 0,
    var bytesTotal: Long = 0,
    var fileIndex: Int = 0,
    var fileCount: Int = 0,
    var error: String = "",
) {
    val percent: Int
        get() = if (bytesTotal <= 0) 0 else ((bytesDone * 100) / bytesTotal).toInt().coerceIn(0, 100)
}

class ModelDownloader(private val ctx: Context) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val jobs = ConcurrentHashMap<String, Job>()
    private val states = ConcurrentHashMap<String, InstallProgress>()
    private val prefs = ctx.getSharedPreferences("models", Context.MODE_PRIVATE)

    // ---------- cesty ----------

    fun modelsDir(): File = File(ctx.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun modelDir(assetId: String): File = File(modelsDir(), assetId)

    private fun stagingDir(assetId: String): File =
        File(modelsDir(), ".staging/$assetId").apply { mkdirs() }

    private fun manifest(assetId: String): File = File(modelsDir(), "$assetId.READY")

    // ---------- stav ----------

/** True only if the manifest exists and all files from it are present. */
    fun isReady(assetId: String): Boolean {
        val man = manifest(assetId)
        val dir = modelDir(assetId)
        if (!man.exists()) return false
        return try {
            val json = JSONObject(man.readText())
            val files = json.optJSONArray("files") ?: return false
            for (i in 0 until files.length()) {
                val o = files.getJSONObject(i)
                val f = File(dir, o.getString("path"))
                if (!f.isFile || f.length() <= 0) return false
            }
            true
        } catch (e: Exception) {
// Legacy marker from the DownloadManager era ("OK\nsha256=…"): the dir with
// weights is valid, so migrate it to a JSON manifest.
            Log.w(TAG, "legacy marker for $assetId, migrating")
            migrateLegacy(assetId, dir)
        }
    }

    private fun migrateLegacy(assetId: String, dir: File): Boolean {
        if (!dir.isDirectory) return false
        val files = dir.walkTopDown().filter { it.isFile && it.length() > 0 }.toList()
        if (files.isEmpty()) return false
        val jsonFiles = JSONArray()
        files.forEach { f ->
            jsonFiles.put(
                JSONObject()
                    .put("path", f.relativeTo(dir).path.replace('\\', '/'))
                    .put("size", f.length())
                    .put("sha256", "")
            )
        }
        manifest(assetId).writeText(
            JSONObject()
                .put("id", assetId)
                .put("kind", "")
                .put("engine", "")
                .put("files", jsonFiles)
                .put("migrated", true)
                .toString(2)
        )
        return true
    }

    fun installedSizeMb(assetId: String): Long {
        val dir = modelDir(assetId)
        if (!dir.exists()) return 0
        return dir.walkTopDown().filter { it.isFile }.sumOf { it.length() } / (1024 * 1024)
    }

/** Main model file (whisper .bin, onnx, gguf, safetensors…). */
    fun getModelFile(assetId: String): File? {
        val dir = modelDir(assetId)
        if (!dir.isDirectory) return null
        val exts = setOf("bin", "gguf", "onnx", "param", "safetensors")
        return dir.walkTopDown().firstOrNull { it.isFile && it.extension.lowercase() in exts }
            ?: dir.walkTopDown().firstOrNull { it.isFile }
    }

    fun progress(assetId: String): InstallProgress? = states[assetId]

    fun allProgress(): Map<String, InstallProgress> = states.toMap()

    fun isDownloading(assetId: String): Boolean = jobs.containsKey(assetId)

// ---------- installation ----------

    fun enqueue(asset: ModelAsset) {
        if (asset.files.isEmpty()) throw IOException("model ${asset.id} has no download source")
        if (isDownloading(asset.id)) return
        states[asset.id] = InstallProgress(asset.id, state = InstallState.QUEUED)
        jobs[asset.id] = scope.launch { runDownload(asset) }
    }

    fun cancel(assetId: String) {
        jobs[assetId]?.cancel()
        jobs.remove(assetId)
        states[assetId]?.state = InstallState.CANCELLED
    }

    fun uninstall(assetId: String): Boolean {
        cancel(assetId)
        states[assetId] = InstallProgress(assetId, state = InstallState.IDLE)
        modelDir(assetId).deleteRecursively()
        stagingDir(assetId).deleteRecursively()
        manifest(assetId).delete()
        if (customAssets().any { it.id == assetId }) removeCustom(assetId)
        return !modelDir(assetId).exists() && !manifest(assetId).exists()
    }

// ---------- custom models (own URL) ----------

    fun customAssets(): List<ModelAsset> {
        val raw = prefs.getString(KEY_CUSTOM, null) ?: return emptyList()
        return try {
            val arr = JSONArray(raw)
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                ModelAsset(
                    id = o.getString("id"),
                    kind = o.optString("kind", "custom"),
                    label = o.optString("label", o.getString("id")),
                    engine = o.optString("engine", "custom"),
                    note = o.optString("note", "Custom model (URL)"),
                    custom = true,
                    files = listOf(
                        ModelFile(
                            path = o.getString("file"),
                            url = o.getString("url"),
                            sizeMb = o.optLong("sizeMb", 0),
                        ),
                    ),
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "bad custom model list: ${e.message}")
            emptyList()
        }
    }

/** All models = built-in catalog + user models. */
    fun allAssets(): List<ModelAsset> = ModelCatalog.assets + customAssets()

    fun findAsset(id: String): ModelAsset? = allAssets().firstOrNull { it.id == id }

    /**
* Register a custom model from any direct URL (GGUF/ONNX/bin/safetensors)
* and start downloading it into `models/<name>/` immediately.
     */
    fun addCustom(name: String, kind: String, url: String, label: String = ""): ModelAsset {
        val clean = sanitizeName(name)
        require(clean.isNotBlank()) { "invalid model name" }
        require(url.startsWith("http://") || url.startsWith("https://")) { "url must be http(s)" }
        if (findAsset(clean) != null) throw IOException("model '$clean' already exists")

        val fileName = url.substringAfterLast('/').substringBefore('?').ifBlank { "$clean.bin" }
        val asset = ModelAsset(
            id = clean,
            kind = kind.ifBlank { "custom" },
            label = label.ifBlank { clean },
            engine = "custom",
            note = "Custom model from $url",
            custom = true,
            files = listOf(ModelFile(path = fileName, url = url, sizeMb = 0)),
        )
        val arr = try { JSONArray(prefs.getString(KEY_CUSTOM, "[]") ?: "[]") } catch (_: Exception) { JSONArray() }
        arr.put(
            JSONObject()
                .put("id", asset.id)
                .put("kind", asset.kind)
                .put("label", asset.label)
                .put("url", url)
                .put("file", fileName)
                .put("sizeMb", 0)
        )
        prefs.edit().putString(KEY_CUSTOM, arr.toString()).apply()
        enqueue(asset)
        return asset
    }

    fun removeCustom(assetId: String) {
        val arr = try { JSONArray(prefs.getString(KEY_CUSTOM, "[]") ?: "[]") } catch (_: Exception) { JSONArray() }
        val out = JSONArray()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            if (o.optString("id") != assetId) out.put(o)
        }
        prefs.edit().putString(KEY_CUSTOM, out.toString()).apply()
    }

    private fun sanitizeName(name: String): String =
        name.trim().replace(Regex("[^A-Za-z0-9._-]"), "_").replace(Regex("^[._]+"), "").take(64)

    // ---------- download engine ----------

    private suspend fun runDownload(asset: ModelAsset) {
        val st = states.getOrPut(asset.id) { InstallProgress(asset.id) }
        st.state = InstallState.QUEUED
        st.error = ""
        st.fileCount = asset.files.size
        st.bytesTotal = asset.files.sumOf { it.sizeMb } * 1024L * 1024L
        st.bytesDone = 0

        val staging = stagingDir(asset.id)
        var completed = 0L
        try {
            asset.files.forEachIndexed { index, file ->
                currentCoroutineContext().ensureActive()
                st.state = InstallState.DOWNLOADING
                st.fileIndex = index
                val target = File(staging, file.path)
                val base = completed
                downloadFile(
                    file, target,
                    onProgress = { done -> st.bytesDone = base + done },
                    onTotal = { total -> if (file.sizeMb <= 0 && total > 0) st.bytesTotal += total },
                )
                completed += if (file.sizeMb > 0) file.sizeMb * 1024L * 1024L else target.length()
                st.bytesDone = completed
            }

            st.state = InstallState.VERIFYING
            val target = modelDir(asset.id).apply { deleteRecursively(); mkdirs() }
            staging.copyRecursively(target, overwrite = true)
            staging.deleteRecursively()
            writeManifest(asset, target)
            st.bytesDone = st.bytesTotal.coerceAtLeast(completed)
            st.state = InstallState.DONE
            Log.i(TAG, "installed ${asset.id} (${st.bytesDone / 1048576} MB)")
        } catch (e: CancellationException) {
            st.state = InstallState.CANCELLED
            throw e
        } catch (e: Exception) {
            st.state = InstallState.FAILED
            st.error = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "download ${asset.id} failed: ${e.message}", e)
        } finally {
            jobs.remove(asset.id)
        }
    }

    private suspend fun downloadFile(
        file: ModelFile,
        target: File,
        onProgress: (Long) -> Unit,
        onTotal: (Long) -> Unit,
    ) {
        target.parentFile?.mkdirs()
        val part = File(target.parentFile, target.name + ".part")
        val expected = file.sizeMb * 1024L * 1024L
        var existing = if (part.exists()) part.length() else 0L

        if (expected > 0 && existing == expected) {
            replace(target, part)
            verifySha(file, target)
            return
        }

        val conn = (URL(file.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Odysseus/1.0 (Android)")
            if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
        }

        try {
            val code = conn.responseCode
            if (code !in 200..299) throw IOException("HTTP $code for ${file.url}")

            val append = code == 206 && existing > 0
            if (!append) existing = 0L
            val total = conn.contentLengthLong + existing
            if (total > 0) onTotal(total)
            val buffer = ByteArray(256 * 1024)
            var done = existing

            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val n = input.read(buffer)
                        if (n < 0) break
                        out.write(buffer, 0, n)
                        done += n
                        onProgress(done)
                    }
                }
            }
            Log.i(TAG, "downloaded ${file.path}: ${done / 1048576} MB / ${if (total > 0) total / 1048576 else '?'} MB")
        } finally {
            conn.disconnect()
        }

        if (part.length() <= 0) throw IOException("empty file ${file.path}")
        replace(target, part)
        verifySha(file, target)
    }

    private fun replace(target: File, part: File) {
        if (target.exists()) target.delete()
        if (!part.renameTo(target)) throw IOException("rename failed: ${part.name}")
    }

    private fun verifySha(file: ModelFile, target: File) {
        val expected = file.sha256
        if (expected.isBlank() || expected.startsWith("TODO")) return
        val actual = sha256(target)
        if (!actual.equals(expected, ignoreCase = true)) {
            target.delete()
            throw IOException("sha256 mismatch for ${file.path}")
        }
    }

    private fun writeManifest(asset: ModelAsset, dir: File) {
        // Archives are extracted first: the manifest must describe the files the
        // engine will load. Recording the .zip path (deleted by unzipArchives)
        // made isReady() false forever, so upscale answered 503 "not downloaded"
        // even though the download had finished.
        unzipArchives(dir)
        val shaByName = asset.files.associate { it.path to it.sha256 }
        val files = JSONArray()
        dir.walkTopDown()
            .filter { it.isFile && it.length() > 0 }
            .sortedBy { it.absolutePath }
            .forEach { f ->
                val rel = f.relativeTo(dir).invariantSeparatorsPath
                files.put(
                    JSONObject()
                        .put("path", rel)
                        .put("size", f.length())
                        .put("sha256", shaByName[rel] ?: "")
                )
            }
        if (files.length() == 0) {
            throw java.io.IOException("install finished with no files in ${dir.name}")
        }
        val root = JSONObject()
            .put("id", asset.id)
            .put("kind", asset.kind)
            .put("engine", asset.engine)
            .put("files", files)
            .put("finished_at", System.currentTimeMillis())
        manifest(asset.id).writeText(root.toString(2))
        Log.i(TAG, "manifest ${asset.id}: ${files.length()} file(s)")
    }

    private fun unzipArchives(dir: File) {
        val zips = dir.listFiles { f -> f.isFile && f.extension.equals("zip", true) } ?: return
        for (zipFile in zips) {
            try {
                ZipInputStream(zipFile.inputStream().buffered()).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        val out = File(dir, entry.name)
                        if (!out.canonicalPath.startsWith(dir.canonicalPath)) {
                            Log.w(TAG, "zip slip skipped: ${entry.name}")
                        } else if (entry.isDirectory) {
                            out.mkdirs()
                        } else {
                            out.parentFile?.mkdirs()
                            FileOutputStream(out).use { zip.copyTo(it) }
                        }
                        zip.closeEntry()
                        entry = zip.nextEntry
                    }
                }
                zipFile.delete()
            } catch (e: Exception) {
                Log.w(TAG, "unzip failed for ${zipFile.name}: ${e.message}")
            }
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(256 * 1024)
            var read: Int
            while (input.read(buffer).also { read = it } != -1) digest.update(buffer, 0, read)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "ModelDownloader"
        private const val KEY_CUSTOM = "custom_models"
    }
}
