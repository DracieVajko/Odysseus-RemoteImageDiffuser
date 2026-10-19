package eu.odysseus.diffusion

import android.app.DownloadManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.NotificationCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * UI: status karta (server beží/stopped + detail), mód Local/Auto/Remote,
 * remote host, modely s reálnym progress barom (DownloadManager), activity log.
 *
 * Server beží ako ForegroundService (notifikácia), takže prežije aj
 * minimalizáciu appky. Všetky akcie sa logujú do ACTIVITY panela.
 */
enum class RunMode { LOCAL, REMOTE, AUTO }

class MainActivity : AppCompatActivity() {
    private lateinit var manager: ModelManager
    private lateinit var downloader: ModelDownloader

    private lateinit var statusDot: android.view.View
    private lateinit var statusTitle: TextView
    private lateinit var statusDetail: TextView
    private lateinit var btnToggle: MaterialButton
    private lateinit var modeGroup: RadioGroup
    private lateinit var remoteHost: TextInputEditText
    private lateinit var btnWebUi: MaterialButton
    private lateinit var modelsContainer: LinearLayout
    private lateinit var activityLog: TextView
    private lateinit var btnClearLog: TextView

    private val logLines = ArrayDeque<String>()
    private val handler = Handler(Looper.getMainLooper())
    private var serverRunning = false

    private val prefs by lazy { getSharedPreferences("dlmap", Context.MODE_PRIVATE) }

    // id modelu -> (progressBar, actionButton, meta)
    private val modelViews = mutableMapOf<String, Triple<ProgressBar, MaterialButton, TextView>>()

    private val ticker = object : Runnable {
        override fun run() {
            refreshServerState()
            refreshDownloads()
            handler.postDelayed(this, 2000)
        }
    }

    private val dlReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            val id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L)
            if (id < 0) return
            val assetId = prefs.getString("dl_$id", null) ?: return
            prefs.edit().remove("dl_$id").apply()
            val asset = ModelCatalog.assets.firstOrNull { it.id == assetId } ?: return
            val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            dm.query(DownloadManager.Query().setFilterById(id))?.use { c ->
                if (!c.moveToFirst()) return
                val status = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                if (status != DownloadManager.STATUS_SUCCESSFUL) {
                    log("download ${asset.id} FAILED (status=$status)")
                    renderModelRow(asset)
                    return
                }
                val uri = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI))
                val file = File(java.net.URI(uri))
                log("verifying ${asset.id} (SHA-256)…")
                Thread {
                    val ok = downloader.verifyAndMark(asset, file)
                    handler.post {
                        log(if (ok) "${asset.id} READY" else "${asset.id} checksum/unzip FAILED")
                        renderModelRow(asset)
                    }
                }.start()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        manager = AppGraph.manager
        downloader = ModelDownloader(this)

        statusDot = findViewById(R.id.statusDot)
        statusTitle = findViewById(R.id.statusTitle)
        statusDetail = findViewById(R.id.statusDetail)
        btnToggle = findViewById(R.id.btnToggle)
        modeGroup = findViewById(R.id.modeGroup)
        remoteHost = findViewById(R.id.remoteHost)
        btnWebUi = findViewById(R.id.btnWebUi)
        modelsContainer = findViewById(R.id.modelsContainer)
        activityLog = findViewById(R.id.activityLog)
        btnClearLog = findViewById(R.id.btnClearLog)

        val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(dlReceiver, filter, Context.RECEIVER_EXPORTED)
        else @Suppress("UnspecifiedRegisterReceiverFlag") registerReceiver(dlReceiver, filter)

        btnToggle.setOnClickListener { toggleServer() }
        btnWebUi.setOnClickListener {
            if (!serverRunning) {
                toast("Start the server first")
                log("WebUI needs the local server (press START)")
            } else {
                startActivity(Intent(this, WebUIActivity::class.java))
            }
        }
        btnClearLog.setOnClickListener { logLines.clear(); renderLog() }

        modeGroup.setOnCheckedChangeListener { _, _ ->
            val mode = selectedMode()
            log("mode -> ${mode.name}")
            if (mode == RunMode.REMOTE) log("remote: ${remoteHost.text}")
        }

        renderModels()
        renderLog()
        log("app started")
        refreshServerState()
        refreshDownloads()
        handler.postDelayed(ticker, 2000)
    }

    override fun onDestroy() {
        handler.removeCallbacks(ticker)
        try { unregisterReceiver(dlReceiver) } catch (_: Exception) {}
        super.onDestroy()
    }

    // ---------- server ----------

    private fun selectedMode(): RunMode = when (modeGroup.checkedRadioButtonId) {
        R.id.modeLocal -> RunMode.LOCAL
        R.id.modeRemote -> RunMode.REMOTE
        else -> RunMode.AUTO
    }

    private fun toggleServer() {
        if (serverRunning) {
            stopService(Intent(this, DiffusionService::class.java))
            serverRunning = false
            log("server stop requested")
            renderServerState()
        } else {
            val mode = selectedMode()
            val i = Intent(this, DiffusionService::class.java).apply {
                putExtra("mode", mode.name)
                putExtra("remote", remoteHost.text.toString())
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(i)
            else startService(i)
            serverRunning = true
            log("server starting (mode=${mode.name})…")
            renderServerState()
        }
    }

    /** Ping local :8101/health on a background thread; updates the status card. */
    private fun refreshServerState() {
        Thread {
            val json = httpGet("http://127.0.0.1:8101/health")
            val parsed = try {
                if (json != null) JSONObject(json) else null
            } catch (_: Exception) { null }
            handler.post {
                if (parsed != null) {
                    serverRunning = true
                    val loaded = parsed.optString("current_model", "")
                    val device = parsed.optString("device", "?")
                    val loading = parsed.optBoolean("loading", false)
                    val model = parsed.optString("model", "?")
                    statusTitle.text = "Server running"
                    statusDetail.text = buildString {
                        append("http://127.0.0.1:8101  ·  device=$device\n")
                        append("default=$model")
                        if (loaded.isNotBlank()) append("  loaded=$loaded")
                        if (loading) append("  loading…")
                    }
                    btnToggle.text = "STOP"
                    btnToggle.setBackgroundColor(0xFFDA3633.toInt())
                    statusDot.setBackgroundResource(R.drawable.dot_running)
                } else {
                    if (serverRunning) {
                        // starting up (or crashed)
                        statusTitle.text = "Server starting…"
                        statusDetail.text = "waiting for http://127.0.0.1:8101/health"
                        statusDot.setBackgroundResource(R.drawable.dot_busy)
                    } else {
                        renderServerState()
                    }
                }
            }
        }.start()
    }

    private fun renderServerState() {
        if (serverRunning) {
            statusTitle.text = "Server starting…"
            statusDetail.text = "waiting for http://127.0.0.1:8101/health"
            statusDot.setBackgroundResource(R.drawable.dot_busy)
            btnToggle.text = "STOP"
            btnToggle.setBackgroundColor(0xFFDA3633.toInt())
        } else {
            statusTitle.text = "Server stopped"
            statusDetail.text = "Press START to run the local API on port 8101"
            statusDot.setBackgroundResource(R.drawable.dot_stopped)
            btnToggle.text = "START"
            btnToggle.setBackgroundColor(0xFF238636.toInt())
        }
    }

    // ---------- models ----------

    private fun renderModels() {
        modelsContainer.removeAllViews()
        modelViews.clear()
        for (asset in ModelCatalog.assets) {
            val row = layoutInflater.inflate(R.layout.item_model, modelsContainer, false)
            val name = row.findViewById<TextView>(R.id.modelName)
            val meta = row.findViewById<TextView>(R.id.modelMeta)
            val action = row.findViewById<MaterialButton>(R.id.modelAction)
            val progress = row.findViewById<ProgressBar>(R.id.modelProgress)

            name.text = asset.id
            meta.text = "${asset.sizeMb} MB · ${asset.kind}"
            action.setOnClickListener {
                if (downloader.isReady(asset.id)) confirmUninstall(asset) else startDownload(asset)
            }

            modelsContainer.addView(row)
            modelViews[asset.id] = Triple(progress, action, meta)
            renderModelRow(asset)
        }
    }

    private fun renderModelRow(asset: ModelAsset) {
        val views = modelViews[asset.id] ?: return
        val (progress, action, meta) = views
        val ready = downloader.isReady(asset.id)
        when {
            ready -> {
                val mb = downloader.installedSizeMb(asset.id)
                action.text = "Uninstall"
                action.isEnabled = true
                action.setBackgroundColor(0xFFDA3633.toInt())
                action.setTextColor(0xFFFFFFFF.toInt())
                progress.visibility = ProgressBar.GONE
                meta.text = "${asset.sizeMb} MB · ${asset.kind} · installed (${mb} MB on disk)"
            }
            asset.url.isBlank() -> {
                action.text = "N/A"
                action.isEnabled = false
                progress.visibility = ProgressBar.GONE
                meta.text = "${asset.sizeMb} MB · ${asset.kind} · no verified source"
            }
            else -> {
                action.text = "Install"
                action.isEnabled = true
                action.setBackgroundColor(0xFF238636.toInt())
                action.setTextColor(0xFFFFFFFF.toInt())
                meta.text = "${asset.sizeMb} MB · ${asset.kind}"
            }
        }
    }

    private fun confirmUninstall(asset: ModelAsset) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Uninstall ${asset.id}?")
            .setMessage("This deletes the downloaded weights (${downloader.installedSizeMb(asset.id)} MB).")
            .setPositiveButton("Uninstall") { _, _ ->
                val ok = downloader.uninstall(asset.id)
                log(if (ok) "uninstalled ${asset.id}" else "uninstall ${asset.id} FAILED")
                renderModelRow(asset)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startDownload(asset: ModelAsset) {
        if (asset.url.isBlank()) {
            log("${asset.id}: no verified download source")
            return
        }
        try {
            val id = downloader.enqueue(asset)
            prefs.edit().putString("dl_$id", asset.id).putLong("dlid_${asset.id}", id).apply()
            log("downloading ${asset.id} (${asset.sizeMb} MB)…")
            val views = modelViews[asset.id] ?: return
            views.first.visibility = ProgressBar.VISIBLE
            views.first.progress = 0
            views.second.isEnabled = false
            views.second.text = "0%"
        } catch (e: Exception) {
            log("download ${asset.id} failed: ${e.message}")
        }
    }

    /** Poll DownloadManager for every tracked id and update progress bars. */
    private fun refreshDownloads() {
        val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        for ((assetId, _) in modelViews) {
            val dlId = prefs.getLong("dlid_$assetId", -1L)
            val realId = if (dlId > 0) dlId else null
            // fallback: scan prefs for any dl_<id> == assetId
            val idFromPrefs = if (realId == null) {
                prefs.all.entries.firstOrNull { it.value == assetId && it.key.startsWith("dl_") }
                    ?.key?.removePrefix("dl_")?.toLongOrNull()
            } else realId
            if (idFromPrefs == null) continue
            dm.query(DownloadManager.Query().setFilterById(idFromPrefs))?.use { c ->
                if (!c.moveToFirst()) return@use
                val status = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                val done = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                val total = c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                val views = modelViews[assetId] ?: return@use
                val (progress, action, meta) = views
                if (status == DownloadManager.STATUS_RUNNING || status == DownloadManager.STATUS_PENDING) {
                    val pct = if (total > 0) ((done * 100) / total).toInt() else 0
                    progress.visibility = ProgressBar.VISIBLE
                    progress.progress = pct
                    action.text = "$pct%"
                    action.isEnabled = false
                    meta.text = "${fmtBytes(done)} / ${fmtBytes(total)}"
                } else if (status == DownloadManager.STATUS_FAILED) {
                    progress.visibility = ProgressBar.GONE
                    action.isEnabled = true
                    action.text = "Retry"
                    prefs.edit().remove("dl_$idFromPrefs").apply()
                    log("download $assetId failed (manager status=FAILED)")
                }
            }
        }
    }

    // ---------- log ----------

    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        logLines.addFirst("$ts  $msg")
        while (logLines.size > 40) logLines.removeLast()
        renderLog()
    }

    private fun renderLog() {
        activityLog.text = if (logLines.isEmpty()) "ready" else logLines.joinToString("\n")
    }

    private fun toast(m: String) = Toast.makeText(this, m, Toast.LENGTH_SHORT).show()

    private fun fmtBytes(b: Long): String = when {
        b <= 0 -> "?"
        b >= 1_000_000_000 -> "%.2f GB".format(b / 1e9)
        b >= 1_000_000 -> "%.1f MB".format(b / 1e6)
        else -> "%.0f KB".format(b / 1e3)
    }

    private fun httpGet(url: String): String? {
        return try {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.connectTimeout = 1500
            conn.readTimeout = 1500
            conn.requestMethod = "GET"
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }
            conn.disconnect()
            if (code in 200..299) body else null
        } catch (_: Exception) {
            null
        }
    }
}

/** Procesný graf — jeden ModelManager pre celú appku (single-resident). */
object AppGraph {
    val manager = ModelManager()
}

/**
 * ForegroundService držiaci Ktor server pri živote.
 * Notifikácia ukazuje reálny stav (port + mód).
 */
class DiffusionService : Service() {
    private var server: ApiServer? = null

    override fun onCreate() {
        super.onCreate()
        val ch = NotificationChannel("srv", "Diffusion server", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(ch)
        startForeground(1, buildNotification("Server running on :8101"))
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, "srv")
            .setContentTitle("Odysseus Diffusion")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val mode = intent?.getStringExtra("mode") ?: "AUTO"
        val remote = intent?.getStringExtra("remote") ?: ""
        if (server == null) {
            server = ApiServer(AppGraph.manager, ModelDownloader(this), applicationContext).also { it.start(wait = false) }
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .notify(1, buildNotification("Server running on :8101 · mode=$mode" + if (remote.isNotBlank()) " · remote=$remote" else ""))
        }
        return START_STICKY
    }

    override fun onDestroy() {
        server?.stop()
        server = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
