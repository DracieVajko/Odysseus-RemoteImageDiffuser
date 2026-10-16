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
import android.os.IBinder
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import androidx.core.app.NotificationCompat

/**
 * UI: Local / Remote / Auto prepínač + Start/Stop servera + Download modelov.
 * Každý asset z ModelCatalog má riadok so stavom a tlačidlom Download
 * (lazy sťahovanie cez DownloadManager → SHA-256 → .READY marker).
 * Bez zdroja (LCM, kým nie je overený NCNN build) je tlačidlo vypnuté.
 *
 * - Local: Ktor server na :8101, inferencia na telefóne.
 * - Remote: tenký klient na PC server (rovnaká zmluva, base URL z poľa nižšie).
 * - Auto (default): skús Local, pri 503 will-not-load / OOM fallback na Remote.
 */
enum class RunMode { LOCAL, REMOTE, AUTO }

class MainActivity : AppCompatActivity() {
    private lateinit var manager: ModelManager
    private lateinit var downloader: ModelDownloader
    private lateinit var status: TextView
    private lateinit var modelsBox: LinearLayout
    private var modePosition = RunMode.AUTO.ordinal

    private val prefs by lazy { getSharedPreferences("dlmap", Context.MODE_PRIVATE) }

    private val dlReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            val id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L)
            if (id < 0) return
            val assetId = prefs.getString("dl_$id", null) ?: return
            val asset = ModelCatalog.assets.firstOrNull { it.id == assetId } ?: return
            val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val q = DownloadManager.Query().setFilterById(id)
            dm.query(q)?.use { c ->
                if (!c.moveToFirst()) return
                val ok = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)) ==
                    DownloadManager.STATUS_SUCCESSFUL
                if (!ok) {
                    toast("Download ${asset.id} zlyhal")
                    return
                }
                val uri = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI))
                val file = java.io.File(java.net.URI(uri))
                val ready = downloader.verifyAndMark(asset, file)
                toast(if (ready) "${asset.id} pripravený" else "${asset.id}: checksum/unzip zlyhal")
                renderAll()
            }
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        manager = AppGraph.manager
        downloader = ModelDownloader(this)

        val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(dlReceiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(dlReceiver, filter)
        }

        status = TextView(this)
        val hostInput = EditText(this).apply {
            hint = "Remote host, napr. http://192.168.1.50:8101"
            setText("http://192.168.1.50:8101")
        }
        val modeSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                RunMode.values().map { it.name },
            )
            setSelection(modePosition)
        }
        val startBtn = Button(this).apply { text = "Start server" }
        val stopBtn = Button(this).apply { text = "Stop server" }
        val refreshBtn = Button(this).apply { text = "Refresh status" }
        modelsBox = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        fun renderStatus() {
            val st = manager.status()
            status.text =
                "mode=${RunMode.values()[modePosition]}\n" +
                "loaded=${st.loadedKind}/${st.loadedId} loading=${st.loading} " +
                "busy=${st.busy} queue=${st.queue}"
        }

        startBtn.setOnClickListener {
            modePosition = modeSpinner.selectedItemPosition
            val mode = RunMode.values()[modePosition]
            val i = Intent(this, DiffusionService::class.java).apply {
                putExtra("mode", mode.name)
                putExtra("remote", hostInput.text.toString())
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(i)
            else startService(i)
            renderStatus()
        }
        stopBtn.setOnClickListener {
            stopService(Intent(this, DiffusionService::class.java))
            renderStatus()
        }
        refreshBtn.setOnClickListener { renderAll() }
        renderAll()

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(modeSpinner)
            addView(hostInput)
            addView(startBtn)
            addView(stopBtn)
            addView(refreshBtn)
            addView(status)
            addView(modelsBox)
        }
        setContentView(layout)
    }

    override fun onDestroy() {
        try {
            unregisterReceiver(dlReceiver)
        } catch (_: Exception) {
        }
        super.onDestroy()
    }

    private fun renderAll() {
        val st = manager.status()
        status.text =
            "mode=${RunMode.values()[modePosition]}\n" +
            "loaded=${st.loadedKind}/${st.loadedId} loading=${st.loading} " +
            "busy=${st.busy} queue=${st.queue}"
        modelsBox.removeAllViews()
        for (asset in ModelCatalog.assets) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            val ready = downloader.isReady(asset.id)
            val label = TextView(this).apply {
                text = "${asset.id} (~${asset.sizeMb} MB): " +
                    if (ready) "READY" else "chýba"
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            val btn = Button(this).apply {
                text = when {
                    ready -> "✓"
                    asset.url.isBlank() -> "bez zdroja"
                    else -> "Download"
                }
                isEnabled = !ready && asset.url.isNotBlank()
                setOnClickListener {
                    val id = downloader.enqueue(asset)
                    prefs.edit().putString("dl_$id", asset.id).apply()
                    toast("Sťahujem ${asset.id} (~${asset.sizeMb} MB)")
                }
            }
            row.addView(label)
            row.addView(btn)
            modelsBox.addView(row)
        }
    }
}

/** Procesný graf — jeden ModelManager pre celú appku (single-resident). */
object AppGraph {
    val manager = ModelManager()
}

/**
 * ForegroundService držiaci Ktor server pri živote.
 * V Auto režime klientská vrstva (okhttp — TODO) preposiela na Remote
 * pri 503 will-not-load z Local.
 */
class DiffusionService : Service() {
    private var server: ApiServer? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var remoteBase: String = ""

    override fun onCreate() {
        super.onCreate()
        val ch = NotificationChannel("srv", "Diffusion server", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(ch)
        val n: Notification = NotificationCompat.Builder(this, "srv")
            .setContentTitle("Odysseus Diffusion :8101")
            .setContentText("Lokálny server beží")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .build()
        startForeground(1, n)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val mode = intent?.getStringExtra("mode") ?: "AUTO"
        val remote = intent?.getStringExtra("remote") ?: ""
        remoteBase = remote
        // Android build.gradle.kts už má ktor-client-core + okhttp, takže žiadna
    // nová závislosť. TODO: neskôr pridať caching + retry.
    // Ruling: Auto režim je len prepínač; reálny fallback sa robí v UI
    // cez RemoteProxy.health() — nie v ApiServer.kt (žiadna sieťová závislosť
    // v serveri samotnom, aby zostal testovateľný).
        if (server == null) {
            server = ApiServer(AppGraph.manager, ModelDownloader(this)).also { it.start(wait = false) }
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
