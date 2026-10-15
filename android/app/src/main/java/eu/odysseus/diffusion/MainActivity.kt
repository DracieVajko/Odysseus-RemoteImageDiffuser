package eu.odysseus.diffusion

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.NotificationCompat

/**
 * UI: Local / Remote / Auto prepínač + Start/Stop servera + stav downloadu.
 *
 * - Local: Ktor server na :8101, inferencia na telefóne.
 * - Remote: tenký klient na PC server (rovnaká zmluva, base URL z poľa nižšie).
 * - Auto (default): skús Local, pri 503 will-not-load / OOM fallback na Remote.
 */
enum class RunMode { LOCAL, REMOTE, AUTO }

class MainActivity : AppCompatActivity() {
    private lateinit var manager: ModelManager
    private lateinit var downloader: ModelDownloader

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        manager = AppGraph.manager
        downloader = ModelDownloader(this)

        val status = TextView(this)
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
            setSelection(RunMode.AUTO.ordinal)
        }
        val startBtn = Button(this).apply { text = "Start server" }
        val stopBtn = Button(this).apply { text = "Stop server" }
        val refreshBtn = Button(this).apply { text = "Refresh status" }

        fun renderStatus() {
            val st = manager.status()
            val models = ModelCatalog.assets.joinToString("\n") {
                "  ${it.id}: ready=${downloader.isReady(it.id)}"
            }
            status.text =
                "mode=${RunMode.values()[modeSpinner.selectedItemPosition]}\n" +
                "loaded=${st.loadedKind}/${st.loadedId} loading=${st.loading} " +
                "busy=${st.busy} queue=${st.queue}\n$models"
        }

        startBtn.setOnClickListener {
            val mode = RunMode.values()[modeSpinner.selectedItemPosition]
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
        refreshBtn.setOnClickListener { renderStatus() }
        renderStatus()

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(modeSpinner)
            addView(hostInput)
            addView(startBtn)
            addView(stopBtn)
            addView(refreshBtn)
            addView(status)
        }
        setContentView(layout)
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
        @Suppress("UNUSED_VARIABLE")
        val mode = intent?.getStringExtra("mode") ?: "AUTO"
        @Suppress("UNUSED_VARIABLE")
        val remote = intent?.getStringExtra("remote") ?: ""
        // TODO(AUTO): pri will-not-load prepnúť klienta na `remote` base URL.
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
