package eu.odysseus.diffusion

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.text.InputType
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import android.widget.EditText
import android.widget.ImageView
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
import java.io.DataOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
* UI: status card (server running/stopped + detail), model manager,
* models grouped by kind (transcription / image / upscale /
* VLM / custom) with collapsible sections and real progress, activity log.
 *
* Downloads are handled by ModelDownloader (own engine, resume, multi-file
* models) — not DownloadManager, which cannot handle LCM DreamShaper (7 files).
 */
class MainActivity : AppCompatActivity() {
    private lateinit var manager: ModelManager
    private lateinit var downloader: ModelDownloader

    private lateinit var statusDot: View
    private lateinit var statusTitle: TextView
    private lateinit var statusDetail: TextView
    private lateinit var btnToggle: MaterialButton
    private lateinit var btnAddModel: MaterialButton
    private lateinit var modelsContainer: LinearLayout
    private lateinit var settingsInfo: TextView
    private lateinit var btnOutputs: android.widget.Button

    // action UI
    private lateinit var genPrompt: EditText
    private lateinit var genResolution: android.widget.Spinner
    private lateinit var genCustomSize: EditText
    private lateinit var genQuality: android.widget.Spinner
    private lateinit var genSteps: EditText
    private lateinit var genNegative: EditText
    private lateinit var genGuidance: EditText
    private lateinit var genStatus: TextView
    private lateinit var genImage: android.widget.ImageView
    private lateinit var btnGenerate: MaterialButton
    private lateinit var btnPickAudio: MaterialButton
    private lateinit var btnRecordAudio: MaterialButton
    private lateinit var audioFileLabel: TextView
    private lateinit var spinnerModel: android.widget.Spinner
    private lateinit var spinnerLang: android.widget.Spinner
    private lateinit var switchGemma: androidx.appcompat.widget.SwitchCompat
    private lateinit var spinnerGemmaMode: android.widget.Spinner
    private lateinit var btnTranscribe: MaterialButton
    private lateinit var transcribeOut: android.widget.EditText
    private lateinit var transcribeProgress: android.widget.ProgressBar
    private lateinit var transcribeStatus: TextView
    private lateinit var btnClearTranscript: MaterialButton
    private lateinit var btnPickImage: MaterialButton
    private lateinit var imageFileLabel: TextView
    private lateinit var btnUpscale: MaterialButton
    private lateinit var btnCaption: MaterialButton
    private lateinit var visionOut: TextView
    private lateinit var btnCopyLog: TextView
    private lateinit var btnSaveImage: MaterialButton
    private lateinit var btnSaveTranscript: MaterialButton

    private var inpaintUri: android.net.Uri? = null
    private var maskUri: android.net.Uri? = null
    private var lastInpaintBitmap: android.graphics.Bitmap? = null
    private var lastBitmap: android.graphics.Bitmap? = null
    private var lastTranscript: String = ""

    // navigation + screen refs
    private lateinit var tabBar: android.widget.LinearLayout
    // tabs
    private lateinit var screenList: List<View>
    private val tabButtons = mutableListOf<MaterialButton>()
    private var currentTab = 0
    private lateinit var swipeDetector: android.view.GestureDetector

    private lateinit var screenHost: android.widget.FrameLayout
    private lateinit var screenServer: View
    private lateinit var screenGenerate: View
    private lateinit var screenTranscribe: View
    private lateinit var screenUpscale: View
    private lateinit var screenCaption: View


    private lateinit var spinnerTheme: android.widget.Spinner
    private lateinit var spinnerLanguage: android.widget.Spinner
    private lateinit var spinnerAcceleration: android.widget.Spinner
    private var settingsReady = false
    private lateinit var screenInpaint: View
    private lateinit var btnPickInpaint: MaterialButton
    private lateinit var inpaintFileLabel: TextView
    private lateinit var btnPickMask: MaterialButton
    private lateinit var maskFileLabel: TextView
    private lateinit var spinnerPaintMode: android.widget.Spinner
    private lateinit var paintColor: EditText
    private lateinit var paintPrompt: EditText
    private lateinit var paintNegative: EditText
    private lateinit var paintGuidance: EditText
    private lateinit var paintStrength: android.widget.SeekBar
    private lateinit var btnInpaint: MaterialButton
    private lateinit var paintStatus: TextView
    private lateinit var paintImage: android.widget.ImageView
    private lateinit var btnSaveInpaint: MaterialButton
    private lateinit var screenTools: View
    private lateinit var captionOut: TextView
    private lateinit var captionImageLabel: TextView
    private lateinit var btnPickImageCaption: MaterialButton
    private lateinit var toolsInfo: TextView

    private lateinit var activityLog: TextView
    private lateinit var btnClearLog: TextView

    private val logLines = ArrayDeque<String>()
    private val handler = Handler(Looper.getMainLooper())
    private var serverRunning = false

    // id modelu -> (progressBar, actionButton, meta)
    private val modelViews = mutableMapOf<String, Triple<ProgressBar, MaterialButton, TextView>>()
    private val finishedNotified = mutableSetOf<String>()

// UI state of sections: closed by default, open state survives re-render
    private val expandedKinds = mutableSetOf<String>()

    private val ticker = object : Runnable {
        override fun run() {
            refreshServerState()
            refreshDownloads()
            handler.postDelayed(this, 1500)
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
        btnAddModel = findViewById(R.id.btnAddModel)
        modelsContainer = findViewById(R.id.modelsContainer)
        settingsInfo = findViewById(R.id.settingsInfo)
        btnOutputs = findViewById(R.id.btnOutputs)
        btnOutputs.setOnClickListener { showOutputsDialog() }
        spinnerTheme = findViewById(R.id.spinnerTheme)
        spinnerLanguage = findViewById(R.id.spinnerLanguage)
        spinnerAcceleration = findViewById(R.id.spinnerAcceleration)


        genPrompt = findViewById(R.id.genPrompt)
        genResolution = findViewById(R.id.genResolution)
        genCustomSize = findViewById(R.id.genCustomSize)
        genQuality = findViewById(R.id.genQuality)
        genSteps = findViewById(R.id.genSteps)
        genNegative = findViewById(R.id.genNegative)
        genGuidance = findViewById(R.id.genGuidance)
        genStatus = findViewById(R.id.genStatus)
        genImage = findViewById(R.id.genImage)
        btnGenerate = findViewById(R.id.btnGenerate)
        btnPickAudio = findViewById(R.id.btnPickAudio)
        btnRecordAudio = findViewById(R.id.btnRecordAudio)
        audioFileLabel = findViewById(R.id.audioFileLabel)
        spinnerModel = findViewById(R.id.spinnerModel)
        spinnerLang = findViewById(R.id.spinnerLang)
        switchGemma = findViewById(R.id.switchGemma)
        spinnerGemmaMode = findViewById(R.id.spinnerGemmaMode)
        btnTranscribe = findViewById(R.id.btnTranscribe)
        transcribeOut = findViewById(R.id.transcribeOut)
        transcribeProgress = findViewById(R.id.transcribeProgress)
        transcribeStatus = findViewById(R.id.transcribeStatus)
        btnClearTranscript = findViewById(R.id.btnClearTranscript)
        btnPickImage = findViewById(R.id.btnPickImage)
        imageFileLabel = findViewById(R.id.imageFileLabel)
        btnUpscale = findViewById(R.id.btnUpscale)
        btnCaption = findViewById(R.id.btnCaption)
        visionOut = findViewById(R.id.visionOut)
        btnCopyLog = findViewById(R.id.btnCopyLog)
        btnSaveImage = findViewById(R.id.btnSaveImage)
        btnSaveTranscript = findViewById(R.id.btnSaveTranscript)

        tabBar = findViewById(R.id.tabBar)
        screenHost = findViewById(R.id.screenHost)
        screenServer = findViewById(R.id.screenServer)
        screenGenerate = findViewById(R.id.screenGenerate)
        screenTranscribe = findViewById(R.id.screenTranscribe)
        screenUpscale = findViewById(R.id.screenUpscale)
        screenCaption = findViewById(R.id.screenCaption)
        screenTools = findViewById(R.id.screenTools)
        screenInpaint = findViewById(R.id.screenInpaint)
        btnPickInpaint = findViewById(R.id.btnPickInpaint)
        inpaintFileLabel = findViewById(R.id.inpaintFileLabel)
        btnPickMask = findViewById(R.id.btnPickMask)
        maskFileLabel = findViewById(R.id.maskFileLabel)
        spinnerPaintMode = findViewById(R.id.spinnerPaintMode)
        paintColor = findViewById(R.id.paintColor)
        paintPrompt = findViewById(R.id.paintPrompt)
        paintNegative = findViewById(R.id.paintNegative)
        paintGuidance = findViewById(R.id.paintGuidance)
        paintStrength = findViewById(R.id.paintStrength)
        btnInpaint = findViewById(R.id.btnInpaint)
        paintStatus = findViewById(R.id.paintStatus)
        paintImage = findViewById(R.id.paintImage)
        btnSaveInpaint = findViewById(R.id.btnSaveInpaint)

        captionOut = findViewById(R.id.captionOut)
        captionImageLabel = findViewById(R.id.captionImageLabel)
        btnPickImageCaption = findViewById(R.id.btnPickImageCaption)
        toolsInfo = findViewById(R.id.toolsInfo)

        activityLog = findViewById(R.id.activityLog)
        btnClearLog = findViewById(R.id.btnClearLog)

        btnToggle.setOnClickListener { toggleServer() }
        btnAddModel.setOnClickListener { showAddModelDialog() }
        btnClearLog.setOnClickListener { logLines.clear(); renderLog() }
        btnCopyLog.setOnClickListener {
            val cm = getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
            cm.setPrimaryClip(android.content.ClipData.newPlainText("Prism log", activityLog.text))
            toast("Log copied")
        }
        btnSaveImage.setOnClickListener {
            val bmp = lastBitmap
            if (bmp == null) { toast("No image"); return@setOnClickListener }
            if (saveBitmapToGallery(bmp)) toast("Saved to Pictures/Prism") else toast("Save failed")
        }
        btnSaveTranscript.setOnClickListener {
            if (lastTranscript.isBlank()) { toast("No transcript"); return@setOnClickListener }
            if (saveTextToDownloads(lastTranscript)) toast("Saved to Download/Prism") else toast("Save failed")
        }


        renderModels()
        initActions()

        swipeDetector = android.view.GestureDetector(this, object : android.view.GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: android.view.MotionEvent): Boolean = true
            override fun onFling(
                e1: android.view.MotionEvent?,
                e2: android.view.MotionEvent,
                velocityX: Float,
                velocityY: Float,
            ): Boolean {
                val dx = e2.x - (e1?.x ?: 0f)
                val dy = e2.y - (e1?.y ?: 0f)
                if (kotlin.math.abs(dx) > kotlin.math.abs(dy) && kotlin.math.abs(dx) > 120f) {
                    if (dx < 0) nextTab() else prevTab()
                    return true
                }
                return false
            }
        })

        val tabLabels = listOf(
            getString(R.string.tab_server), getString(R.string.tab_generate),
            getString(R.string.tab_transcribe), getString(R.string.tab_upscale),
            getString(R.string.tab_caption), getString(R.string.tab_inpaint),
            getString(R.string.tab_tools))
        screenList = listOf(
            screenServer, screenGenerate, screenTranscribe,
            screenUpscale, screenCaption, screenInpaint, screenTools,
        )
        for ((index, label) in tabLabels.withIndex()) {
            val btn = MaterialButton(this).apply {
                text = label
                textSize = 12f
                isAllCaps = false
                cornerRadius = 18
                minWidth = 0
                minimumWidth = 0
                setPadding(24, 8, 24, 8)
                layoutParams = android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { marginEnd = 8 }
                setOnClickListener { selectTab(index) }
            }
            tabBar.addView(btn)
            tabButtons.add(btn)
        }
        selectTab(0)
        // Headless start for tests / automation:
        //   adb shell am start -n eu.odysseus.diffusion/.MainActivity --ez start_server true
        if (intent?.getBooleanExtra("start_server", false) == true) {
            window.decorView.post { if (!serverRunning) toggleServer() }
        }
        tabBar.post { selectTab(0) }

        renderLog()
        settingsInfo.text = "loading..."
        log("app started")
        refreshServerState()
        handler.postDelayed(ticker, 1500)
    }

    override fun onNewIntent(intent: android.content.Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        // `am start` on an already running activity delivers here, not to onCreate.
        if (intent?.getBooleanExtra("start_server", false) == true && !serverRunning) {
            window.decorView.post { toggleServer() }
        }
    }

    /** Default output folder browser: open, share and clear what the jobs produced. */
    private fun showOutputsDialog() {
        val files = OutputStore.list(this, 200)
        val (count, bytes) = OutputStore.total(this)
        if (files.isEmpty()) {
            toast("No outputs yet")
            return
        }
        val labels = files.map {
            val kind = it.parentFile?.name ?: ""
            "$kind/${it.name}  (${Math.max(1, it.length() / 1024)} KB)"
        }.toTypedArray()
        android.app.AlertDialog.Builder(this)
            .setTitle("Outputs · $count files, ${bytes / 1048576} MB")
            .setItems(labels) { _, which -> openOutputFile(files[which]) }
            .setPositiveButton("Save all to Downloads") { _, _ ->
                Thread {
                    val n = OutputStore.saveAllToDownloads(this)
                    handler.post { toast("Saved $n file(s) to Download/Prism") }
                }.start()
            }
            .setNeutralButton("Open folder path") { _, _ ->
                toast(OutputStore.root(this).absolutePath)
                log("outputs: " + OutputStore.root(this).absolutePath)
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun openOutputFile(file: java.io.File) {
        val mime = when (file.extension.lowercase()) {
            "png" -> "image/png"
            "jpg", "jpeg" -> "image/jpeg"
            "webp" -> "image/webp"
            "mp4" -> "video/mp4"
            "m4a" -> "audio/mp4"
            "mp3" -> "audio/mpeg"
            "wav" -> "audio/wav"
            else -> "text/plain"
        }
        try {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                this, "$packageName.files", file)
            startActivity(Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, mime)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            })
        } catch (e: Exception) {
            toast("No app can open ${file.name}")
            log("open output failed: ${e.message}")
        }
    }

    override fun onDestroy() {
        if (isRecording) { try { mediaRecorder?.stop() } catch (_: Exception) { /* cleanup */ } }
        mediaRecorder?.release()
        mediaRecorder = null
        handler.removeCallbacks(ticker)
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 2001 && grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            startRecording()
        } else if (requestCode == 2001) {
            toast("Microphone permission denied")
        }
    }

    // ---------- server ----------


    private fun toggleServer() {
        if (serverRunning) {
            stopService(Intent(this, DiffusionService::class.java))
            serverRunning = false
            log("server stop requested")
            renderServerState()
        } else {
            val i = Intent(this, DiffusionService::class.java).apply {
            }
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(i)
                else startService(i)
                serverRunning = true
                log("server starting…")
            } catch (e: Throwable) {
                // Background start restrictions: report it instead of crashing.
                serverRunning = false
                log("server start blocked: ${e.message}. Open the app and tap START.")
                toast("Open the app and tap START")
            }
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
                    statusTitle.text = getString(R.string.server_running)
                    statusDetail.text = buildString {
                        append("http://127.0.0.1:8101  ·  device=$device\n")
                        append("default=$model")
                        if (loaded.isNotBlank()) append("  loaded=$loaded")
                        if (loading) append("  loading…")
                    }
                    btnToggle.text = getString(R.string.btn_stop)
                    btnToggle.setBackgroundColor(0xFFDA3633.toInt())
                    statusDot.setBackgroundResource(R.drawable.dot_running)
                    refreshSettings()
                            } else {
                    if (serverRunning) {
                        statusTitle.text = getString(R.string.server_starting)
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
            statusTitle.text = getString(R.string.server_starting)
            statusDetail.text = "waiting for http://127.0.0.1:8101/health"
            statusDot.setBackgroundResource(R.drawable.dot_busy)
            btnToggle.text = getString(R.string.btn_stop)
            btnToggle.setBackgroundColor(0xFFDA3633.toInt())
        } else {
            statusTitle.text = getString(R.string.server_stopped)
            statusDetail.text = "Press START to run the local API on port 8101"
            statusDot.setBackgroundResource(R.drawable.dot_stopped)
            btnToggle.text = getString(R.string.btn_start)
            btnToggle.setBackgroundColor(0xFF238636.toInt())
        }
    }


    private fun refreshSettings() {
        Thread {
            val json = httpGet("http://127.0.0.1:8101/v1/settings")
            val parsed = try { if (json != null) org.json.JSONObject(json) else null } catch (_: Exception) { null }
            handler.post {
                if (parsed != null) {
                    val sb = StringBuilder()
                    sb.append("version: ").append(parsed.optString("version")).append("\n")
                    sb.append("port: ").append(parsed.optInt("port")).append("\n")
                    sb.append("models_dir: ").append(parsed.optString("models_dir")).append("\n")
                    sb.append("default_audio: ").append(parsed.optString("default_audio_model")).append("\n")
                    sb.append("audio_models: ").append(parsed.optJSONArray("audio_models")?.join(", ") ?: "").append("\n")
                    val engines = parsed.optJSONObject("engines")
                    if (engines != null) {
                        sb.append("engines: torch=").append(engines.optBoolean("torch"))
                            .append(" cuda=").append(engines.optBoolean("cuda"))
                            .append(" onnx=").append(engines.optBoolean("onnxruntime"))
                            .append(" whisper=").append(engines.optBoolean("whisper_native")).append("\n")
                        sb.append("heap_free=").append(engines.optInt("java_heap_free_mb")).append("MB")
                            .append(" heap_max=").append(engines.optInt("java_heap_max_mb")).append("MB")
                    }
                    settingsInfo.text = sb.toString()
                } else {
                    settingsInfo.text = "server not running"
                }
            }
        }.start()
    }

    // ---------- models ----------

/** Models grouped by kind; sections are closed by default and animated. */
    private fun renderModels() {
        modelsContainer.removeAllViews()
        modelViews.clear()
        val assets = downloader.allAssets()
        for (kind in ModelCatalog.kinds) {
            val group = assets.filter { it.kind == kind }
            if (group.isEmpty()) continue

            val header = layoutInflater.inflate(R.layout.item_model_section, modelsContainer, false)
            val chevron = header.findViewById<ImageView>(R.id.sectionChevron)
            val title = header.findViewById<TextView>(R.id.sectionTitle)
            val count = header.findViewById<TextView>(R.id.sectionCount)
            title.text = ModelCatalog.kindLabel(kind)
            count.text = group.size.toString()

            val body = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                )
            }

            header.setOnClickListener { toggleSection(kind, chevron, body) }

            modelsContainer.addView(header)
            modelsContainer.addView(body)

            for (asset in group) {
                val row = layoutInflater.inflate(R.layout.item_model, body, false)
                val name = row.findViewById<TextView>(R.id.modelName)
                val meta = row.findViewById<TextView>(R.id.modelMeta)
                val action = row.findViewById<MaterialButton>(R.id.modelAction)
                val progress = row.findViewById<ProgressBar>(R.id.modelProgress)

                name.text = asset.label
                meta.text = "${asset.sizeMb} MB · ${asset.id}"
                action.setOnClickListener {
                    if (downloader.isReady(asset.id)) confirmUninstall(asset) else startDownload(asset)
                }

                body.addView(row)
                modelViews[asset.id] = Triple(progress, action, meta)
                renderModelRow(asset)
            }

// Restore open sections without animation (e.g. after adding a custom model).
            if (kind in expandedKinds) {
                body.visibility = View.VISIBLE
                chevron.rotation = 90f
            }
        }
    }

// ---------- collapsible sections (animated) ----------

    private fun toggleSection(kind: String, chevron: ImageView, body: LinearLayout) {
        val expanded = body.visibility == View.VISIBLE
        if (expanded) {
            collapse(body)
            expandedKinds.remove(kind)
            chevron.animate().rotation(0f).setDuration(220)
                .setInterpolator(DecelerateInterpolator()).start()
        } else {
            expand(body)
            expandedKinds.add(kind)
            chevron.animate().rotation(90f).setDuration(220)
                .setInterpolator(DecelerateInterpolator()).start()
        }
    }

    private fun expand(view: View) {
        view.visibility = View.VISIBLE
        view.measure(
            View.MeasureSpec.makeMeasureSpec(
                modelsContainer.width.coerceAtLeast(1),
                View.MeasureSpec.EXACTLY,
            ),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
        )
        val target = view.measuredHeight
        if (target <= 0) {
            view.layoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
            view.requestLayout()
            return
        }
        val lp = view.layoutParams
        lp.height = 0
        view.layoutParams = lp

        ValueAnimator.ofInt(0, target).apply {
            duration = 260
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                lp.height = it.animatedValue as Int
                view.requestLayout()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    lp.height = ViewGroup.LayoutParams.WRAP_CONTENT
                    view.requestLayout()
                }
            })
            start()
        }
    }

    private fun collapse(view: View) {
        val start = view.height.takeIf { it > 0 } ?: run {
            view.visibility = View.GONE
            return
        }
        val lp = view.layoutParams
        ValueAnimator.ofInt(start, 0).apply {
            duration = 220
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                lp.height = it.animatedValue as Int
                view.requestLayout()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    view.visibility = View.GONE
                    lp.height = ViewGroup.LayoutParams.WRAP_CONTENT
                    view.requestLayout()
                }
            })
            start()
        }
    }

    private fun renderModelRow(asset: ModelAsset) {
        val views = modelViews[asset.id] ?: return
        val (progress, action, meta) = views
        val ready = downloader.isReady(asset.id)
        val p = downloader.progress(asset.id)
        val downloading = p != null && (p.state == InstallState.QUEUED ||
            p.state == InstallState.DOWNLOADING || p.state == InstallState.VERIFYING)

        when {
            downloading -> {
                progress.visibility = ProgressBar.VISIBLE
                progress.progress = p!!.percent
                action.text = "Stop ${p.percent}%"
                action.isEnabled = true
                action.setBackgroundColor(0xFFDA3633.toInt())
                action.setTextColor(0xFFE8EAF0.toInt())
                action.setOnClickListener { downloader.cancel(asset.id) }
                meta.text = if (p.state == InstallState.VERIFYING) "verifying…"
                else "${fmtBytes(p.bytesDone)} / ${fmtBytes(p.bytesTotal)} · ${asset.engine}"
            }
            ready -> {
                val mb = downloader.installedSizeMb(asset.id)
                action.text = "Uninstall"
                action.isEnabled = true
                action.setBackgroundColor(0xFFDA3633.toInt())
                action.setTextColor(0xFFFFFFFF.toInt())
                action.setOnClickListener { confirmUninstall(asset) }
                progress.visibility = ProgressBar.GONE
                meta.text = "${asset.sizeMb} MB · ${asset.engine} · installed ($mb MB)"
            }
            p?.state == InstallState.FAILED -> {
                progress.visibility = ProgressBar.GONE
                action.text = "Retry"
                action.isEnabled = true
                action.setBackgroundColor(0xFF238636.toInt())
                action.setTextColor(0xFFFFFFFF.toInt())
                action.setOnClickListener { startDownload(asset) }
                meta.text = "${asset.sizeMb} MB · failed: ${p.error.take(70)}"
            }
            !asset.hasSource -> {
                action.text = "N/A"
                action.isEnabled = false
                progress.visibility = ProgressBar.GONE
                meta.text = "${asset.sizeMb} MB · no source"
            }
            else -> {
                action.text = if (asset.custom) "Download" else "Install"
                action.isEnabled = true
                action.setBackgroundColor(0xFF238636.toInt())
                action.setTextColor(0xFFFFFFFF.toInt())
                action.setOnClickListener { startDownload(asset) }
                progress.visibility = ProgressBar.GONE
                val badge = if (asset.runsOnDevice) asset.engine else asset.engine + " (experimental)"
                meta.text = "${asset.sizeMb} MB · $badge"
            }
        }
    }

    private fun confirmUninstall(asset: ModelAsset) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Remove ${asset.id}?")
            .setMessage("Deletes weights (${downloader.installedSizeMb(asset.id)} MB on disk).")
            .setPositiveButton("Remove") { _, _ ->
                val ok = downloader.uninstall(asset.id)
                log(if (ok) "removed ${asset.id}" else "remove ${asset.id} FAILED")
                renderModelRow(asset)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startDownload(asset: ModelAsset) {
        if (!asset.hasSource) {
            log("${asset.id}: no verified download source")
            return
        }
        try {
            downloader.enqueue(asset)
            log("downloading ${asset.id} (${asset.sizeMb} MB, ${asset.files.size} files)…")
            renderModelRow(asset)
        } catch (e: Exception) {
            log("download ${asset.id} failed: ${e.message}")
        }
    }

/** Poll progress directly from the downloader (no DownloadManager). */
    private fun refreshDownloads() {
        for ((assetId, views) in modelViews) {
            val p = downloader.progress(assetId) ?: continue
            if (p.state == InstallState.DOWNLOADING || p.state == InstallState.QUEUED ||
                p.state == InstallState.VERIFYING
            ) {
                val (progress, action, meta) = views
                progress.visibility = ProgressBar.VISIBLE
                progress.progress = p.percent
                action.text = "Stop ${p.percent}%"
                meta.text = "${fmtBytes(p.bytesDone)} / ${fmtBytes(p.bytesTotal)}"
            } else if ((p.state == InstallState.DONE || p.state == InstallState.FAILED) &&
                finishedNotified.add("$assetId:${p.state}")
            ) {
                val asset = downloader.findAsset(assetId) ?: continue
                if (p.state == InstallState.DONE) log("${assetId} READY")
                else log("${assetId} FAILED: ${p.error}")
                renderModelRow(asset)
            }
        }
    }

/** Custom model: name + direct URL + kind. */
    private fun showAddModelDialog() {
        val pad = dp(16)
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, dp(8), pad, 0)
        }
        val name = EditText(this).apply {
            hint = "name (e.g. my-model)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val url = EditText(this).apply {
            hint = "https://…/model.gguf"
            inputType = InputType.TYPE_TEXT_VARIATION_URI
        }
        val kind = EditText(this).apply {
            hint = "kind: audio / image / upscale / vlm / custom"
            inputType = InputType.TYPE_CLASS_TEXT
            setText("custom")
        }
        box.addView(name)
        box.addView(url)
        box.addView(kind)

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Add model by URL")
            .setMessage("Direct file URL (GGUF/ONNX/bin/safetensors). Saves to models/<name>/ with resume.")
            .setView(box)
            .setPositiveButton("Download") { _, _ ->
                val n = name.text.toString().trim()
                val u = url.text.toString().trim()
                val k = kind.text.toString().trim().ifBlank { "custom" }
                try {
                    val asset = downloader.addCustom(n, k, u)
                    log("added ${asset.id} (${asset.kind}) → models/${asset.id}")
                    renderModels()
                } catch (e: Exception) {
                    log("add model failed: ${e.message}")
                    toast(e.message ?: "add failed")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ---------- actions (generate / transcribe / upscale / caption) ----------

    private var audioUri: android.net.Uri? = null
    private var imageUri: android.net.Uri? = null
    private var mediaRecorder: android.media.MediaRecorder? = null
    private var isRecording = false
    private var recordingFile: java.io.File? = null

    private val pickAudio = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                audioUri = uri
                audioFileLabel.text = uri.lastPathSegment ?: uri.toString()
            }
        }
    }

    private val pickImage = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                imageUri = uri
                imageFileLabel.text = uri.lastPathSegment ?: uri.toString()
                captionImageLabel.text = uri.lastPathSegment ?: uri.toString()
            }
        }
    }


    private val pickInpaint = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                inpaintUri = uri
                inpaintFileLabel.text = uri.lastPathSegment ?: uri.toString()
            }
        }
    }

    private val pickMask = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                maskUri = uri
                maskFileLabel.text = uri.lastPathSegment ?: uri.toString()
            }
        }
    }

    private fun openDocument(mime: String, initialPath: String, launcher: androidx.activity.result.ActivityResultLauncher<android.content.Intent>) {
        val intent = android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(android.content.Intent.CATEGORY_OPENABLE)
            type = mime
            putExtra(
                "android.provider.extra.INITIAL_URI",
                android.net.Uri.parse("content://com.android.externalstorage.documents/document/primary%3A" + android.net.Uri.encode(initialPath)),
            )
        }
        launcher.launch(intent)
    }

    private fun initActions() {
        // model spinner: audio models only
        val audio = downloader.allAssets().filter { it.kind == "audio" }
        spinnerModel.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, audio.map { it.id })
        spinnerLang.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("sk", "cs", "en", "de", "fr", "es", "it", "pl", "ru", "uk", "hu", "auto"))
        spinnerGemmaMode.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("cleanup", "translate", "summarize", "rewrite"))

        genResolution.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("256x256", "512x512", "768x768", "custom"))
        genResolution.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                genCustomSize.visibility = if (genResolution.selectedItem.toString() == "custom") View.VISIBLE else View.GONE
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        genQuality.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, listOf("low", "medium", "high"))
        spinnerPaintMode.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("add", "remove", "color", "outpaint (25%)"))

        btnPickInpaint.setOnClickListener { openDocument("image/*", "Pictures", pickInpaint) }
        btnPickMask.setOnClickListener { openDocument("image/*", "Pictures", pickMask) }
        btnInpaint.setOnClickListener { runInpaint() }
        btnSaveInpaint.setOnClickListener {
            val bmp = lastInpaintBitmap
            if (bmp == null) toast("No image") else if (saveBitmapToGallery(bmp)) toast("Saved to Pictures/Prism") else toast("Save failed")
        }


        // ---------- settings: theme / language / acceleration ----------
        val prefs = getSharedPreferences("settings", android.content.Context.MODE_PRIVATE)
        val theme = prefs.getString("theme", "system") ?: "system"
        val lang = prefs.getString("language", "en") ?: "en"
        val accel = prefs.getString("acceleration", "auto") ?: "auto"
        AppConfig.acceleration = accel

        val themeModes = listOf("system" to "System", "light" to "Light", "dark" to "Dark")
        val langs = listOf("en" to "English", "sk" to "Slovak", "cs" to "Czech", "de" to "German")
        val accels = listOf("auto" to "Auto", "cpu" to "CPU", "gpu" to "GPU (NNAPI)")

        spinnerTheme.adapter = android.widget.ArrayAdapter(this,
            android.R.layout.simple_spinner_dropdown_item, themeModes.map { it.second })
        spinnerLanguage.adapter = android.widget.ArrayAdapter(this,
            android.R.layout.simple_spinner_dropdown_item, langs.map { it.second })
        spinnerAcceleration.adapter = android.widget.ArrayAdapter(this,
            android.R.layout.simple_spinner_dropdown_item, accels.map { it.second })

        spinnerTheme.setSelection(themeModes.indexOfFirst { it.first == theme }.coerceAtLeast(0))
        spinnerLanguage.setSelection(langs.indexOfFirst { it.first == lang }.coerceAtLeast(0))
        spinnerAcceleration.setSelection(accels.indexOfFirst { it.first == accel }.coerceAtLeast(0))

        fun applyTheme(mode: String) {
            val m = when (mode) {
                "light" -> androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO
                "dark" -> androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES
                else -> androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
            }
            if (androidx.appcompat.app.AppCompatDelegate.getDefaultNightMode() != m) {
                androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(m)
            }
        }
        fun applyLanguage(tag: String) {
            val current = androidx.appcompat.app.AppCompatDelegate.getApplicationLocales().toLanguageTags()
            if (current != tag && !current.startsWith(tag + "-")) {
                androidx.appcompat.app.AppCompatDelegate.setApplicationLocales(
                    androidx.core.os.LocaleListCompat.forLanguageTags(tag))
            }
        }
        // apply on startup (before the user touches the spinners)
        applyTheme(theme)
        applyLanguage(lang)

        spinnerTheme.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!settingsReady) return
                val mode = themeModes[position].first
                prefs.edit().putString("theme", mode).apply()
                applyTheme(mode)
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        spinnerLanguage.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!settingsReady) return
                val tag = langs[position].first
                prefs.edit().putString("language", tag).apply()
                applyLanguage(tag)
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        spinnerAcceleration.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (!settingsReady) return
                val mode = accels[position].first
                prefs.edit().putString("acceleration", mode).apply()
                AppConfig.acceleration = mode
                LcmOnnxEngine.release()
                log("acceleration -> $mode (model will reload on next request)")
            }
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }
        settingsReady = true

        btnPickAudio.setOnClickListener { openDocument("*/*", "Recordings", pickAudio) }
        btnRecordAudio.setOnClickListener { toggleRecording() }
        btnPickImageCaption.setOnClickListener { openDocument("image/*", "Pictures", pickImage) }
        btnPickImage.setOnClickListener { openDocument("image/*", "Pictures", pickImage) }

        btnGenerate.setOnClickListener { runGenerate() }
        btnTranscribe.setOnClickListener { runTranscribe() }
        btnClearTranscript.setOnClickListener {
            transcribeOut.setText("")
            transcribeStatus.text = ""
            btnSaveTranscript.visibility = android.view.View.GONE
        }
        btnUpscale.setOnClickListener { runUpscale() }
        btnCaption.setOnClickListener { runCaption() }
    }

    private fun runGenerate() {
        val prompt = genPrompt.text.toString().trim()
        if (prompt.isEmpty()) { toast("Enter a prompt"); return }
        val size = if (genResolution.selectedItem.toString() == "custom")
            genCustomSize.text.toString().trim().ifEmpty { "512x512" }
        else genResolution.selectedItem.toString()
        val steps = genSteps.text.toString().trim().toIntOrNull() ?: 6
        val negative = genNegative.text.toString().trim()
        val guidance = genGuidance.text.toString().trim().toFloatOrNull() ?: 1.0f
        val quality = genQuality.selectedItem?.toString() ?: "medium"
        genStatus.text = "generating…"
        btnGenerate.isEnabled = false
        Thread {
            val body = org.json.JSONObject()
                .put("prompt", prompt).put("negative_prompt", negative)
                .put("guidance", guidance).put("quality", quality)
                .put("size", size).put("n", 1).put("steps", steps)
                .toString()
            val resp = httpPostJson("http://127.0.0.1:8101/v1/images/generations", body)
            handler.post {
                btnGenerate.isEnabled = true
                if (resp == null) { genStatus.text = "error: no response"; return@post }
                try {
                    val j = org.json.JSONObject(resp)
                    val b64 = j.optJSONArray("data")?.optJSONObject(0)?.optString("b64_json") ?: ""
                    if (b64.isEmpty()) {
                        genStatus.text = "error: " + j.optString("error", "no image")
                        return@post
                    }
                    val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                    val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    lastBitmap = bmp
                    genImage.setImageBitmap(bmp)
                    genImage.visibility = View.VISIBLE
                    btnSaveImage.visibility = View.VISIBLE
                    genStatus.text = "done (${size}, ${steps} steps)"
                } catch (e: Exception) {
                    genStatus.text = "error: ${e.message}"
                }
            }
        }.start()
    }

    private fun runTranscribe() {
        val uri = audioUri ?: run { toast("Pick or record an audio file first"); return }
        val model = spinnerModel.selectedItem?.toString() ?: ""
        val lang = spinnerLang.selectedItem?.toString() ?: "sk"
        transcribeOut.setText("")
        transcribeStatus.text = "transcribing…"
        transcribeProgress.visibility = android.view.View.VISIBLE
        btnTranscribe.isEnabled = false
        btnSaveTranscript.visibility = android.view.View.GONE
        Thread {
            val bytes = if (uri.scheme == "file") {
                java.io.File(uri.path!!).readBytes()
            } else {
                contentResolver.openInputStream(uri)?.use { it.readBytes() }
            }
            if (bytes == null) {
                handler.post {
                    btnTranscribe.isEnabled = true
                    transcribeProgress.visibility = android.view.View.GONE
                    transcribeStatus.text = "error: cannot read file"
                }
                return@Thread
            }
            val name = uri.lastPathSegment ?: "audio.wav"
            handler.post { transcribeStatus.text = "uploading ${bytes.size / 1024} KB…" }
            val resp = httpMultipart(
                "http://127.0.0.1:8101/v1/audio/transcriptions",
                mapOf("model" to model, "language" to lang),
                "file", name, "application/octet-stream", bytes,
            )
            handler.post {
                btnTranscribe.isEnabled = true
                transcribeProgress.visibility = android.view.View.GONE
                if (resp == null) {
                    transcribeStatus.text = "error: no response"
                    return@post
                }
                try {
                    val j = org.json.JSONObject(resp)
                    val text = j.optString("text", "")
                    if (text.isNotEmpty()) {
                        var finalText = text
                        if (switchGemma.isChecked) {
                            transcribeStatus.text = "post-processing with Gemma4..."
                            transcribeProgress.visibility = android.view.View.VISIBLE
                            try {
                                val modelPath = findGemmaModelPath()
                                if (modelPath != null) {
                                    val handle = GemmaLib.loadTextModel(modelPath)
                                    if (handle != 0L) {
                                        val mode = spinnerGemmaMode.selectedItem?.toString() ?: "cleanup"
                                        val targetLang = spinnerLang.selectedItem?.toString() ?: "sk"
                                        val prompt = when (mode) {
                                            "cleanup" -> "Fix spelling, grammar, and punctuation in this transcription. Keep the original language. Return only the corrected text:\\n\\n$text"
                                            "translate" -> "Translate the following text to ${langName(targetLang)}. Return only the translation:\\n\\n$text"
                                            "summarize" -> "Summarize the following text in a few sentences. Return only the summary:\\n\\n$text"
                                            "rewrite" -> "Rewrite the following text to be clearer and more readable. Return only the rewritten text:\\n\\n$text"
                                            else -> text
                                        }
                                        val result = GemmaLib.generateText(handle, prompt, 2048)
                                        if (result != null && result.isNotBlank()) {
                                            finalText = result.trim()
                                        }
                                        GemmaLib.releaseTextModel(handle)
                                    }
                                } else {
                                    transcribeStatus.text = "Gemma4 model not downloaded"
                                }
                            } catch (e: Exception) {
                                transcribeStatus.text = "Gemma4 error: ${e.message}"
                            }
                            transcribeProgress.visibility = android.view.View.GONE
                        }
                        transcribeOut.setText(finalText)
                        transcribeStatus.text = "done: ${finalText.length} chars, ${j.optInt("chunks", 1)} chunk(s)" +
                            if (switchGemma.isChecked) " (Gemma4)" else ""
                        btnSaveTranscript.visibility = android.view.View.VISIBLE
                    } else {
                        transcribeStatus.text = "error: " + j.optString("error", resp)
                    }
                } catch (e: Exception) {
                    transcribeStatus.text = "error: ${e.message}"
                }
            }
        }.start()
    }

    private fun toggleRecording() {
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
            != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), 2001)
            return
        }
        try {
            val dir = cacheDir
            recordingFile = java.io.File(dir, "recording_${System.currentTimeMillis()}.m4a")
            mediaRecorder = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                android.media.MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                android.media.MediaRecorder()
            }
            mediaRecorder!!.apply {
                setAudioSource(android.media.MediaRecorder.AudioSource.MIC)
                setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC)
                setAudioEncodingBitRate(128000)
                setAudioSamplingRate(44100)
                setOutputFile(recordingFile!!.absolutePath)
                prepare()
                start()
            }
            isRecording = true
            btnRecordAudio.text = "Stop"
            audioFileLabel.text = "recording..."
            toast("Recording started")
        } catch (e: Exception) {
            toast("record failed: ${e.message}")
            mediaRecorder?.release()
            mediaRecorder = null
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply { stop(); release() }
        } catch (e: Exception) {
            toast("stop failed: ${e.message}")
        }
        mediaRecorder = null
        isRecording = false
        btnRecordAudio.text = "Record"
        recordingFile?.let { f ->
            if (f.exists() && f.length() > 0) {
                audioUri = android.net.Uri.fromFile(f)
                audioFileLabel.text = "recorded: ${f.name} (${f.length() / 1024} KB)"
                toast("Recording saved, ready to transcribe")
            } else {
                toast("recording empty")
            }
        }
    }

    private fun runUpscale() {
        val uri = imageUri ?: run { toast("Pick an image first"); return }
        visionOut.text = "upscaling…"
        btnUpscale.isEnabled = false
        Thread {
            val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes == null) {
                handler.post { btnUpscale.isEnabled = true; visionOut.text = "error: cannot read file" }
                return@Thread
            }
            val b64 = android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
            val body = org.json.JSONObject().put("image", b64).put("scale", 2).toString()
            val resp = httpPostJson("http://127.0.0.1:8101/v1/images/upscale", body)
            handler.post {
                btnUpscale.isEnabled = true
                if (resp == null) { captionOut.text = "error: no response"; return@post }
                try {
                    val j = org.json.JSONObject(resp)
                    val out = j.optString("image", "")
                    if (out.isEmpty()) { visionOut.text = "error: " + j.optString("error", resp); return@post }
                    val outBytes = android.util.Base64.decode(out, android.util.Base64.DEFAULT)
                    val bmp = android.graphics.BitmapFactory.decodeByteArray(outBytes, 0, outBytes.size)
                    lastBitmap = bmp
                    genImage.setImageBitmap(bmp)
                    genImage.visibility = View.VISIBLE
                    btnSaveImage.visibility = View.VISIBLE
                    visionOut.text = "upscaled 2x"
                } catch (e: Exception) {
                    visionOut.text = "error: ${e.message}"
                }
            }
        }.start()
    }

    private fun runCaption() {
        val uri = imageUri ?: run { toast("Pick an image first"); return }
        captionOut.text = "captioning…"
        btnCaption.isEnabled = false
        Thread {
            val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes == null) {
                handler.post { btnCaption.isEnabled = true; captionOut.text = "error: cannot read file" }
                return@Thread
            }
            val b64 = android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
            val body = org.json.JSONObject().put("image", b64).toString()
            val resp = httpPostJson("http://127.0.0.1:8101/v1/images/caption", body)
            handler.post {
                btnCaption.isEnabled = true
                if (resp == null) { visionOut.text = "error: no response"; return@post }
                try {
                    val j = org.json.JSONObject(resp)
                    val caption = j.optString("caption", j.optString("text", ""))
                    captionOut.text = if (caption.isNotEmpty()) caption
                    else "error: " + j.optString("error", resp)
                } catch (e: Exception) {
                    captionOut.text = "error: ${e.message}"
                }
            }
        }.start()
    }


    private fun saveBitmapToGallery(bmp: android.graphics.Bitmap): Boolean {
        return try {
            val name = "prism_${System.currentTimeMillis()}.png"
            val values = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Prism")
            }
            val uri = contentResolver.insert(
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
            ) ?: return false
            contentResolver.openOutputStream(uri)?.use {
                bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
            } ?: return false
            true
        } catch (e: Exception) {
            log("save image failed: ${e.message}")
            false
        }
    }

    private fun saveTextToDownloads(text: String): Boolean {
        return try {
            val name = "prism_${System.currentTimeMillis()}.txt"
            val values = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Downloads.DISPLAY_NAME, name)
                put(android.provider.MediaStore.Downloads.MIME_TYPE, "text/plain")
                put(android.provider.MediaStore.Downloads.RELATIVE_PATH, "Download/Prism")
            }
            val uri = contentResolver.insert(
                android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values
            ) ?: return false
            contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray()) } ?: return false
            true
        } catch (e: Exception) {
            log("save transcript failed: ${e.message}")
            false
        }
    }


    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // LMKD killed the process for low memory during a transcription run.
        // Drop the heavy native engines as soon as the system asks for RAM.
        if (level >= android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            try {
                LcmOnnxEngine.release()
                RealEsrganNcnnEngine.release()
                log("memory pressure ($level): released image/upscale engines")
            } catch (e: Exception) {
                log("trim release failed: ${e.message}")
            }
        }
        if (level >= android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
            try {
                SmolVlmEngine.release()
                WhisperLib.release()
            } catch (e: Exception) {
                log("critical trim release failed: ${e.message}")
            }
        }
    }

    private fun runInpaint() {
        val img = inpaintUri ?: run { toast("Pick an image first"); return }
        val rawMode = spinnerPaintMode.selectedItem?.toString() ?: "add"
        val outpaint = rawMode.startsWith("outpaint")
        // Outpaint expands the canvas, so it needs no mask and no strength.
        val mask = if (outpaint) null else maskUri ?: run { toast("Pick a mask first"); return }
        val mode = if (outpaint) "add" else rawMode
        val prompt = paintPrompt.text.toString().trim()
        if (mode != "color" && prompt.isEmpty()) { toast("Enter a prompt"); return }
        paintStatus.text = "inpainting…"
        btnInpaint.isEnabled = false
        Thread {
            val imgB64 = contentResolver.openInputStream(img)?.use { it.readBytes() }?.let {
                android.util.Base64.encodeToString(it, android.util.Base64.NO_WRAP)
            }
            val maskB64 = mask?.let { uri ->
                contentResolver.openInputStream(uri)?.use { it.readBytes() }?.let {
                    android.util.Base64.encodeToString(it, android.util.Base64.NO_WRAP)
                }
            }
            if (imgB64 == null || (!outpaint && maskB64 == null)) {
                handler.post { btnInpaint.isEnabled = true; paintStatus.text = "error: cannot read files" }
                return@Thread
            }
            val body = org.json.JSONObject().put("image", imgB64)
            val endpoint = if (outpaint) {
                body.put("prompt", prompt)
                    .put("negative_prompt", paintNegative.text.toString().trim())
                    .put("guidance", paintGuidance.text.toString().trim().toFloatOrNull() ?: 1.0f)
                    .put("margin", 0.25)
                    .put("steps", 6)
                "http://127.0.0.1:8101/v1/images/outpaint"
            } else {
                body.put("mask", maskB64)
                    .put("prompt", prompt)
                    .put("negative_prompt", paintNegative.text.toString().trim())
                    .put("guidance", paintGuidance.text.toString().trim().toFloatOrNull() ?: 1.0f)
                    .put("strength", paintStrength.progress / 100.0)
                    .put("mode", mode)
                    .put("color", paintColor.text.toString().trim().ifEmpty { "#0066FF" })
                    .put("width", 512).put("height", 512)
                "http://127.0.0.1:8101/v1/images/inpaint"
            }
            val resp = httpPostJson(endpoint, body.toString())
            handler.post {
                btnInpaint.isEnabled = true
                if (resp == null) { paintStatus.text = "error: no response"; return@post }
                try {
                    val j = org.json.JSONObject(resp)
                    val b64 = j.optString("image", "")
                    if (b64.isEmpty()) { paintStatus.text = "error: " + j.optString("error", resp); return@post }
                    val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                    val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    lastInpaintBitmap = bmp
                    paintImage.setImageBitmap(bmp)
                    paintImage.visibility = View.VISIBLE
                    btnSaveInpaint.visibility = View.VISIBLE
                    val outSize = j.optJSONArray("output_size")
                    val sizeTxt = if (outSize != null) {
                        " " + outSize.optInt(0) + "x" + outSize.optInt(1)
                    } else ""
                    paintStatus.text = "done" + sizeTxt + " (" + j.optString("engine", "onnx") +
                        ", " + j.optDouble("elapsed", 0.0) + "s)"
                } catch (e: Exception) {
                    paintStatus.text = "error: ${e.message}"
                }
            }
        }.start()
    }

    private fun httpPostJson(url: String, body: String): String? {
        return try {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.connectTimeout = 10_000
            conn.readTimeout = 600_000
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.use { it.write(body.toByteArray()) }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val resp = stream?.bufferedReader()?.use { it.readText() }
            conn.disconnect()
            resp
        } catch (e: Exception) {
            log("HTTP error: ${e.message}")
            null
        }
    }

    private fun httpMultipart(
        url: String,
        fields: Map<String, String>,
        fileField: String,
        fileName: String,
        mime: String,
        fileBytes: ByteArray,
    ): String? {
        return try {
            val boundary = "----PrismBoundary" + System.currentTimeMillis()
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.connectTimeout = 10_000
            conn.readTimeout = 600_000
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            DataOutputStream(conn.outputStream).use { out ->
                for ((k, v) in fields) {
                    out.writeBytes("--$boundary\r\n")
                    out.writeBytes("Content-Disposition: form-data; name=\"$k\"\r\n\r\n")
                    out.writeBytes("$v\r\n")
                }
                out.writeBytes("--$boundary\r\n")
                out.writeBytes("Content-Disposition: form-data; name=\"$fileField\"; filename=\"$fileName\"\r\n")
                out.writeBytes("Content-Type: $mime\r\n\r\n")
                out.write(fileBytes)
                out.writeBytes("\r\n--$boundary--\r\n")
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val resp = stream?.bufferedReader()?.use { it.readText() }
            conn.disconnect()
            resp
        } catch (e: Exception) {
            log("HTTP error: ${e.message}")
            null
        }
    }


    private fun selectTab(index: Int, animate: Boolean = true) {
        val safe = index.coerceIn(0, screenList.lastIndex)
        val old = screenList.getOrNull(currentTab)
        val next = screenList[safe]
        val forward = safe > currentTab

        if (animate && old != null && old !== next) {
            val width = screenHost.width.coerceAtLeast(1).toFloat()
            val dir = if (forward) 1f else -1f
            next.visibility = View.VISIBLE
            next.translationX = dir * width
            next.alpha = 0.85f
            next.animate()
                .translationX(0f).alpha(1f)
                .setDuration(260)
                .setInterpolator(android.view.animation.DecelerateInterpolator())
                .start()
            old.animate()
                .translationX(-dir * width * 0.28f).alpha(0f)
                .setDuration(220)
                .setInterpolator(android.view.animation.DecelerateInterpolator())
                .withEndAction {
                    old.visibility = View.GONE
                    old.translationX = 0f
                    old.alpha = 1f
                }
                .start()
        } else {
            for ((i, v) in screenList.withIndex()) {
                v.visibility = if (i == safe) View.VISIBLE else View.GONE
                v.translationX = 0f
                v.alpha = 1f
            }
        }

        currentTab = safe
        for ((i, b) in tabButtons.withIndex()) {
            val selected = i == safe
            val bg = if (selected) 0xFF1F6FEB.toInt() else 0xFF30363D.toInt()
            b.backgroundTintList = android.content.res.ColorStateList.valueOf(bg)
            b.setTextColor(if (selected) 0xFFFFFFFF.toInt() else 0xFF9FB3C8.toInt())
            b.animate().scaleX(if (selected) 1.05f else 1f).scaleY(if (selected) 1.05f else 1f)
                .setDuration(180).start()
        }
        tabButtons.getOrNull(safe)?.let { b ->
            tabBar.post {
                val parentWidth = (tabBar.parent as? View)?.width ?: 0
                val target = (b.left + b.width / 2 - parentWidth / 2).coerceAtLeast(0)
                (tabBar.parent as? android.widget.HorizontalScrollView)?.smoothScrollTo(target, 0)
            }
        }
    }

    private fun nextTab() { if (currentTab < screenList.lastIndex) selectTab(currentTab + 1) }
    private fun prevTab() { if (currentTab > 0) selectTab(currentTab - 1) }

    override fun dispatchTouchEvent(ev: android.view.MotionEvent?): Boolean {
        if (ev != null && ::swipeDetector.isInitialized) swipeDetector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
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

    private fun findGemmaModelPath(): String? {
        val modelsDir = java.io.File(filesDir, "models/gemma4-e2b-gguf")
        val gguf = modelsDir.listFiles()?.firstOrNull { it.name.endsWith(".gguf") }
        return gguf?.absolutePath
    }

    private fun langName(code: String): String = when (code) {
        "sk" -> "Slovak"; "cs" -> "Czech"; "en" -> "English"; "de" -> "German"
        "fr" -> "French"; "es" -> "Spanish"; "it" -> "Italian"; "pl" -> "Polish"
        "ru" -> "Russian"; "uk" -> "Ukrainian"; "hu" -> "Hungarian"; else -> code
    }

    private fun toast(m: String) = Toast.makeText(this, m, Toast.LENGTH_SHORT).show()

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

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

/** Process graph — one ModelManager for the whole app (single-resident). */
object AppGraph {
    val manager = ModelManager()
}

/**
* ForegroundService keeping the Ktor server alive.
* The notification shows the real state (port + mode).
 */
class DiffusionService : Service() {
    private var server: ApiServer? = null

    override fun onCreate() {
        super.onCreate()
        val ch = NotificationChannel("srv", "Diffusion server", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(ch)
        try {
            startForeground(1, buildNotification("Server running on :8101"))
        } catch (e: Throwable) {
            // Android 12+ forbids promoting a service to foreground when the app
            // was started from the background (adb, widgets, automation):
            //   ForegroundServiceStartNotAllowedException
            // The server is still useful without the notification, so keep
            // running instead of taking the process down with it.
            android.util.Log.w("DiffusionService", "startForeground not allowed: ${e.message}")
        }
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, "srv")
            .setContentTitle("Prism")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (server == null) {
            server = ApiServer(AppGraph.manager, ModelDownloader(this), applicationContext).also { it.start(wait = false) }
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .notify(1, buildNotification("Server running on :8101"))
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
