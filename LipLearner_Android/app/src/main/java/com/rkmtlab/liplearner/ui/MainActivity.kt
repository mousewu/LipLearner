package com.rkmtlab.liplearner.ui

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.InputType
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.widget.EditText
import android.widget.PopupMenu
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.rkmtlab.liplearner.camera.CameraController
import com.rkmtlab.liplearner.databinding.ActivityMainBinding
import com.rkmtlab.liplearner.exec.CommandExecutor
import com.rkmtlab.liplearner.kws.LipRecognitionController
import com.rkmtlab.liplearner.ml.CommandStore
import com.rkmtlab.liplearner.ml.FreeVsrRecognizer
import com.rkmtlab.liplearner.ml.LipEncoder
import com.rkmtlab.liplearner.ml.ModelRegistry
import com.rkmtlab.liplearner.ml.ModelSpec
import com.rkmtlab.liplearner.speech.SpeechRecognizerHelper
import com.rkmtlab.liplearner.vision.LipLandmarker
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), LipRecognitionController.Listener {

    private lateinit var binding: ActivityMainBinding

    private lateinit var landmarker: LipLandmarker
    private lateinit var executor: CommandExecutor
    private lateinit var speech: SpeechRecognizerHelper
    private lateinit var camera: CameraController

    // Rebuilt whenever the active model changes.
    private lateinit var encoder: LipEncoder
    private lateinit var store: CommandStore
    private lateinit var controller: LipRecognitionController
    private var activeModel: ModelSpec? = null

    private val bg = Executors.newSingleThreadExecutor()
    private var userName = "User1"
    @Volatile private var modelReady = false

    // --- Free VSR (open-vocabulary, no registration) ---
    private var freeVsr: FreeVsrRecognizer? = null
    private var freeVsrLang = FreeVsrRecognizer.Lang.EN
    @Volatile private var freeVsrMode = false
    @Volatile private var freeVsrRecording = false
    private val freeVsrFrames = ArrayList<FloatArray>(400)
    // Raw face crops + keypoints of the utterance, aligned in one pass on release so the live path
    // gets the same centered-window smoothing as the (measurably better) video-file path.
    private val freeVsrRawFrames = ArrayList<android.graphics.Bitmap>(400)
    private val freeVsrRawKps = ArrayList<com.rkmtlab.liplearner.vision.FaceAligner.Keypoints>(400)
    // Wall-clock time of each captured frame. The analysis stream delivers ~15fps (MediaPipe +
    // alignment are expensive and CameraX drops frames), but the model was trained on 25fps video —
    // feeding the raw sequence makes speech look ~2x too fast. We resample by timestamp instead.
    private val freeVsrTimes = ArrayList<Long>(400)

    private var pendingCommandVector: FloatArray? = null
    private var pendingKeywordCandidate: FloatArray? = null
    private var voiceLabel: String? = null // best-effort Voice2Lip suggestion (may be null)

    private val languageTags = listOf("en-US", "ja-JP", "zh-CN", "ms-MY", "fr-FR", "es-419", "vi-VN")
    private val languageNames = listOf("English", "日本語", "中文", "Melayu", "français", "español", "Tiếng Việt")
    private var languageIndex = 0

    private val recentFreeUse = mutableListOf<String>()

    private val prefs by lazy { getSharedPreferences("app_prefs", Context.MODE_PRIVATE) }

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            if (result[Manifest.permission.CAMERA] == true) initEverything()
            else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }

    /** Lets the user transcribe a recorded clip — the controlled-conditions reference path. */
    private val videoPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) transcribeVideoFile(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val needed = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        val missing = needed.any { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing) permissionLauncher.launch(needed) else initEverything()
    }

    private fun initEverything() {
        landmarker = LipLandmarker(this)
        executor = CommandExecutor(this)
        speech = SpeechRecognizerHelper(this)

        val available = ModelRegistry.available(this)
        modelReady = available.isNotEmpty()
        if (modelReady) {
            val saved = ModelRegistry.byId(prefs.getString("model_id", null))?.takeIf { it.isAvailable(this) }
            buildForModel(saved ?: ModelRegistry.default(this)!!)
        }

        // Open-vocabulary recognizer is independent of the few-shot models; load it lazily in bg.
        freeVsrLang = FreeVsrRecognizer.Lang.entries
            .firstOrNull { it.name == prefs.getString("free_vsr_lang", null) }
            ?.takeIf { FreeVsrRecognizer.isAvailable(this, it) }
            ?: FreeVsrRecognizer.availableLangs(this).firstOrNull()
            ?: FreeVsrRecognizer.Lang.EN
        if (FreeVsrRecognizer.isAvailable(this, freeVsrLang)) {
            bg.execute { loadFreeVsr(freeVsrLang) }
        }

        camera = CameraController(this, this, binding.previewView, landmarker) { gray, mod, hasFace, preview ->
            if (!freeVsrMode && modelReady) {
                controller.onFrame(gray, mod, hasFace)
            }
            if (preview != null) runOnUiThread { binding.lipPreview.setImageBitmap(preview) }
        }
        // Free VSR consumes mean-face-aligned crops (see FaceAligner) rather than the plain lip box.
        camera.onRawFaceFrame = { faceBmp, kp ->
            if (freeVsrRecording && freeVsrRawFrames.size < FreeVsrRecognizer.MAX_FRAMES) {
                freeVsrRawFrames.add(faceBmp)
                freeVsrRawKps.add(kp)
                freeVsrTimes.add(System.currentTimeMillis())
            } else {
                faceBmp.recycle()
            }
        }
        camera.start()

        wireUi()
        if (modelReady) warmUpModel()
        else runOnUiThread {
            binding.commandLabel.text = "No model asset found.\nAdd an .onnx model to assets."
        }
        promptUserName()
    }

    /** Creates encoder + store + controller for [spec]. Caller is responsible for warm-up/loading. */
    private fun buildForModel(spec: ModelSpec) {
        encoder = LipEncoder(this, spec)
        store = CommandStore(this, spec.embedDim, spec.id)
        controller = LipRecognitionController(encoder, store).also { it.listener = this }
        activeModel = spec
    }

    private fun warmUpModel() = bg.execute {
        val s = encoder.frameSize
        runCatching { encoder.encode(List(29) { FloatArray(s * s) }) }
        controller.warmUp = false
        runOnUiThread {
            binding.recordButton.isEnabled = true
            binding.commandLabel.text = activeModel?.displayName ?: ""
        }
    }

    // --- model switching -----------------------------------------------------

    private fun switchModel(spec: ModelSpec) {
        if (spec.id == activeModel?.id) return
        modelReady = false // gate camera frames during the swap
        val loading = AlertDialog.Builder(this).setMessage("Switching model…").setCancelable(false).create()
        loading.show()
        bg.execute {
            runCatching { store.save(userName) }
            val oldController = controller
            val oldEncoder = encoder

            buildForModel(spec)
            prefs.edit().putString("model_id", spec.id).apply()
            runCatching { store.load(userName) }
            store.trainAll()
            val s = encoder.frameSize
            runCatching { encoder.encode(List(29) { FloatArray(s * s) }) }
            controller.warmUp = false

            oldController.shutdown()
            oldEncoder.close()
            modelReady = true
            runOnUiThread {
                loading.dismiss()
                binding.kwsSwitch.isChecked = false
                binding.modeGroup.check(binding.modeRegister.id)
                binding.commandLabel.text = spec.displayName
                toast("Model: ${spec.displayName} (${store.registeredCommands.size} commands)")
            }
        }
    }

    private fun showModelPicker() {
        val available = ModelRegistry.available(this)
        if (available.isEmpty()) { alert("No models available."); return }
        val names = available.map { it.displayName }.toTypedArray()
        var idx = available.indexOfFirst { it.id == activeModel?.id }.coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle("Select model")
            .setSingleChoiceItems(names, idx) { _, which -> idx = which }
            .setPositiveButton("Switch") { _, _ -> switchModel(available[idx]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // --- UI wiring -----------------------------------------------------------

    private fun wireUi() {
        binding.recordButton.isEnabled = false
        binding.menuButton.setOnClickListener { showMenu(it) }
        binding.settingButton.setOnClickListener { showSettings() }

        if (!modelReady) {
            binding.kwsSwitch.isEnabled = false
            binding.recordModeButton.isEnabled = false
            return
        }

        binding.modeGroup.check(binding.modeRegister.id)
        binding.modeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener

            // Free VSR is a separate pipeline (open vocabulary, no registration/training).
            freeVsrMode = checkedId == binding.modeFreeVsr.id
            camera.alignedOutput = freeVsrMode // only pay for the warp in free-VSR mode
            if (freeVsrMode) {
                binding.kwsSwitch.isChecked = false
                controller.keywordSpotting = false
                controller.clearBuffers()
                binding.recordModeButton.visibility = View.GONE
                binding.commandLabel.text =
                    if (freeVsr != null) "Free VSR (EN) — hold to speak"
                    else "Loading free-VSR model…"
                return@addOnButtonCheckedListener
            }

            controller.workMode = when (checkedId) {
                binding.modeRecognize.id -> LipRecognitionController.WorkMode.RECOGNITION
                binding.modeFreeUse.id -> {
                    recentFreeUse.clear()
                    LipRecognitionController.WorkMode.FREE_USE
                }
                else -> LipRecognitionController.WorkMode.REGISTRATION
            }
            binding.recordModeButton.visibility =
                if (controller.workMode == LipRecognitionController.WorkMode.REGISTRATION) View.VISIBLE else View.GONE
        }

        binding.recordModeButton.setOnClickListener { showRecordModeMenu(it) }

        binding.kwsSwitch.setOnCheckedChangeListener { _, isOn ->
            if (isOn) {
                if (store.kwsReady) controller.keywordSpotting = true
                else {
                    binding.kwsSwitch.isChecked = false
                    alert("Record keyword and non-speaking samples first!")
                }
            } else controller.keywordSpotting = false
        }

        binding.recordButton.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (freeVsrMode) {
                        startFreeVsrRecording()
                        return@setOnTouchListener true
                    }
                    if (controller.workMode == LipRecognitionController.WorkMode.REGISTRATION &&
                        controller.recordMode == LipRecognitionController.RecordMode.COMMAND
                    ) {
                        voiceLabel = null
                        startVoice2Lip()
                    }
                    controller.startRecording()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    v.performClick()
                    if (freeVsrMode) {
                        stopFreeVsrRecording()
                        return@setOnTouchListener true
                    }
                    controller.stopRecording()
                    speech.stop()
                    true
                }
                else -> false
            }
        }
    }

    private fun showRecordModeMenu(anchor: View) {
        PopupMenu(this, anchor).apply {
            menu.add("Keyword"); menu.add("Non speaking"); menu.add("Command")
            setOnMenuItemClickListener {
                controller.recordMode = when (it.title) {
                    "Keyword" -> LipRecognitionController.RecordMode.KEYWORD
                    "Non speaking" -> LipRecognitionController.RecordMode.NON_SPEAKING
                    else -> LipRecognitionController.RecordMode.COMMAND
                }
                binding.recordModeButton.text = it.title
                true
            }
            show()
        }
    }

    private fun showMenu(anchor: View) {
        PopupMenu(this, anchor).apply {
            menu.add("Select model")
            menu.add("Free VSR language")
            menu.add("Transcribe video file")
            if (modelReady) { menu.add("Save and Train"); menu.add("Reset keyword") }
            setOnMenuItemClickListener {
                when (it.title) {
                    "Select model" -> showModelPicker()
                    "Free VSR language" -> showFreeVsrLangPicker()
                    "Transcribe video file" -> videoPicker.launch("video/*")
                    "Save and Train" -> saveAndTrain()
                    "Reset keyword" -> confirmResetKeyword()
                }
                true
            }
            show()
        }
    }

    // --- user name / load ----------------------------------------------------

    private fun promptUserName() {
        val input = EditText(this).apply { hint = userName }
        AlertDialog.Builder(this)
            .setTitle("Enter your name")
            .setView(input)
            .setCancelable(false)
            .setPositiveButton("New User") { _, _ ->
                userName = input.text.toString().ifBlank { userName }
            }
            .setNeutralButton("Load Data") { _, _ ->
                userName = input.text.toString().ifBlank { userName }
                loadData()
            }
            .show()
    }

    private fun loadData() {
        if (!modelReady) return
        bg.execute {
            val ok = runCatching { store.load(userName) }.getOrDefault(false)
            if (ok) {
                store.trainAll()
                runOnUiThread { toast("Loaded ${store.registeredCommands.size} commands") }
            } else runOnUiThread { toast("No saved data for $userName") }
        }
    }

    // --- Free VSR (open vocabulary, no registration) -------------------------

    /** Loads (or swaps) the open-vocabulary model for [lang]. Runs on the bg executor. */
    private fun loadFreeVsr(lang: FreeVsrRecognizer.Lang) {
        freeVsr?.close()
        freeVsr = null
        freeVsr = runCatching { FreeVsrRecognizer(this, lang) }
            .onFailure { android.util.Log.e("LipLearner", "FreeVSR[$lang] init failed", it) }
            .getOrNull()
        freeVsrLang = lang
        // Chinese aligns on 68 landmarks; English on 4 — the smoothing history must not mix them.
        camera.alignedNeeds68 = false // both languages use the verified 4-point BlazeFace path
        camera.resetAligner()
        runOnUiThread {
            if (freeVsrMode) binding.commandLabel.text = "Free VSR: ${lang.displayName} — hold to speak"
        }
    }

    private fun showFreeVsrLangPicker() {
        val langs = FreeVsrRecognizer.availableLangs(this)
        if (langs.isEmpty()) { alert("No free-VSR model bundled."); return }
        val names = langs.map { it.displayName }.toTypedArray()
        var idx = langs.indexOf(freeVsrLang).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle("Free VSR language")
            .setSingleChoiceItems(names, idx) { _, w -> idx = w }
            .setPositiveButton("Use") { _, _ ->
                val lang = langs[idx]
                if (lang == freeVsrLang && freeVsr != null) return@setPositiveButton
                prefs.edit().putString("free_vsr_lang", lang.name).apply()
                val dlg = AlertDialog.Builder(this)
                    .setMessage("Loading ${lang.displayName}…").setCancelable(false).create()
                dlg.show()
                bg.execute {
                    loadFreeVsr(lang)
                    runOnUiThread { dlg.dismiss(); toast("Free VSR: ${lang.displayName}") }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startFreeVsrRecording() {
        if (freeVsr == null) { toast("Free-VSR model still loading…"); return }
        camera.resetAligner() // drop the smoothing window from the previous utterance
        freeVsrFrames.clear()
        freeVsrTimes.clear()
        freeVsrRawFrames.forEach { it.recycle() }
        freeVsrRawFrames.clear()
        freeVsrRawKps.clear()
        freeVsrRecording = true
        binding.commandLabel.text = "● Recording…"
    }

    private fun stopFreeVsrRecording() {
        if (!freeVsrRecording) return
        freeVsrRecording = false
        val rawFrames = ArrayList(freeVsrRawFrames)
        val rawKps = ArrayList(freeVsrRawKps)
        val times = ArrayList(freeVsrTimes)
        freeVsrRawFrames.clear(); freeVsrRawKps.clear(); freeVsrTimes.clear()
        val recognizer = freeVsr ?: return
        if (rawFrames.size < FreeVsrRecognizer.MIN_FRAMES) {
            rawFrames.forEach { it.recycle() }
            binding.commandLabel.text = "Too short — hold longer"
            return
        }
        binding.commandLabel.text = "Aligning ${rawFrames.size} frames…"
        binding.recordButton.isEnabled = false
        bg.execute {
            // Same offline, centered-window alignment as the video-file path.
            val aligned = com.rkmtlab.liplearner.vision.FaceAligner()
                .alignSequence(rawFrames, rawKps)
            rawFrames.forEach { it.recycle() }
            val clip = resampleTo25Fps(aligned, times)
            // Debug aid: dump what the model actually sees, so the mobile crop can be compared
            // pixel-for-pixel against the reference desktop pipeline.
            runCatching { dumpFreeVsrFrames(clip) }

            val t0 = System.currentTimeMillis()
            val text = runCatching { recognizer.transcribe(clip) }
                .onFailure { android.util.Log.e("LipLearner", "free-VSR failed", it) }
                .getOrNull()
            val ms = System.currentTimeMillis() - t0
            android.util.Log.i("LipLearner", "🗣 FREE-VSR (${clip.size} frames, ${ms}ms): \"$text\"")
            runOnUiThread {
                binding.commandLabel.text =
                    if (text.isNullOrBlank()) "(no speech recognized)" else text
                binding.recordButton.isEnabled = true
            }
        }
    }

    /**
     * Resamples the captured clip onto a uniform 25fps timeline (nearest frame by timestamp).
     *
     * The VSR model was trained on 25fps video. The analysis stream only delivers ~15fps, so
     * handing it the raw frames compresses the utterance in time — the model then "hears" speech
     * roughly twice as fast and the transcript degrades badly. Stretching by real elapsed time
     * restores the tempo the model expects.
     */
    private fun resampleTo25Fps(frames: List<FloatArray>, times: List<Long>): List<FloatArray> {
        if (frames.size < 2 || times.size != frames.size) return frames
        val durationMs = times.last() - times.first()
        if (durationMs <= 0) return frames
        val target = ((durationMs / 1000.0) * 25.0).toInt().coerceIn(
            FreeVsrRecognizer.MIN_FRAMES, FreeVsrRecognizer.MAX_FRAMES
        )
        val out = ArrayList<FloatArray>(target)
        var j = 0
        for (i in 0 until target) {
            val t = times.first() + (durationMs * i / (target - 1).coerceAtLeast(1))
            while (j + 1 < times.size && times[j + 1] <= t) j++
            out.add(frames[j])
        }
        val srcFps = frames.size * 1000.0 / durationMs
        android.util.Log.i(
            "LipLearner",
            "resample: ${frames.size} frames @ ${"%.1f".format(srcFps)}fps (${durationMs}ms) -> $target @ 25fps"
        )
        return out
    }

    /**
     * Transcribes a recorded video file. This is the controlled-conditions reference path: it uses
     * the same model and the same offline (centered-window) alignment as the desktop pipeline, so a
     * clip recorded with the system camera should reproduce the desktop result.
     */
    private fun transcribeVideoFile(uri: android.net.Uri) {
        val recognizer = freeVsr
        if (recognizer == null) { toast("Free-VSR model still loading…"); return }
        binding.commandLabel.text = "Reading video…"
        binding.recordButton.isEnabled = false
        bg.execute {
            val result = runCatching {
                val retriever = android.media.MediaMetadataRetriever()
                retriever.setDataSource(this, uri)
                val durUs = (retriever.extractMetadata(
                    android.media.MediaMetadataRetriever.METADATA_KEY_DURATION
                )?.toLongOrNull() ?: 0L) * 1000L
                // Sample at the 25fps the model expects.
                val stepUs = 1_000_000L / 25
                val needs68 = false // 4-point alignment works for both models
                val detector = if (needs68) null else com.rkmtlab.liplearner.vision.FastFaceDetector(this)
                val aligner = com.rkmtlab.liplearner.vision.FaceAligner()
                val frames = ArrayList<android.graphics.Bitmap>()
                val kps = ArrayList<com.rkmtlab.liplearner.vision.FaceAligner.Keypoints>()
                var t = 0L
                while (t < durUs && frames.size < FreeVsrRecognizer.MAX_FRAMES) {
                    val bmp = retriever.getFrameAtTime(
                        t, android.media.MediaMetadataRetriever.OPTION_CLOSEST
                    )
                    if (bmp != null) {
                        val pts = if (needs68) landmarker.detect(bmp).points68
                                  else detector!!.detect(bmp).alignPoints
                        pts?.let {
                            frames.add(bmp)
                            kps.add(com.rkmtlab.liplearner.vision.FaceAligner.Keypoints(it))
                        }
                    }
                    t += stepUs
                }
                retriever.release()
                detector?.close()
                android.util.Log.i("LipLearner", "video: ${frames.size} frames with a face (${durUs / 1000}ms)")
                if (frames.size < FreeVsrRecognizer.MIN_FRAMES) {
                    null
                } else {
                    val rois = aligner.alignSequence(frames, kps)
                    frames.forEach { it.recycle() }
                    recognizer.transcribe(rois)
                }
            }.onFailure { android.util.Log.e("LipLearner", "video transcribe failed", it) }
                .getOrNull()
            android.util.Log.i("LipLearner", "🎬 VIDEO-FILE result: \"$result\"")
            runOnUiThread {
                binding.commandLabel.text = result ?: "(could not read / no face found)"
                binding.recordButton.isEnabled = true
            }
        }
    }

    /** Writes the aligned ROIs as a contact-sheet PNG for offline comparison with the desktop pipeline. */
    private fun dumpFreeVsrFrames(clip: List<FloatArray>) {
        val n = minOf(clip.size, 40)
        val cols = 10
        val rows = (n + cols - 1) / cols
        val s = FreeVsrRecognizer.FRAME_SIZE
        val bmp = android.graphics.Bitmap.createBitmap(cols * s, rows * s, android.graphics.Bitmap.Config.ARGB_8888)
        val px = IntArray(s * s)
        for (i in 0 until n) {
            val f = clip[i * clip.size / n]
            for (j in 0 until s * s) {
                val v = (f[j].coerceIn(0f, 1f) * 255).toInt()
                px[j] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
            bmp.setPixels(px, 0, s, (i % cols) * s, (i / cols) * s, s, s)
        }
        val out = java.io.File(getExternalFilesDir(null), "freevsr_frames.png")
        out.outputStream().use { bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()

        // Also dump the exact float tensor fed to the model (T x 88 x 88, little-endian float32),
        // so the desktop can replay the phone's input bit-for-bit and isolate capture vs. inference.
        val raw = java.io.File(getExternalFilesDir(null), "freevsr_frames.f32")
        java.io.DataOutputStream(raw.outputStream().buffered()).use { o ->
            for (f in clip) for (v in f) {
                val bits = java.lang.Float.floatToIntBits(v)
                o.write(bits and 0xFF); o.write((bits shr 8) and 0xFF)
                o.write((bits shr 16) and 0xFF); o.write((bits shr 24) and 0xFF)
            }
        }
        android.util.Log.i("LipLearner", "dumped ${clip.size} frames -> ${out.absolutePath} + .f32")
    }

    // --- Voice2Lip -----------------------------------------------------------

    // Best-effort: if the device has a speech recognizer, capture a suggested label while recording.
    // The save dialog is NOT driven from here — it is shown on release (onCommandCaptured), so it
    // works even when no recognizer is available (the user just types the label).
    private fun startVoice2Lip() {
        if (!speech.isAvailable) return
        speech.languageTag = languageTags[languageIndex]
        speech.start(
            onPartial = { if (it.isNotBlank()) voiceLabel = it },
            onFinal = { if (!it.isNullOrBlank()) voiceLabel = it },
        )
    }

    private fun confirmSaveCommand(recognized: String?) {
        val input = EditText(this).apply {
            setText(recognized ?: "")
            hint = "Type command name"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        AlertDialog.Builder(this)
            .setTitle("Name this command")
            .setView(input)
            .setPositiveButton("OK") { _, _ ->
                commitCommand(input.text.toString().trim())
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .setNegativeButton("Misactivated") { _, _ ->
                pendingKeywordCandidate?.let {
                    store.addKwsSample("N", it); bg.execute { store.trainKws() }
                }
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .setNeutralButton("Cancel") { _, _ ->
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .show()
    }

    private fun commitCommand(label: String) {
        if (label.isBlank()) return
        val vec = pendingCommandVector ?: return
        store.addCommandSample(label, vec)
        pendingKeywordCandidate?.let { store.addKwsSample("P", it) }
        pendingCommandVector = null
        android.util.Log.i("LipLearner", "＋ registered command \"$label\" — now ${store.registeredCommands.size} commands, ${store.commandSamples.size} samples total")
        toast("Registered: $label")
    }

    // --- LipRecognitionController.Listener -----------------------------------

    override fun onRecordingStateChanged(recording: Boolean) = runOnUiThread {
        binding.commandLabel.text = if (recording) "● Recording" else binding.commandLabel.text
    }

    override fun onStatus(text: String) = runOnUiThread { binding.commandLabel.text = text }

    override fun onHaptic() = runOnUiThread {
        binding.recordButton.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
    }

    override fun onClassifierMissing() = runOnUiThread {
        alert("Please register at least two commands, then Save and Train.")
        if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
    }

    override fun onCommandCaptured(vector: FloatArray, frames: List<FloatArray>) {
        // The lip clip is now encoded; show the save dialog, prefilled with the Voice2Lip suggestion
        // if one was captured (otherwise empty, for manual entry). Timing is driven by release, not
        // by the speech callback.
        pendingCommandVector = vector
        speech.stop()
        runOnUiThread { confirmSaveCommand(voiceLabel) }
    }

    override fun onRecognitionResult(
        result: String,
        sortedCommands: List<String>,
        vector: FloatArray,
        keywordCandidate: FloatArray?,
    ) = runOnUiThread {
        binding.commandLabel.text = result
        pendingKeywordCandidate = keywordCandidate
        AlertDialog.Builder(this)
            .setTitle("Is \"$result\" correct?")
            .setPositiveButton("Add sample") { _, _ ->
                chooseLabel(sortedCommands, result) { label ->
                    store.addCommandSample(label, vector)
                    keywordCandidate?.let { store.addKwsSample("P", it) }
                    toast("Added sample: $label")
                }
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .setNegativeButton("Misactivated") { _, _ ->
                keywordCandidate?.let {
                    store.addKwsSample("N", it); bg.execute { store.trainKws() }
                }
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .setNeutralButton("Cancel") { _, _ ->
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
            .show()
    }

    override fun onFreeUseResult(result: String, vector: FloatArray) = runOnUiThread {
        binding.commandLabel.text = result
        recentFreeUse.add(result)
        executor.execute(result)
        if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
    }

    override fun onTryAgain() = runOnUiThread {
        binding.commandLabel.text = "Try again"
        if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
    }

    // --- dialogs / menus -----------------------------------------------------

    private fun chooseLabel(options: List<String>, preselected: String, onPick: (String) -> Unit) {
        val items = options.toTypedArray()
        var idx = options.indexOf(preselected).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle("Choose the correct command")
            .setSingleChoiceItems(items, idx) { _, which -> idx = which }
            .setPositiveButton("OK") { _, _ -> if (items.isNotEmpty()) onPick(items[idx]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun saveAndTrain() {
        if (!modelReady) return
        controller.keywordSpotting = false
        val dialog = AlertDialog.Builder(this).setMessage("Please wait…").setCancelable(false).create()
        dialog.show()
        bg.execute {
            store.trainAll()
            runCatching { store.save(userName) }
            android.util.Log.i("LipLearner", "✅ trained: commands=${store.registeredCommands} " +
                "(${store.commandSamples.size} samples), classifier=${if (store.commandClassifier != null) "OK" else "NOT trained (need ≥2 commands)"}")
            runOnUiThread {
                dialog.dismiss()
                toast("Trained ${store.registeredCommands.size} commands")
                if (binding.kwsSwitch.isChecked) controller.keywordSpotting = true
            }
        }
    }

    private fun confirmResetKeyword() {
        AlertDialog.Builder(this)
            .setTitle("Reset keyword and non-speaking data?")
            .setPositiveButton("OK") { _, _ -> if (modelReady) store.resetKeyword() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showSettings() {
        AlertDialog.Builder(this)
            .setTitle("Command registration language")
            .setSingleChoiceItems(languageNames.toTypedArray(), languageIndex) { d, which ->
                languageIndex = which
                speech.languageTag = languageTags[which]
                d.dismiss()
            }
            .setPositiveButton("Close", null)
            .show()
    }

    private fun alert(msg: String) =
        AlertDialog.Builder(this).setMessage(msg).setPositiveButton("OK", null).show()

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        if (::camera.isInitialized) camera.stop()
        if (::controller.isInitialized) controller.shutdown()
        if (::encoder.isInitialized) encoder.close()
        freeVsr?.close()
        if (::landmarker.isInitialized) landmarker.close()
        if (::speech.isInitialized) speech.stop()
        bg.shutdown()
    }
}
