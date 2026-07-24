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

        camera = CameraController(this, this, binding.previewView, landmarker) { gray, mod, hasFace, preview ->
            if (modelReady) controller.onFrame(gray, mod, hasFace)
            if (preview != null) runOnUiThread { binding.lipPreview.setImageBitmap(preview) }
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
            if (modelReady) { menu.add("Save and Train"); menu.add("Reset keyword") }
            setOnMenuItemClickListener {
                when (it.title) {
                    "Select model" -> showModelPicker()
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
        if (::landmarker.isInitialized) landmarker.close()
        if (::speech.isInitialized) speech.stop()
        bg.shutdown()
    }
}
