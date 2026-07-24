package com.rkmtlab.liplearner.kws

import android.util.Log
import com.rkmtlab.liplearner.ml.CommandStore
import com.rkmtlab.liplearner.ml.LipEncoder
import com.rkmtlab.liplearner.ml.VectorMath
import java.util.concurrent.Executors

private const val TAG = "LipLearner"

/**
 * The real-time recognition state machine. This is the Android port of the core logic in
 * CameraViewController.swift: keyword spotting (KWS), silent-speech activity detection (SSAD),
 * automatic start/stop of recording, and per-mode handling of a captured utterance.
 *
 * UIKit is replaced by the [Listener] callbacks; all heavy work (encoding/classifying) runs on a
 * background single-thread executor, and results are handed back through the listener.
 */
class LipRecognitionController(
    private val encoder: LipEncoder,
    private val store: CommandStore,
) {
    enum class WorkMode { REGISTRATION, RECOGNITION, FREE_USE }
    enum class RecordMode { COMMAND, KEYWORD, NON_SPEAKING }

    interface Listener {
        fun onRecordingStateChanged(recording: Boolean)
        fun onStatus(text: String)
        fun onHaptic()
        fun onClassifierMissing()
        /** Registration/command utterance captured; UI should run Voice2Lip and ask for a label. */
        fun onCommandCaptured(vector: FloatArray, frames: List<FloatArray>)
        fun onRecognitionResult(result: String, sortedCommands: List<String>, vector: FloatArray, keywordCandidate: FloatArray?)
        fun onFreeUseResult(result: String, vector: FloatArray)
        fun onTryAgain()
    }

    var listener: Listener? = null
    var workMode = WorkMode.REGISTRATION
    var recordMode = RecordMode.COMMAND
    @Volatile var recording = false
        private set
    @Volatile var warmUp = true

    var keywordThreshold = 0.65f
    var nonSpeakingThreshold = 0.65f

    @Volatile var keywordSpotting = false
        set(value) {
            field = value
            if (value) keywordSpottingBuffer.clear()
        }

    private val kwsWindowSize = 30
    private val hopSize = 10
    private val MAX_RECORD_FRAMES = 128 // safety cap on a single utterance

    private val modelInput = ArrayList<FloatArray>(160)
    private val keywordSpottingBuffer = ArrayList<FloatArray>(64)
    private var keywordCandidate: FloatArray? = null

    // MOD tracking (delayed by 15 frames, matching the iOS MODQueue)
    private val modQueue = ArrayDeque<Float>()
    @Volatile private var mod = 0f
    @Volatile private var delayedMod = 0f

    private val worker = Executors.newSingleThreadExecutor()

    /** Called for every camera frame (background thread). */
    fun onFrame(grayFrame: FloatArray, frameMod: Float, hasFace: Boolean) {
        mod = frameMod
        modQueue.addLast(frameMod)
        if (modQueue.size >= 15) delayedMod = modQueue.removeFirst()

        if (keywordSpotting) {
            keywordSpottingBuffer.add(grayFrame)
            if (keywordSpottingBuffer.size == kwsWindowSize) {
                val window = ArrayList(keywordSpottingBuffer)
                // keep the last (window - hop) frames for the next hop
                val keep = keywordSpottingBuffer.takeLast(kwsWindowSize - hopSize)
                keywordSpottingBuffer.clear()
                keywordSpottingBuffer.addAll(keep)
                val gate = (mod >= 0.1f || delayedMod >= 0.1f || recording)
                if (gate) worker.execute { keywordDetection(window) }
            }
        }
        if (recording) {
            modelInput.add(grayFrame)
            if (modelInput.size == MAX_RECORD_FRAMES) {
                stopRecording()
            }
        }
    }

    // --- manual (long-press) recording ---------------------------------------

    fun startRecording() {
        recording = true
        Log.i(TAG, "▶ start recording (mode=$workMode/$recordMode)")
        listener?.onRecordingStateChanged(true)
    }

    /** Mirrors iOS recordReleased: stop, then process the captured clip in the background. */
    fun stopRecording() {
        // In recognition/free-use with KWS off and no keyword-triggered clip, just reset.
        if (workMode != WorkMode.REGISTRATION && !keywordSpotting && !recording) {
            modelInput.clear(); keywordSpottingBuffer.clear()
            return
        }
        recording = false
        listener?.onRecordingStateChanged(false)

        val frames = ArrayList(modelInput)
        modelInput.clear()
        worker.execute { processClip(frames) }
    }

    private fun processClip(rawFrames: List<FloatArray>) {
        var frames = rawFrames
        if (keywordSpotting) {
            // drop the trailing ~20 frames captured after EOS was detected
            frames = if (frames.size > 20) frames.subList(0, frames.size - 20) else frames
            keywordSpotting = false
        }
        if (workMode == WorkMode.REGISTRATION && recordMode == RecordMode.NON_SPEAKING) {
            frames = frames.subList(0, minOf(frames.size, 30))
        }

        Log.i(TAG, "⏹ stop recording, captured ${frames.size} frames → encoding")
        val vector = encoder.encode(frames)
        if (vector == null) {
            Log.w(TAG, "encode returned null (clip too short/long: ${frames.size} frames) → Try again")
            listener?.onTryAgain()
            if (keywordSpotting) keywordSpotting = true
            return
        }
        Log.i(TAG, "encoded ok (dim=${vector.size})")

        when (workMode) {
            WorkMode.REGISTRATION -> handleRegistration(vector, ArrayList(frames))
            WorkMode.RECOGNITION -> handleRecognition(vector)
            WorkMode.FREE_USE -> handleFreeUse(vector)
        }
    }

    private fun handleRegistration(vector: FloatArray, frames: List<FloatArray>) {
        when (recordMode) {
            RecordMode.KEYWORD -> {
                store.addKeywordSample(vector)
                store.addKwsSample("P", store.keywordCenter.mean.copyOf())
                Log.i(TAG, "＋ keyword sample #${store.keywordCenter.count.toInt()}")
            }
            RecordMode.NON_SPEAKING -> {
                store.addNonSpeakingSample(vector)
                store.addKwsSample("N", store.nonSpeakingCenter.mean.copyOf())
                Log.i(TAG, "＋ non-speaking sample #${store.nonSpeakingCenter.count.toInt()}")
            }
            RecordMode.COMMAND -> {
                if (warmUp) {
                    warmUp = false
                    recordMode = RecordMode.KEYWORD
                    return
                }
                Log.i(TAG, "＋ command clip captured → asking for label")
                listener?.onCommandCaptured(vector, frames)
            }
        }
    }

    private fun handleRecognition(vector: FloatArray) {
        val clf = store.commandClassifier
        if (clf == null) { Log.w(TAG, "no classifier — register ≥2 commands then Save and Train"); listener?.onClassifierMissing(); return }
        val result = clf.predict(vector)
        Log.i(TAG, "🔎 RECOGNIZED: \"$result\"  | ${topProbs(clf, vector)}")
        val sorted = store.sortCommandsBySimilarity(vector).toMutableList()
        sorted.remove(result)
        sorted.add(0, result)
        listener?.onRecognitionResult(result, sorted, vector, keywordCandidate)
    }

    private fun handleFreeUse(vector: FloatArray) {
        val clf = store.commandClassifier
        if (clf == null) { Log.w(TAG, "no classifier — register ≥2 commands then Save and Train"); listener?.onClassifierMissing(); return }
        val result = clf.predict(vector)
        Log.i(TAG, "▶ FREE-USE: \"$result\"  | ${topProbs(clf, vector)}")
        store.addCommandSample(result, vector) // continuous learning buffer (reviewable in settings)
        listener?.onFreeUseResult(result, vector)
    }

    private fun topProbs(clf: com.rkmtlab.liplearner.ml.SoftmaxRegression, vector: FloatArray): String =
        clf.predictProba(vector).entries.sortedByDescending { it.value }.take(3)
            .joinToString(", ") { "${it.key}=${"%.2f".format(it.value)}" }

    // --- KWS core (iOS keywordDetection) -------------------------------------

    private fun keywordDetection(window: List<FloatArray>) {
        val keywordVector = encoder.encode(window) ?: return
        if (!recording) {
            val sim = VectorMath.kwsScore(keywordVector, store.keywordCenter.mean)
            Log.d(TAG, "KWS keyword sim=${"%.3f".format(sim)} (thr=$keywordThreshold)")
            if (sim > keywordThreshold) {
                keywordCandidate = keywordVector
                val kws = store.kwsClassifier
                if (kws != null && kws.predict(keywordVector) != "P") { Log.d(TAG, "KWS classifier rejected (not P)"); return }
                Log.i(TAG, "🎙 KWS TRIGGERED (sim=${"%.3f".format(sim)}) → recording")
                listener?.onHaptic()
                listener?.onStatus("")
                recording = true
                listener?.onRecordingStateChanged(true)
            }
        } else {
            if (modelInput.size > 45) {
                val sim = VectorMath.kwsScore(keywordVector, store.nonSpeakingCenter.mean)
                if (sim > nonSpeakingThreshold) {
                    stopRecording()
                }
            }
        }
    }

    fun clearBuffers() {
        modelInput.clear()
        keywordSpottingBuffer.clear()
    }

    fun shutdown() = worker.shutdown()
}
