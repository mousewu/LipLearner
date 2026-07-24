package com.rkmtlab.liplearner.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale

/**
 * Voice2Lip helper. Android replacement for iOS `SFSpeechRecognizer`: while the user records a
 * command aloud during registration, this transcribes the speech so it can label the silent-speech
 * sample (fast command registration).
 */
class SpeechRecognizerHelper(private val context: Context) {

    private var recognizer: SpeechRecognizer? = null
    var languageTag: String = "en-US"

    val isAvailable: Boolean get() = SpeechRecognizer.isRecognitionAvailable(context)

    /** Starts listening. [onFinal] is called once with the best transcription (or null). */
    fun start(onPartial: (String) -> Unit, onFinal: (String?) -> Unit) {
        stop()
        if (!isAvailable) { onFinal(null); return }
        val sr = SpeechRecognizer.createSpeechRecognizer(context)
        recognizer = sr
        sr.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) { onFinal(null) }
            override fun onEvent(eventType: Int, params: Bundle?) {}

            override fun onPartialResults(partialResults: Bundle?) {
                partialResults?.stringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()?.let(onPartial)
            }

            override fun onResults(results: Bundle?) {
                val text = results?.stringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                onFinal(text)
            }
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, Locale(languageTag).language)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        sr.startListening(intent)
    }

    fun stop() {
        recognizer?.apply {
            stopListening()
            destroy()
        }
        recognizer = null
    }

    private fun Bundle.stringArrayList(key: String): ArrayList<String>? =
        getStringArrayList(key)
}
