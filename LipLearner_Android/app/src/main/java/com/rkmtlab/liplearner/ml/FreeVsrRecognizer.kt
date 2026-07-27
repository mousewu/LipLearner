package com.rkmtlab.liplearner.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * Open-vocabulary ("free") visual speech recognition — transcribes arbitrary English sentences
 * with **no command registration**.
 *
 * Backed by the auto_avsr Conformer encoder + CTC head exported to ONNX (fp16). Only the encoder
 * and the CTC projection are on-device: the graph is a SINGLE forward pass, so it can run on the
 * NPU/GPU. The autoregressive Transformer decoder and beam search are deliberately dropped —
 * decoding here is plain CTC greedy (argmax → collapse repeats → drop blank), which needs no
 * model state and costs microseconds.
 *
 * Contract: input `v` (1, T, 1, 88, 88) grayscale in [0,1], normalized with the model's own
 * mean/std here (auto_avsr uses 0.421/0.165); output `ctc_logits` (1, T', 5047).
 */
class FreeVsrRecognizer(
    context: Context,
    private val lang: Lang = Lang.EN,
) {
    /**
     * The two open-vocabulary models. They share the input contract and the 96x96 mean-face crop
     * convention, but differ in vocabulary, tokenization and the landmarks used for alignment:
     * English (auto_avsr) aligns on 4 BlazeFace keypoints, Chinese (CNVSRC) on 68 dlib-style points.
     */
    enum class Lang(
        val displayName: String,
        val modelAsset: String,
        val tokensAsset: String,
        /** English ships bare units and needs <blank>/<eos> added; Chinese ships the full list. */
        val tokensNeedSpecials: Boolean,
    ) {
        // Both models consume the same mean-face-aligned 96x96 crop, so both use the cheap
        // 4-keypoint BlazeFace alignment. (CNVSRC trains with 68 landmarks, but feeding it the
        // 4-point crop was verified to work end-to-end, and the 68-point path was both slower —
        // ~15fps — and fragile on device.)
        EN("English (auto_avsr)", "free_vsr_en.onnx", "free_vsr_en_tokens.txt", true),
        ZH("中文 (CNVSRC)", "free_vsr_cn.onnx", "free_vsr_cn_tokens.txt", false),
    }

    companion object {
        private const val TAG = "LipLearner"
        const val FRAME_SIZE = 88
        const val MIN_FRAMES = 12
        const val MAX_FRAMES = 400
        private const val MEAN = 0.421f
        private const val STD = 0.165f
        private const val BLANK = 0
        private const val WORD_PREFIX = '▁' // SentencePiece '▁'

        fun isAvailable(context: Context, lang: Lang = Lang.EN): Boolean =
            runCatching {
                context.assets.list("")?.contains(lang.modelAsset) == true
            }.getOrDefault(false)

        fun availableLangs(context: Context): List<Lang> =
            Lang.entries.filter { isAvailable(context, it) }
    }

    private val modelAsset = lang.modelAsset
    private val tokensAsset = lang.tokensAsset

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val tokens: List<String>

    init {
        // Stream the (large) model out of the APK once, then let ORT mmap it from disk.
        val modelFile = File(context.filesDir, "models/$modelAsset")
        // Re-extract when the packaged asset differs in size from the cached copy, otherwise an
        // updated model in the APK would be silently ignored.
        val assetSize = runCatching {
            context.assets.openFd(modelAsset).use { it.declaredLength }
        }.getOrDefault(-1L)
        if (!modelFile.exists() || modelFile.length() == 0L ||
            (assetSize > 0 && modelFile.length() != assetSize)
        ) {
            modelFile.parentFile?.mkdirs()
            context.assets.open(modelAsset).use { input ->
                modelFile.outputStream().use { out -> input.copyTo(out, 64 * 1024) }
            }
        }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // NOTE: NNAPI is deliberately NOT enabled. It silently miscomputes this graph — with
            // the identical input tensor, NNAPI produced "YORK" where CPU (and the desktop
            // reference) produce "PLEASE CALLED THE DAUGHTER". The Conformer's dynamic-length
            // attention isn't fully supported, and the partitioned fallback corrupts the result.
        }
        session = env.createSession(modelFile.absolutePath, opts)
        inputName = session.inputNames.iterator().next()
        // Rebuild the exact vocabulary the model was trained with:
        //   token_list = ["<blank>"] + units file (5047 entries) + ["<eos>"]  => 5049 ids
        val lines = context.assets.open(tokensAsset).bufferedReader(Charsets.UTF_8).readLines()
        tokens = if (lang.tokensNeedSpecials) {
            buildList {
                add("<blank>")
                addAll(lines.filter { it.isNotBlank() })
                add("<eos>")
            }
        } else {
            // Chinese export already contains <blank>/<unk>/.../<eos>; keep blank lines out but do
            // not re-index, so drop only a trailing empty line from the file write.
            lines.dropLastWhile { it.isEmpty() }
        }
        Log.i(TAG, "FreeVSR[${lang.name}] ready: ${tokens.size} tokens")
    }

    /**
     * @param frames grayscale 88x88 frames in [0,1] (same crops the few-shot encoder consumes).
     * @return the transcribed sentence, or null if the clip is unusable.
     */
    fun transcribe(frames: List<FloatArray>): String? {
        if (frames.size < MIN_FRAMES) return null
        val clip = if (frames.size > MAX_FRAMES) frames.subList(0, MAX_FRAMES) else frames
        val t = clip.size
        val pixels = FRAME_SIZE * FRAME_SIZE

        val buffer = FloatBuffer.allocate(t * pixels)
        for (f in clip) {
            require(f.size == pixels) { "frame must be $pixels floats, got ${f.size}" }
            for (p in f) buffer.put((p - MEAN) / STD)
        }
        buffer.rewind()

        val shape = longArrayOf(1, t.toLong(), 1, FRAME_SIZE.toLong(), FRAME_SIZE.toLong())
        OnnxTensor.createTensor(env, buffer, shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result[0].value as Array<Array<FloatArray>>)[0] // (T', vocab)
                return decodeGreedy(logits)
            }
        }
    }

    /** CTC greedy: per-frame argmax, collapse repeats, drop blanks, then de-tokenize. */
    private fun decodeGreedy(logits: Array<FloatArray>): String {
        val ids = ArrayList<Int>(logits.size)
        var prev = -1
        for (frame in logits) {
            var best = 0
            for (i in frame.indices) if (frame[i] > frame[best]) best = i
            if (best != prev) {
                if (best != BLANK) ids.add(best)
                prev = best
            }
        }
        val sb = StringBuilder()
        for (id in ids) {
            val tok = tokens.getOrNull(id) ?: continue
            if (tok.isEmpty() || tok.startsWith("<")) continue // skip <blank>/<unk>/<eos>
            if (tok[0] == WORD_PREFIX) {
                // '▁' marks a word start in English; in Chinese it is just the space token.
                if (lang == Lang.EN) {
                    if (sb.isNotEmpty()) sb.append(' ')
                    sb.append(tok.substring(1))
                } else if (tok.length > 1) {
                    sb.append(tok.substring(1))
                }
            } else {
                sb.append(tok)
            }
        }
        return sb.toString().trim()
    }

    fun close() = session.close()
}
