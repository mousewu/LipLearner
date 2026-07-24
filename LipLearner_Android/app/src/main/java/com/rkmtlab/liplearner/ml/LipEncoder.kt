package com.rkmtlab.liplearner.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * On-device lip encoder, backed by ONNX Runtime Mobile and driven by a [ModelSpec].
 *
 * Takes a sequence of grayscale [ModelSpec.frameSize]² lip frames and returns an L2-normalized
 * embedding. Two length regimes:
 *   - variable length (spec.fixedFrames == null): the clip is passed as-is if in [minFrames,maxFrames]
 *     (original LipLearner GRU encoder).
 *   - fixed length (spec.fixedFrames != null): the clip is resampled to exactly that many frames,
 *     matching how the mpc001 models were trained (29-frame LRW clips) and the fixed ONNX graph.
 *
 * All models take input `v` = (1, T, 1, S, S) in [0,1]; per-model normalization is baked into the
 * ONNX graph, so preprocessing is identical across models.
 */
class LipEncoder(context: Context, val spec: ModelSpec) {

    companion object {
        // Kept for the warm-up call and generic callers.
        const val FRAME_SIZE = 88
    }

    val embedDim: Int get() = spec.embedDim
    val frameSize: Int get() = spec.frameSize
    private val pixels = spec.frameSize * spec.frameSize

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        // Stream the model out of the APK into internal storage (small buffer, no full-file heap
        // allocation), then let ONNX Runtime mmap it from the file path. Loading a large model via
        // assets.readBytes() would OOM the Java heap (the 145MB ResNet exceeds the 256MB heap limit).
        val modelFile = File(context.filesDir, "models/${spec.asset}")
        if (!modelFile.exists() || modelFile.length() == 0L) {
            modelFile.parentFile?.mkdirs()
            context.assets.open(spec.asset).use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output, 64 * 1024) }
            }
        }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelFile.absolutePath, opts)
        inputName = session.inputNames.iterator().next()
    }

    /**
     * @param frames each element is a length-(frameSize²) grayscale FloatArray in [0,1], row-major.
     * @return L2-normalized embedding, or null if the clip length is unusable.
     */
    fun encode(frames: List<FloatArray>): FloatArray? {
        val prepared = prepareFrames(frames) ?: return null
        val t = prepared.size

        val buffer = FloatBuffer.allocate(t * pixels)
        for (frame in prepared) {
            require(frame.size == pixels) { "frame must be $pixels floats, got ${frame.size}" }
            buffer.put(frame)
        }
        buffer.rewind()

        val shape = longArrayOf(1, t.toLong(), 1, spec.frameSize.toLong(), spec.frameSize.toLong())
        OnnxTensor.createTensor(env, buffer, shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = (result[0].value as Array<FloatArray>)[0]
                return l2Normalize(out)
            }
        }
    }

    private fun prepareFrames(frames: List<FloatArray>): List<FloatArray>? {
        val fixed = spec.fixedFrames
        if (fixed != null) {
            if (frames.size < 5) return null
            return resample(frames, fixed)
        }
        if (frames.size < spec.minFrames || frames.size > spec.maxFrames) return null
        return frames
    }

    /** Even-spaced temporal resample to exactly [target] frames (nearest-index). */
    private fun resample(frames: List<FloatArray>, target: Int): List<FloatArray> {
        if (frames.size == target) return frames
        val n = frames.size
        return List(target) { i ->
            val src = if (target == 1) 0 else (i.toDouble() * (n - 1) / (target - 1)).roundToInt()
            frames[src.coerceIn(0, n - 1)]
        }
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq).coerceAtLeast(1e-12f)
        return FloatArray(v.size) { v[it] / norm }
    }

    fun close() = session.close()
}
