package com.rkmtlab.liplearner.ml

import android.util.Log
import kotlin.math.exp
import kotlin.math.ln

/**
 * Multinomial logistic-regression classifier trained fully on-device.
 *
 * Android replacement for iOS `MLLogisticRegressionClassifier` (CreateML). Because the lip encoder
 * already produces a well-separated 500-D embedding space, a light L2-regularized softmax trained
 * with a few hundred gradient steps matches the few-shot behaviour of the paper. Training a fresh
 * model over a handful of samples takes only a few milliseconds.
 */
class SoftmaxRegression private constructor(
    private val labels: List<String>,
    private val weights: Array<FloatArray>, // [numClasses][dim]
    private val bias: FloatArray,           // [numClasses]
) {
    val numClasses: Int get() = labels.size

    /** Returns the most likely label. */
    fun predict(x: FloatArray): String {
        val logits = logits(x)
        var best = 0
        for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
        return labels[best]
    }

    /** Returns label -> probability (softmax over classes). */
    fun predictProba(x: FloatArray): Map<String, Float> {
        val p = softmax(logits(x))
        return labels.indices.associate { labels[it] to p[it] }
    }

    private fun logits(x: FloatArray): FloatArray {
        val out = FloatArray(labels.size)
        for (c in labels.indices) {
            var s = bias[c]
            val w = weights[c]
            for (j in x.indices) s += w[j] * x[j]
            out[c] = s
        }
        return out
    }

    companion object {
        private fun softmax(z: FloatArray): FloatArray {
            var max = z[0]
            for (v in z) if (v > max) max = v
            var sum = 0f
            val e = FloatArray(z.size)
            for (i in z.indices) {
                e[i] = exp(z[i] - max)
                sum += e[i]
            }
            for (i in z.indices) e[i] /= sum
            return e
        }

        /**
         * Trains a classifier. Returns null if fewer than 2 distinct labels are present, mirroring
         * iOS `trainClassifier` (which requires uniqueCount > 1).
         */
        fun train(
            samples: List<Pair<String, FloatArray>>,
            epochs: Int = 300,
            lr: Float = 0.5f,
            l2: Float = 1e-4f,
        ): SoftmaxRegression? {
            if (samples.isEmpty()) return null
            val labels = samples.map { it.first }.distinct().sorted()
            if (labels.size < 2) return null
            val labelIndex = labels.withIndex().associate { (i, l) -> l to i }
            val dim = samples.first().second.size
            val n = samples.size
            val c = labels.size

            val w = Array(c) { FloatArray(dim) }
            val b = FloatArray(c)

            val xs = Array(n) { samples[it].second }
            val ys = IntArray(n) { labelIndex.getValue(samples[it].first) }

            val gradW = Array(c) { FloatArray(dim) }
            val gradB = FloatArray(c)

            for (epoch in 0 until epochs) {
                for (k in 0 until c) {
                    gradW[k].fill(0f)
                    gradB[k] = 0f
                }
                var epochLoss = 0f
                for (i in 0 until n) {
                    val x = xs[i]
                    // logits
                    val z = FloatArray(c)
                    for (k in 0 until c) {
                        var s = b[k]
                        val wk = w[k]
                        for (j in 0 until dim) s += wk[j] * x[j]
                        z[k] = s
                    }
                    val p = softmax(z)
                    epochLoss += -ln(p[ys[i]].coerceIn(1e-9f, 1f))
                    for (k in 0 until c) {
                        val err = p[k] - if (ys[i] == k) 1f else 0f
                        gradB[k] += err
                        val gk = gradW[k]
                        for (j in 0 until dim) gk[j] += err * x[j]
                    }
                }
                val scale = lr / n
                for (k in 0 until c) {
                    val wk = w[k]
                    val gk = gradW[k]
                    for (j in 0 until dim) {
                        wk[j] -= scale * (gk[j] + l2 * wk[j])
                    }
                    b[k] -= scale * gradB[k]
                }
                if (epoch == 0 || epoch == epochs / 3 || epoch == 2 * epochs / 3 || epoch == epochs - 1) {
                    Log.i("LipLearner", "  train epoch $epoch/$epochs  loss=${"%.4f".format(epochLoss / n)}")
                }
            }
            Log.i("LipLearner", "classifier fit: $n samples, $c classes, ${dim} dims, ${c * (dim + 1)} params updated")
            return SoftmaxRegression(labels, w, b)
        }
    }
}

/** Cross-entropy helper kept for parity/debugging with the PyTorch pretraining loss. */
internal fun crossEntropy(prob: Float): Float = -ln(prob.coerceIn(1e-9f, 1f))
