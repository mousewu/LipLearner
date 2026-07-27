package com.rkmtlab.liplearner.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.sqrt

/**
 * Reproduces auto_avsr's `VideoProcess` mouth-ROI pipeline on Android.
 *
 * The open-vocabulary VSR model was trained on frames that are **affinely aligned to a mean face**
 * before cropping — not on a plain "square around the lips". Feeding it the simple crop used by the
 * few-shot encoder puts the input off-distribution and the transcript degrades badly, so this class
 * mirrors the reference pipeline:
 *
 *   1. take 4 stable face keypoints (right eye, left eye, nose tip, mouth center),
 *   2. temporally smooth them over a +/-6 frame window (window_margin = 12),
 *   3. estimate a similarity transform onto the mean-face reference (256x256 space),
 *   4. warp the frame and cut a 96x96 patch centered on the transformed mouth point.
 *
 * The model's own transform then center-crops 96 -> 88 and converts to grayscale.
 */
class FaceAligner {

    companion object {
        const val CROP = 96
        private const val REF_SIZE = 256
        const val OUT = 88
        private const val WINDOW_MARGIN = 12

        /**
         * MediaPipe FaceMesh indices approximating the 68-point dlib layout that CNVSRC (Chinese)
         * aligns with. English (auto_avsr) only needs the 4 keypoints below.
         */
        val MP_TO_68 = intArrayOf(
            127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
            70, 63, 105, 66, 107,
            336, 296, 334, 293, 300,
            168, 197, 5, 4,
            75, 97, 2, 326, 305,
            33, 160, 158, 133, 153, 144,
            362, 385, 387, 263, 373, 380,
            61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
            78, 82, 13, 312, 308, 317, 14, 87,
        )

        /** Mean-face reference points in the 256x256 target space (from 20words_mean_face.npy). */
        private val STABLE_REFERENCE = arrayOf(
            floatArrayOf(102.073943f, 94.272304f),  // right eye center
            floatArrayOf(156.361305f, 93.578156f),  // left eye center
            floatArrayOf(129.003738f, 135.903430f), // nose
            floatArrayOf(129.313373f, 157.822996f), // mouth center
        )

        /** Full 68-point mean face (20words_mean_face.npy), used by the Chinese pipeline. */
        private val REFERENCE_68 = arrayOf(
            floatArrayOf(70.924f, 97.138f), floatArrayOf(72.625f, 114.904f), floatArrayOf(75.989f, 131.074f),
            floatArrayOf(79.270f, 146.212f), floatArrayOf(83.613f, 163.260f), floatArrayOf(91.331f, 177.445f),
            floatArrayOf(100.272f, 187.089f), floatArrayOf(112.124f, 196.004f), floatArrayOf(130.742f, 200.530f),
            floatArrayOf(149.855f, 195.312f), floatArrayOf(163.102f, 186.441f), floatArrayOf(173.653f, 176.573f),
            floatArrayOf(182.436f, 162.280f), floatArrayOf(187.167f, 145.094f), floatArrayOf(190.229f, 129.724f),
            floatArrayOf(193.021f, 113.459f), floatArrayOf(194.439f, 95.580f),
            floatArrayOf(81.331f, 80.795f), floatArrayOf(87.759f, 75.280f), floatArrayOf(96.227f, 73.839f),
            floatArrayOf(104.555f, 74.740f), floatArrayOf(112.232f, 76.977f),
            floatArrayOf(144.496f, 76.424f), floatArrayOf(152.348f, 73.833f), floatArrayOf(161.131f, 72.636f),
            floatArrayOf(170.587f, 73.848f), floatArrayOf(178.214f, 79.438f),
            floatArrayOf(128.734f, 95.360f), floatArrayOf(128.489f, 106.925f), floatArrayOf(128.245f, 118.273f),
            floatArrayOf(128.266f, 127.699f),
            floatArrayOf(118.760f, 135.194f), floatArrayOf(122.963f, 136.146f), floatArrayOf(128.870f, 137.303f),
            floatArrayOf(134.943f, 135.997f), floatArrayOf(139.483f, 134.878f),
            floatArrayOf(92.522f, 94.369f), floatArrayOf(97.585f, 90.960f), floatArrayOf(105.414f, 90.913f),
            floatArrayOf(112.772f, 94.944f), floatArrayOf(106.104f, 97.085f), floatArrayOf(98.046f, 97.363f),
            floatArrayOf(145.525f, 94.535f), floatArrayOf(152.590f, 90.215f), floatArrayOf(160.612f, 90.199f),
            floatArrayOf(166.677f, 93.566f), floatArrayOf(160.560f, 96.481f), floatArrayOf(152.205f, 96.473f),
            floatArrayOf(107.168f, 157.196f), floatArrayOf(114.476f, 152.120f), floatArrayOf(123.849f, 148.519f),
            floatArrayOf(128.976f, 149.416f), floatArrayOf(134.144f, 148.426f), floatArrayOf(144.177f, 151.793f),
            floatArrayOf(152.193f, 156.987f), floatArrayOf(143.860f, 164.003f), floatArrayOf(136.744f, 167.943f),
            floatArrayOf(129.153f, 168.819f), floatArrayOf(121.795f, 168.023f), floatArrayOf(115.275f, 164.152f),
            floatArrayOf(109.231f, 157.002f), floatArrayOf(122.503f, 154.407f), floatArrayOf(129.029f, 154.121f),
            floatArrayOf(135.836f, 154.312f), floatArrayOf(150.758f, 156.795f), floatArrayOf(135.662f, 160.630f),
            floatArrayOf(128.952f, 161.288f), floatArrayOf(122.488f, 160.509f),
        )
    }

    /**
     * One frame's keypoints in image pixels. Either 4 points ([rightEye, leftEye, nose, mouth],
     * English/auto_avsr) or the full 68-point dlib layout (Chinese/CNVSRC).
     */
    class Keypoints(val pts: Array<FloatArray>)

    private val history = ArrayDeque<Keypoints>()

    fun reset() = history.clear()

    /**
     * Offline variant matching the reference pipeline exactly: smooths each frame's keypoints with a
     * **centered** +/-6 window (the streaming path can only look backwards). Free-VSR runs after the
     * user releases the button, so the whole utterance is available and we can afford this.
     *
     * @param frames the upright camera bitmaps of the utterance
     * @param kps    per-frame keypoints, same length as [frames]
     */
    fun alignSequence(frames: List<Bitmap>, kps: List<Keypoints>): List<FloatArray> {
        val n = frames.size
        val out = ArrayList<FloatArray>(n)
        for (i in 0 until n) {
            val margin = minOf(WINDOW_MARGIN / 2, i, n - 1 - i)
            val np = kps[i].pts.size
            val mean = Array(np) { FloatArray(2) }
            var count = 0
            for (j in (i - margin)..(i + margin)) {
                val p = kps[j].pts
                for (k in 0 until np) { mean[k][0] += p[k][0]; mean[k][1] += p[k][1] }
                count++
            }
            for (k in 0 until np) {
                mean[k][0] = mean[k][0] / count
                mean[k][1] = mean[k][1] / count
            }
            // re-center the smoothed shape on this frame's own centroid
            val cur = kps[i].pts
            var cx = 0f; var cy = 0f; var sx = 0f; var sy = 0f
            for (k in 0 until np) {
                cx += cur[k][0] / np; cy += cur[k][1] / np
                sx += mean[k][0] / np; sy += mean[k][1] / np
            }
            for (k in 0 until np) { mean[k][0] += cx - sx; mean[k][1] += cy - sy }
            out.add(warpAndCut(frames[i], mean))
        }
        return out
    }

    /**
     * Aligns [bitmap] using [kp] (smoothed against recent frames) and returns the 88x88 grayscale
     * ROI as row-major floats in [0,1] — the exact input the ONNX graph expects.
     */
    fun alignAndCrop(bitmap: Bitmap, kp: Keypoints): FloatArray {
        history.addLast(kp)
        while (history.size > WINDOW_MARGIN + 1) history.removeFirst()
        return warpAndCut(bitmap, smooth())
    }

    /**
     * Warps [bitmap] onto the mean-face reference using [pts] and cuts the 88x88 grayscale ROI.
     *
     * With 4 points the transform is fitted on the stable eye/nose/mouth references (auto_avsr);
     * with 68 points it is fitted on the full mean face and the crop is centered on the mouth
     * contour (points 48..67), which is what CNVSRC does.
     */
    private fun warpAndCut(bitmap: Bitmap, pts: Array<FloatArray>): FloatArray {
        val use68 = pts.size == 68
        val reference = if (use68) REFERENCE_68 else STABLE_REFERENCE
        val m = estimateSimilarity(pts, reference)

        // Warp the frame into the 256x256 reference space (the mean face is defined in this space).
        val warped = Bitmap.createBitmap(REF_SIZE, REF_SIZE, Bitmap.Config.ARGB_8888)
        Canvas(warped).drawBitmap(bitmap, m, Paint(Paint.FILTER_BITMAP_FLAG))

        // Center the crop on where the mouth ACTUALLY lands after the warp — the reference mouth
        // position is only a good approximation for the 4-point fit; with a 68-point whole-face fit
        // the mouth can sit noticeably off it, which visibly mis-frames the ROI.
        val flat = FloatArray(pts.size * 2)
        for (i in pts.indices) { flat[i * 2] = pts[i][0]; flat[i * 2 + 1] = pts[i][1] }
        m.mapPoints(flat)
        val from = if (use68) 48 else 3
        val to = if (use68) 68 else 4
        var cx = 0f; var cy = 0f
        for (i in from until to) { cx += flat[i * 2]; cy += flat[i * 2 + 1] }
        cx /= (to - from); cy /= (to - from)
        val half = CROP / 2
        val left = (cx - half).toInt().coerceIn(0, REF_SIZE - CROP)
        val top = (cy - half).toInt().coerceIn(0, REF_SIZE - CROP)
        val patch = Bitmap.createBitmap(warped, left, top, CROP, CROP)

        // Center-crop 96 -> 88 (the model's own transform does this) and grayscale.
        val off = (CROP - OUT) / 2
        val pixels = IntArray(OUT * OUT)
        patch.getPixels(pixels, 0, OUT, off, off, OUT, OUT)
        val out = FloatArray(OUT * OUT)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // torchvision Grayscale uses the luma weights
            out[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
        }
        warped.recycle(); patch.recycle()
        return out
    }

    /** Mean of the recent keypoints, re-centered on the current frame's centroid. */
    private fun smooth(): Array<FloatArray> {
        val n = history.size
        val np = history.last().pts.size // 4 (English) or 68 (Chinese)
        val mean = Array(np) { FloatArray(2) }
        for (k in history) {
            if (k.pts.size != np) continue // ignore stale frames from a previous language
            for (i in 0 until np) {
                mean[i][0] += k.pts[i][0] / n
                mean[i][1] += k.pts[i][1] / n
            }
        }
        val cur = history.last().pts
        var cmx = 0f; var cmy = 0f; var smx = 0f; var smy = 0f
        for (i in 0 until np) {
            cmx += cur[i][0] / np; cmy += cur[i][1] / np
            smx += mean[i][0] / np; smy += mean[i][1] / np
        }
        val dx = cmx - smx; val dy = cmy - smy
        for (i in 0 until np) { mean[i][0] += dx; mean[i][1] += dy }
        return mean
    }

    /**
     * Least-squares similarity transform (rotation + uniform scale + translation) mapping [src]
     * onto [dst] — the equivalent of cv2.estimateAffinePartial2D for 4 correspondences (Umeyama).
     */
    private fun estimateSimilarity(src: Array<FloatArray>, dst: Array<FloatArray>): Matrix {
        val n = src.size
        var sx = 0f; var sy = 0f; var dx = 0f; var dy = 0f
        for (i in 0 until n) {
            sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]
        }
        sx /= n; sy /= n; dx /= n; dy /= n

        var sxx = 0f  // sum of src·dst rotational terms
        var sxy = 0f
        var srcVar = 0f
        for (i in 0 until n) {
            val ax = src[i][0] - sx; val ay = src[i][1] - sy
            val bx = dst[i][0] - dx; val by = dst[i][1] - dy
            sxx += ax * bx + ay * by
            sxy += ax * by - ay * bx
            srcVar += ax * ax + ay * ay
        }
        if (srcVar < 1e-6f) srcVar = 1e-6f
        val a = sxx / srcVar   // s*cos(theta)
        val b = sxy / srcVar   // s*sin(theta)

        val m = Matrix()
        m.setValues(
            floatArrayOf(
                a, -b, dx - (a * sx - b * sy),
                b, a, dy - (b * sx + a * sy),
                0f, 0f, 1f,
            )
        )
        return m
    }
}

/** Euclidean distance helper. */
internal fun dist(a: FloatArray, b: FloatArray): Float {
    val dx = a[0] - b[0]; val dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)
}
