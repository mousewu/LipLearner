package com.rkmtlab.liplearner.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.hypot

/**
 * Wraps MediaPipe Face Landmarker (Face Mesh). Android replacement for the iOS Vision face
 * detection + landmark pipeline.
 *
 * For every frame it produces a [LipResult]: the lip-centered square crop rect (in image pixels)
 * and the Mouth-Opening-Degree (MOD = inner-lip height / width), used for SSAD, EOS and KWS gating
 * exactly like the iOS controller.
 */
class LipLandmarker(context: Context, modelAsset: String = "face_landmarker.task") {

    data class LipResult(val cropRect: Rect, val mod: Float, val hasFace: Boolean)

    // MediaPipe Face Mesh landmark indices.
    private companion object {
        const val UPPER_INNER_LIP = 13
        const val LOWER_INNER_LIP = 14
        const val LEFT_MOUTH_CORNER = 78
        const val RIGHT_MOUTH_CORNER = 308
        // A representative set of lip contour points for centering the crop.
        val LIP_POINTS = intArrayOf(
            61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 88, 318, 95, 324,
        )
    }

    private val landmarker: FaceLandmarker

    init {
        val base = BaseOptions.builder().setModelAssetPath(modelAsset).build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .build()
        landmarker = FaceLandmarker.createFromOptions(context, options)
    }

    /** Runs detection on [bitmap] (already upright & mirrored as desired). */
    fun detect(bitmap: Bitmap): LipResult {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: FaceLandmarkerResult = landmarker.detect(mpImage)
        val faces = result.faceLandmarks()
        if (faces.isEmpty()) {
            return LipResult(Rect(0, 0, bitmap.width, bitmap.height), 0f, false)
        }
        val lm = faces[0]
        val w = bitmap.width
        val h = bitmap.height

        fun px(i: Int) = lm[i].x() * w
        fun py(i: Int) = lm[i].y() * h

        // MOD = inner-lip vertical gap / mouth width
        val vertical = hypot((px(UPPER_INNER_LIP) - px(LOWER_INNER_LIP)).toDouble(),
            (py(UPPER_INNER_LIP) - py(LOWER_INNER_LIP)).toDouble())
        val horizontal = hypot((px(LEFT_MOUTH_CORNER) - px(RIGHT_MOUTH_CORNER)).toDouble(),
            (py(LEFT_MOUTH_CORNER) - py(RIGHT_MOUTH_CORNER)).toDouble()).coerceAtLeast(1.0)
        val mod = (vertical / horizontal).toFloat()

        // Face width from the full landmark bounding box (proxy for iOS faceBounds.width).
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        for (p in lm) {
            val x = p.x() * w
            if (x < minX) minX = x
            if (x > maxX) maxX = x
        }
        val faceWidth = (maxX - minX).coerceAtLeast(1f)

        // Lip center from the lip contour points.
        var sx = 0f; var sy = 0f
        for (i in LIP_POINTS) { sx += px(i); sy += py(i) }
        val cx = sx / LIP_POINTS.size
        val cy = sy / LIP_POINTS.size

        val cropSize = faceWidth * 0.75f // matches iOS cropSize = faceBounds.width * 0.75
        val half = cropSize / 2f
        val rect = Rect(
            (cx - half).toInt(),
            (cy - half).toInt(),
            (cx + half).toInt(),
            (cy + half).toInt(),
        )
        return LipResult(rect, mod, true)
    }

    fun close() = landmarker.close()
}
