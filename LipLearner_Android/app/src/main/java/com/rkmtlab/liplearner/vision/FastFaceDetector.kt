package com.rkmtlab.liplearner.vision

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector

/**
 * Lightweight BlazeFace detector used by the open-vocabulary VSR path.
 *
 * The 468-point Face Mesh costs ~25-30ms per frame, which caps the analysis stream around 20fps —
 * below the 25fps the VSR model expects, so frames get dropped and speech tempo is distorted.
 * All the alignment needs is 4 stable keypoints, and BlazeFace provides exactly the ones the
 * reference (desktop) pipeline uses: right eye, left eye, nose tip, mouth center.
 */
class FastFaceDetector(context: Context, modelAsset: String = "blaze_face_short_range.tflite") {

    /** [rightEye, leftEye, nose, mouthCenter] in image pixels, or null when no face is found. */
    class Result(val alignPoints: Array<FloatArray>?, val mouthX: Float, val mouthY: Float, val faceWidth: Float)

    private val detector: FaceDetector

    init {
        val base = BaseOptions.builder().setModelAssetPath(modelAsset).build()
        val options = FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setMinDetectionConfidence(0.5f)
            .build()
        detector = FaceDetector.createFromOptions(context, options)
    }

    fun detect(bitmap: Bitmap): Result {
        val res = detector.detect(BitmapImageBuilder(bitmap).build())
        val det = res.detections().maxByOrNull { it.boundingBox().width() * it.boundingBox().height() }
            ?: return Result(null, 0f, 0f, 0f)

        // BlazeFace keypoint order: 0=right eye, 1=left eye, 2=nose tip, 3=mouth, 4/5=ear tragions.
        val kp = det.keypoints().orElse(null) ?: return Result(null, 0f, 0f, 0f)
        if (kp.size < 4) return Result(null, 0f, 0f, 0f)

        val w = bitmap.width
        val h = bitmap.height
        fun pt(i: Int) = floatArrayOf(kp[i].x() * w, kp[i].y() * h)

        val points = arrayOf(pt(0), pt(1), pt(2), pt(3))
        val box = det.boundingBox()
        return Result(points, points[3][0], points[3][1], box.width())
    }

    fun close() = detector.close()
}
