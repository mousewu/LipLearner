package com.rkmtlab.liplearner.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.rkmtlab.liplearner.util.FrameUtils
import com.rkmtlab.liplearner.vision.LipLandmarker
import java.util.concurrent.Executors

/**
 * CameraX front-camera pipeline (Android replacement for AVCaptureSession + Vision). Each analyzed
 * frame is converted to an upright, mirrored Bitmap, passed to [LipLandmarker], cropped to the
 * lip ROI and delivered as an 88x88 grayscale FloatArray.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val landmarker: LipLandmarker,
    private val onLipFrame: (gray: FloatArray, mod: Float, hasFace: Boolean, lipPreview: Bitmap?) -> Unit,
) {
    /**
     * Set to true to also produce mean-face-aligned 88x88 ROIs for the open-vocabulary VSR model,
     * which was trained on aligned crops (the plain lip square used by the few-shot encoder is
     * off-distribution for it). Kept off by default because the warp costs extra work per frame.
     */
    @Volatile var alignedOutput: Boolean = false

    /**
     * When true the free-VSR path needs the 68-point layout (Chinese/CNVSRC), which only the Face
     * Mesh provides; otherwise the much cheaper 4-keypoint BlazeFace detector is used (English).
     */
    @Volatile var alignedNeeds68: Boolean = false
    /** Receives the aligned ROI when [alignedOutput] is on. */
    @Volatile var onAlignedFrame: ((FloatArray) -> Unit)? = null

    /**
     * When set, raw (unaligned) frames plus their keypoints are handed over instead of a
     * pre-aligned ROI, so the caller can run the same **centered-window** alignment the video-file
     * path uses. Streaming alignment can only smooth over past frames, which is measurably worse.
     */
    @Volatile var onRawFaceFrame: ((Bitmap, com.rkmtlab.liplearner.vision.FaceAligner.Keypoints) -> Unit)? = null

    private val aligner = com.rkmtlab.liplearner.vision.FaceAligner()
    private var fastDetector: com.rkmtlab.liplearner.vision.FastFaceDetector? = null

    fun resetAligner() = aligner.reset()

    /** Lazily created BlazeFace detector for the free-VSR path (much cheaper than Face Mesh). */
    private fun fastDetector(): com.rkmtlab.liplearner.vision.FastFaceDetector =
        fastDetector ?: com.rkmtlab.liplearner.vision.FastFaceDetector(context).also { fastDetector = it }

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null

    fun start() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            provider = future.get()
            bind()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bind() {
        val cameraProvider = provider ?: return
        cameraProvider.unbindAll()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        // 720p is plenty for an 88x88 mouth ROI and keeps the per-frame cost (MediaPipe + warp) low
        // enough to sustain a high analysis frame rate — dropped frames distort speech tempo.
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            // 1080p made the per-frame Bitmap rotate/mirror cost ~24ms and capped the stream at
            // ~19fps — below the 25fps the VSR models expect. 720p keeps plenty of detail for an
            // 88x88 mouth ROI while restoring ~30fps.
            .setTargetResolution(android.util.Size(720, 1280))
            .build()
        analysis.setAnalyzer(analysisExecutor, ::analyze)

        cameraProvider.bindToLifecycle(
            lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis,
        )
    }

    private var profN = 0
    private var profConvert = 0L
    private var profDetect = 0L
    private var profAlign = 0L

    private fun analyze(image: ImageProxy) {
        try {
            val t0 = System.nanoTime()
            val bitmap = image.toBitmap()
            val upright = orientUpright(bitmap, image.imageInfo.rotationDegrees, mirror = true)
            val t1 = System.nanoTime()

            // Free-VSR path: BlazeFace only (~5ms) instead of the 468-point Face Mesh (~25-30ms),
            // so the analysis stream can sustain the 25fps the VSR model was trained on.
            if (alignedOutput) {
                // Chinese needs 68 landmarks (Face Mesh); English only 4 (BlazeFace, ~5ms).
                val pts0: Array<FloatArray>?
                val faceW: Float
                if (alignedNeeds68) {
                    val lm = landmarker.detect(upright)
                    pts0 = lm.points68
                    faceW = (lm.cropRect.width().toFloat() / 0.75f).coerceAtLeast(96f)
                } else {
                    val det = fastDetector().detect(upright)
                    pts0 = det.alignPoints
                    faceW = det.faceWidth
                }
                val det = com.rkmtlab.liplearner.vision.FastFaceDetector.Result(pts0, 0f, 0f, faceW)
                val t2f = System.nanoTime()
                det.alignPoints?.let { pts ->
                    val kp = com.rkmtlab.liplearner.vision.FaceAligner.Keypoints(pts)
                    val sink = onRawFaceFrame
                    if (sink != null) {
                        // Keep only a generous box around the face (a full frame would cost ~3.7MB
                        // each), shifting the keypoints into its coordinates. The box is derived
                        // from the keypoints' own bounding box: a centroid-based box is wrong for
                        // the 68-point set, whose centroid sits near the nose, and it cropped the
                        // mouth right out of the saved region.
                        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
                        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
                        for (p in pts) {
                            if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0]
                            if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1]
                        }
                        // Pad generously so the warp never samples outside the saved region.
                        val pad = maxOf(maxX - minX, maxY - minY).coerceAtLeast(96f) * 0.8f
                        val l = (minX - pad).toInt().coerceIn(0, upright.width - 1)
                        val t = (minY - pad).toInt().coerceIn(0, upright.height - 1)
                        val r = (maxX + pad).toInt().coerceIn(l + 1, upright.width)
                        val b = (maxY + pad).toInt().coerceIn(t + 1, upright.height)
                        val sub = Bitmap.createBitmap(upright, l, t, r - l, b - t)
                        val shifted = Array(4) { i -> floatArrayOf(pts[i][0] - l, pts[i][1] - t) }
                        sink(sub, com.rkmtlab.liplearner.vision.FaceAligner.Keypoints(shifted))
                    }
                    // Always produce a streaming ROI for the live preview thumbnail.
                    runCatching { aligner.alignAndCrop(upright, kp) }.getOrNull()?.let { aligned ->
                        if (sink == null) onAlignedFrame?.invoke(aligned)
                        onLipFrame(aligned, 0f, true, grayToBitmap(aligned))
                    }
                }
                val t3 = System.nanoTime()
                profConvert += (t1 - t0); profDetect += (t2f - t1); profAlign += (t3 - t2f)
                if (++profN % 30 == 0) {
                    android.util.Log.i(
                        "LipLearner",
                        "per-frame ms: convert=${profConvert / profN / 1_000_000} " +
                            "detect=${profDetect / profN / 1_000_000} align=${profAlign / profN / 1_000_000}"
                    )
                }
                return
            }

            val lip = landmarker.detect(upright)
            val t2 = System.nanoTime()
            val gray = FrameUtils.cropToGrayFloats(upright, lip.cropRect)

            val preview = try {
                val r = lip.cropRect
                if (lip.hasFace && r.width() > 0 && r.height() > 0 &&
                    r.left >= 0 && r.top >= 0 && r.right <= upright.width && r.bottom <= upright.height
                ) Bitmap.createBitmap(upright, r.left, r.top, r.width(), r.height()) else null
            } catch (e: Exception) { null }
            onLipFrame(gray, lip.mod, lip.hasFace, preview)
        } catch (e: Exception) {
            // drop frame
        } finally {
            image.close()
        }
    }

    /** Renders an 88x88 grayscale ROI (values in [0,1]) as a Bitmap for the preview thumbnail. */
    private fun grayToBitmap(gray: FloatArray): Bitmap {
        val s = com.rkmtlab.liplearner.vision.FaceAligner.OUT
        val px = IntArray(s * s)
        for (i in px.indices) {
            val v = (gray[i].coerceIn(0f, 1f) * 255).toInt()
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return Bitmap.createBitmap(px, s, s, Bitmap.Config.ARGB_8888)
    }

    private fun orientUpright(src: Bitmap, rotationDegrees: Int, mirror: Boolean): Bitmap {
        if (rotationDegrees == 0 && !mirror) return src
        val m = Matrix()
        m.postRotate(rotationDegrees.toFloat())
        if (mirror) m.postScale(-1f, 1f)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    fun stop() {
        provider?.unbindAll()
        fastDetector?.close()
        fastDetector = null
        analysisExecutor.shutdown()
    }
}
