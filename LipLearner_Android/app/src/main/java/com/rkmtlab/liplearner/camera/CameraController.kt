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
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analysisExecutor, ::analyze)

        cameraProvider.bindToLifecycle(
            lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis,
        )
    }

    private fun analyze(image: ImageProxy) {
        try {
            val bitmap = image.toBitmap()
            val upright = orientUpright(bitmap, image.imageInfo.rotationDegrees, mirror = true)
            val lip = landmarker.detect(upright)
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

    private fun orientUpright(src: Bitmap, rotationDegrees: Int, mirror: Boolean): Bitmap {
        if (rotationDegrees == 0 && !mirror) return src
        val m = Matrix()
        m.postRotate(rotationDegrees.toFloat())
        if (mirror) m.postScale(-1f, 1f)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    fun stop() {
        provider?.unbindAll()
        analysisExecutor.shutdown()
    }
}
