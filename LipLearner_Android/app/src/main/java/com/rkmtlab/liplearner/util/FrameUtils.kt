package com.rkmtlab.liplearner.util

import android.graphics.Bitmap
import android.graphics.Rect
import androidx.core.graphics.scale

/** Geometry + pixel helpers shared by the vision and ML layers. */
object FrameUtils {

    const val FRAME_SIZE = 88

    /**
     * Crops [src] to [rect] (clamped to image bounds), scales to 88x88, and returns a row-major
     * grayscale FloatArray in [0,1].
     *
     * Mirrors the iOS pipeline: crop a lip-centered square, resize to 88x88, then
     * `gray = floor((r+g+b)/3)/255` (see UIImage.getPixelBuffer in CameraViewController.swift).
     */
    fun cropToGrayFloats(src: Bitmap, rect: Rect): FloatArray {
        val clamped = Rect(
            rect.left.coerceIn(0, src.width - 1),
            rect.top.coerceIn(0, src.height - 1),
            rect.right.coerceIn(1, src.width),
            rect.bottom.coerceIn(1, src.height),
        )
        val w = (clamped.right - clamped.left).coerceAtLeast(1)
        val h = (clamped.bottom - clamped.top).coerceAtLeast(1)
        val cropped = Bitmap.createBitmap(src, clamped.left, clamped.top, w, h)
        val scaled = cropped.scale(FRAME_SIZE, FRAME_SIZE)

        val pixels = IntArray(FRAME_SIZE * FRAME_SIZE)
        scaled.getPixels(pixels, 0, FRAME_SIZE, 0, 0, FRAME_SIZE, FRAME_SIZE)
        val out = FloatArray(FRAME_SIZE * FRAME_SIZE)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // integer average, floored, then /255 — matches the iOS getPixelBuffer()
            out[i] = ((r + g + b) / 3) / 255f
        }
        if (cropped != src) cropped.recycle()
        if (scaled != cropped) scaled.recycle()
        return out
    }
}
