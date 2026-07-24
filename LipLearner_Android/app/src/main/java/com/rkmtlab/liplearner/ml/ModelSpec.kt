package com.rkmtlab.liplearner.ml

import android.content.Context

/**
 * Describes one selectable on-device encoder. All current models emit a 500-D vector from 88x88
 * grayscale lip frames; they differ in backbone, size, and whether they need a fixed clip length.
 *
 * `fixedFrames` != null  → the ONNX graph has a fixed time axis; the client resamples every clip to
 *                          exactly that many frames (mpc001 models, trained on 29-frame LRW clips).
 * `fixedFrames` == null  → variable length in [minFrames, maxFrames] (the original LipLearner GRU
 *                          encoder).
 *
 * Model asset & preprocessing (grayscale [0,1]) are uniform, so switching models only swaps the
 * ONNX session and the per-model learned data.
 */
data class ModelSpec(
    val id: String,
    val displayName: String,
    val asset: String,
    val embedDim: Int = 500,
    val frameSize: Int = 88,
    val fixedFrames: Int? = null,
    val minFrames: Int = 10,
    val maxFrames: Int = 128,
) {
    fun isAvailable(context: Context): Boolean =
        runCatching { context.assets.list("")?.contains(asset) == true }.getOrDefault(false)
}

object ModelRegistry {

    /** All known models. Only those whose asset is bundled are shown to the user. */
    val ALL: List<ModelSpec> = listOf(
        ModelSpec(
            id = "liplearner",
            displayName = "LipLearner (ResNet18+BiGRU, contrastive)",
            asset = "lip_encoder.onnx",
            fixedFrames = null, minFrames = 10, maxFrames = 128,
        ),
        ModelSpec(
            id = "mpc001_resnet18_mstcn",
            displayName = "mpc001 ResNet18+MS-TCN (88.9%)",
            asset = "mpc001_resnet18_mstcn.onnx",
            fixedFrames = 29,
        ),
        ModelSpec(
            id = "mpc001_snv1x_dsmstcn3x",
            displayName = "mpc001 ShuffleNet1x+DS-MSTCN (85.3%)",
            asset = "mpc001_snv1x_dsmstcn3x.onnx",
            fixedFrames = 29,
        ),
        ModelSpec(
            id = "mpc001_snv1x_tcn1x",
            displayName = "mpc001 ShuffleNet1x+TCN (82.7%)",
            asset = "mpc001_snv1x_tcn1x.onnx",
            fixedFrames = 29,
        ),
        ModelSpec(
            id = "mpc001_snv05x_tcn1x",
            displayName = "mpc001 ShuffleNet0.5x+TCN (79.9%, lightest)",
            asset = "mpc001_snv05x_tcn1x.onnx",
            fixedFrames = 29,
        ),
    )

    fun available(context: Context): List<ModelSpec> = ALL.filter { it.isAvailable(context) }

    fun byId(id: String?): ModelSpec? = ALL.firstOrNull { it.id == id }

    /**
     * Preferred default: the original LipLearner model if present, otherwise the lightest available
     * model (smallest footprint) so first launch is fast and memory-safe. The user can switch to a
     * heavier/more-accurate model from the picker.
     */
    fun default(context: Context): ModelSpec? {
        val avail = available(context)
        avail.firstOrNull { it.id == "liplearner" }?.let { return it }
        return avail.firstOrNull { it.id == "mpc001_snv05x_tcn1x" } ?: avail.firstOrNull()
    }
}
